#include "avi/engine.h"
#include "avi/speculative.h"
#include "avi/ops.h"
#include <mpi.h>
#include <algorithm>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

namespace avi {
int64_t Engine::greedy(Tensor values){
  int64_t token=0;
  if(rank_==0){
    TORCH_CHECK(at::isfinite(values).all().item<bool>(),"Nonfinite speculative logits");
    token=values.argmax(-1).item<int64_t>();
  }
  MPI_Bcast(&token,1,MPI_INT64_T,0,MPI_COMM_WORLD);return token;
}

Tensor Engine::local_candidates(Tensor normalized){
  auto values=linear(normalized,"lm_head").to(at::kFloat);
  auto top=at::max(values,-1);
  auto ids=std::get<1>(top);
  if(options_.tp_lm_head&&world_>1)ids=ids+rank_*values.size(-1);
  return at::stack({std::get<0>(top).to(at::kDouble),ids.to(at::kDouble)},-1).contiguous();
}
std::vector<int64_t> Engine::gather_candidates(Tensor candidates){
  int ranks=options_.tp_lm_head?world_:1;auto rows=candidates.size(0);
  auto gathered=at::empty({ranks,rows,2},candidates.options());
  if(ranks>1){
    auto status=ncclAllGather(candidates.data_ptr(),gathered.data_ptr(),candidates.numel(),ncclDouble,comm_,at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(status==ncclSuccess,ncclGetErrorString(status));
  }else gathered.select(0,0).copy_(candidates);
  std::vector<int64_t> result(rows);
  if(rank_==0){
    auto cpu=gathered.to(at::kCPU);auto data=cpu.data_ptr<double>();
    for(int64_t row=0;row<rows;++row){
      double score=-INFINITY;int64_t id=0;
      for(int rank=0;rank<ranks;++rank){
        auto offset=(rank*rows+row)*2;double value=data[offset];int64_t token=int64_t(data[offset+1]);
        TORCH_CHECK(std::isfinite(value),"Nonfinite local vocabulary maximum");
        if(value>score||(value==score&&token<id)){score=value;id=token;}
      }
      result[row]=id;
    }
  }
  MPI_Bcast(result.data(),int(rows),MPI_INT64_T,0,MPI_COMM_WORLD);return result;
}

Tensor Engine::mtp_step(Tensor embeddings,Tensor positions,Tensor hidden,bool single_decode){
  // MTP consumes the next token embedding and the preceding normalized hidden.
  // The MTP output norm is already applied; do not apply the target norm again.
  struct Restore {bool& v;bool old;~Restore(){v=old;}} restore{decode_mode_,decode_mode_};
  decode_mode_=single_decode;
  auto x=linear(at::cat({norm(embeddings,"mtp.pre_fc_norm_embedding"),norm(hidden,"mtp.pre_fc_norm_hidden")},-1),"mtp.fc");
  const std::string p="mtp.layers.0";
  x=x+full_attention(norm(x,p+".input_layernorm"),positions,0,p+".self_attn",{},true,&drafts_[active_].state);
  auto n=norm(x,p+".post_attention_layernorm");
  auto gate=options_.optimized?fused_swiglu(linear(n,p+".mlp.gate_up")):at::silu(linear(n,p+".mlp.gate_proj"))*linear(n,p+".mlp.up_proj");
  x=x+linear(gate,p+".mlp.down_proj",true);
  return norm(x,"mtp.norm");
}

struct DraftGraph {
  std::unique_ptr<at::cuda::CUDAGraph> graph;
  Tensor ids,positions,input,output,keys,values,candidates;
  int capacity=0;
  ~DraftGraph(){graph.reset();}
};
void Engine::release_draft(int id){
  auto it=draft_graphs_.find(id);
  if(it!=draft_graphs_.end()){
    // Two idle shapes at most. Retain KV allocations whose pointers are captured.
    if(draft_graph_pool_.size()>=2)draft_graph_pool_.erase(draft_graph_pool_.begin());
    draft_graph_pool_[it->second->capacity]=it->second;draft_graphs_.erase(it);
  }
  drafts_.erase(id);
}
Tensor Engine::draft_one(int64_t token,int64_t position,Tensor hidden){
  draft_candidates_=Tensor();
  auto& state=drafts_[active_].state;
  if(!draft_graph_enabled_)return mtp_step(embed(at::full({1},token,decode_offset_.options())),at::full({3,1},position,decode_offset_.options()),hidden);
  TORCH_CHECK(state.length<session_capacity(),"MTP graph KV capacity exceeded");
  decode_offset_.fill_(state.length);
  auto& holder=draft_graphs_[active_];if(!holder)holder=std::make_shared<DraftGraph>();auto& g=*holder;
  if(!g.graph){
    g.ids=at::full({1},token,decode_offset_.options());g.positions=at::full({3,1},position,decode_offset_.options());g.input=hidden.clone();
    C10_CUDA_CHECK(cudaDeviceSynchronize());auto stream=c10::cuda::getStreamFromPool(false,device_);
    {
      c10::cuda::CUDAStreamGuard guard(stream);
      // Only the pending KV slot is overwritten; no accepted prefix state changes.
      for(int i=0;i<2;++i){g.output=mtp_step(embed(g.ids),g.positions,g.input,true);g.candidates=local_candidates(g.output);}
      g.keys=state.key;g.values=state.value;g.capacity=session_capacity();
      C10_CUDA_CHECK(cudaStreamSynchronize(stream));
      g.graph=std::make_unique<at::cuda::CUDAGraph>();g.graph->capture_begin();
      g.output=mtp_step(embed(g.ids),g.positions,g.input,true);g.candidates=local_candidates(g.output);g.graph->capture_end();
    }
  }else {g.ids.fill_(token);g.positions.fill_(position);g.input.copy_(hidden);}
  g.graph->replay();draft_candidates_=g.candidates;++state.length;return g.output;
}

struct VerifyGraph {
  std::unique_ptr<at::cuda::CUDAGraph> graph;
  Tensor embeddings,positions,hidden,candidates;
  std::vector<Tensor> trajectories,conv_inputs;
  ~VerifyGraph(){graph.reset();}
};

std::pair<Tensor,Tensor> Engine::verify_graph(Tensor embeddings,Tensor positions,int consumed){
  TORCH_CHECK(verifying_&&embeddings.size(0)==options_.mtp_tokens+1&&
      consumed>=0&&consumed+embeddings.size(0)<=session_capacity(),"Invalid verification graph request");
  decode_offset_.fill_(consumed);
  auto& holder=verify_graphs_[active_];if(!holder)holder=std::make_shared<VerifyGraph>();
  auto& graph=*holder;
  if(!graph.graph){
    graph.embeddings=embeddings.clone();graph.positions=positions.clone();
    std::vector<std::pair<Tensor,Tensor>> saved;
    for(const auto& state:states_)saved.emplace_back(state.conv.defined()?state.conv.clone():Tensor(),
                                                    state.recurrent.defined()?state.recurrent.clone():Tensor());
    auto restore=[&](){for(size_t i=0;i<states_.size();++i)if(saved[i].first.defined()){
      states_[i].conv.copy_(saved[i].first);states_[i].recurrent.copy_(saved[i].second);
    }};
    struct Reset {bool& flag;~Reset(){flag=false;}} reset{verifying_graph_};
    verifying_graph_=true;
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    auto stream=c10::cuda::getStreamFromPool(false,device_);
    {
      c10::cuda::CUDAStreamGuard guard(stream);
      graph.hidden=step(graph.embeddings,graph.positions);
      graph.candidates=local_candidates(norm(graph.hidden,"model.language_model.norm"));
      restore();C10_CUDA_CHECK(cudaStreamSynchronize(stream));
      graph.graph=std::make_unique<at::cuda::CUDAGraph>();graph.graph->capture_begin();
      graph.hidden=step(graph.embeddings,graph.positions);
      graph.candidates=local_candidates(norm(graph.hidden,"model.language_model.norm"));
      graph.graph->capture_end();
      // Capture records operations; the warmup state was restored before capture.
      graph.trajectories=verify_recurrent_;graph.conv_inputs=verify_conv_;
    }
    ++verify_graph_builds_;
  }else{
    graph.embeddings.copy_(embeddings);graph.positions.copy_(positions);
  }
  // Other eager verification shapes may have replaced these host tensor handles.
  verify_recurrent_=graph.trajectories;verify_conv_=graph.conv_inputs;
  graph.graph->replay();++verify_graph_replays_;
  return {graph.hidden,graph.candidates};
}

void Engine::advance_draft(Tensor embeddings,Tensor positions,Tensor target_hidden){
  auto normalized=norm(target_hidden,"model.language_model.norm");
  auto& draft=drafts_[active_];int64_t count=embeddings.size(0);
  if(draft_graph_enabled_&&!draft.hidden.defined()){
    auto idle=draft_graph_pool_.find(session_capacity());
    if(idle!=draft_graph_pool_.end()){
      draft_graphs_[active_]=idle->second;draft.state.key=idle->second->keys;draft.state.value=idle->second->values;
      draft.state.length=0;draft_graph_pool_.erase(idle);
    }
  }
  if(draft.hidden.defined()){
    auto previous=count==1?draft.hidden:at::cat({draft.hidden,normalized.narrow(0,0,count-1)},0);
    mtp_step(embeddings,positions,previous);
  }else if(count>1){
    mtp_step(embeddings.narrow(0,1,count-1),positions.narrow(1,1,count-1),normalized.narrow(0,0,count-1));
  }
  draft.hidden=normalized.narrow(0,count-1,1).clone();
}

SpeculativeResult Engine::speculate(int64_t pending,int64_t position,int consumed,int budget,
    const std::vector<int64_t>& eos,const std::function<int64_t(Tensor)>& select){
  TORCH_CHECK(options_.mtp_tokens>0&&!verifying_&&budget>0,"Invalid speculative invocation");
  auto& draft=drafts_[active_];TORCH_CHECK(draft.hidden.defined(),"MTP prompt state not initialized");
  int count=std::min(options_.mtp_tokens,budget-1);
  TORCH_CHECK(consumed>=0&&consumed+count+1<=session_capacity(),"Speculative capacity exceeded");
  const int draft_old=draft.state.length;
  auto previous=draft.hidden;
  std::vector<int64_t> inputs{pending},proposals;
  auto hidden=previous;
  for(int i=0;i<count;++i){
    hidden=draft_one(inputs.back(),position+i,hidden);
    int64_t token=gather_candidates(draft_candidates_.defined()?draft_candidates_:local_candidates(hidden))[0];proposals.push_back(token);inputs.push_back(token);
    if(std::find(eos.begin(),eos.end(),token)!=eos.end())break;
  }
  auto host=at::empty({int64_t(inputs.size())},at::TensorOptions().dtype(at::kLong));
  std::copy(inputs.begin(),inputs.end(),host.data_ptr<int64_t>());
  auto embeddings=embed(host.to(decode_offset_.device()));
  auto positions=(at::arange(int64_t(inputs.size()),decode_offset_.options())+position).unsqueeze(0).repeat({3,1});
  for(const auto& state:states_)if(state.key.defined()){TORCH_CHECK(state.length==consumed,"Target KV length mismatch");}
  struct Restore {bool& v;~Restore(){v=false;}} restore{verifying_};
  verifying_=true;
  Tensor target_hidden,captured_candidates;
  if(options_.mtp_verify_graph&&!select&&inputs.size()==size_t(options_.mtp_tokens+1)){
    auto result=verify_graph(embeddings,positions,consumed);target_hidden=result.first;captured_candidates=result.second;
  }else target_hidden=step(embeddings,positions);
  verifying_=false;
  Tensor values;std::vector<int64_t> choices;
  if(select)values=logits(target_hidden);
  else choices=gather_candidates(captured_candidates.defined()?captured_candidates:local_candidates(norm(target_hidden,"model.language_model.norm")));
  GreedyVerification decision(proposals,budget,eos);
  for(int i=0;!decision.done;++i){
    TORCH_CHECK(i<target_hidden.size(0),"Verification row out of range");
    decision.observe(select?select(values.narrow(0,i,1)):choices.at(i));
  }
  int committed=decision.committed_inputs();
  TORCH_CHECK(committed<=int(inputs.size()),"Invalid speculative commit");
  for(size_t layer=0;layer<states_.size();++layer){
    auto& state=states_[layer];
    if(state.key.defined())state.length=consumed+committed;
    if(state.recurrent.defined()){
      state.recurrent.copy_(verify_recurrent_[layer].select(0,committed-1));
      state.conv.copy_(verify_conv_[layer].narrow(-1,committed,state.conv.size(-1)));
      verify_conv_[layer]=Tensor();
    }
  }
  // Discard hypothetical MTP KV and rebuild accepted positions using verified
  // target hidden states. This is one draft layer, never a target-model replay.
  draft.state.length=draft_old;
  auto normalized=norm(target_hidden.narrow(0,0,committed),"model.language_model.norm");
  auto verified_previous=committed==1?previous:at::cat({previous,normalized.narrow(0,0,committed-1)},0);
  mtp_step(embeddings.narrow(0,0,committed),positions.narrow(1,0,committed),verified_previous);
  draft.hidden=normalized.narrow(0,committed-1,1).clone();
  return {decision.tokens,committed,int(proposals.size()),decision.accepted};
}
}
