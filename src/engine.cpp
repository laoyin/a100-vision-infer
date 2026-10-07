#include "avi/engine.h"
#include "avi/ops.h"
#include "avi/tilelang.h"
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <openssl/evp.h>
#include <iomanip>
#include <sstream>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <cmath>
#include <limits>
#include <functional>
#include <algorithm>
namespace avi {
struct ProfileEvent {
  cudaEvent_t start=nullptr,end=nullptr;std::string label;
  ~ProfileEvent(){if(start)cudaEventDestroy(start);if(end)cudaEventDestroy(end);}
};
size_t Engine::profile_begin(const std::string& label){
  if(!options_.profile_kernels)return std::numeric_limits<size_t>::max();
  auto event=std::make_shared<ProfileEvent>();event->label=label;
  C10_CUDA_CHECK(cudaEventCreate(&event->start));C10_CUDA_CHECK(cudaEventCreate(&event->end));
  C10_CUDA_CHECK(cudaEventRecord(event->start,at::cuda::getCurrentCUDAStream()));
  profile_events_.push_back(event);return profile_events_.size()-1;
}
void Engine::profile_end(size_t index){
  if(index!=std::numeric_limits<size_t>::max())C10_CUDA_CHECK(cudaEventRecord(profile_events_.at(index)->end,at::cuda::getCurrentCUDAStream()));
}
json Engine::profile_report(){
  json report=json::object();if(!options_.profile_kernels)return report;
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for(const auto& event:profile_events_){
    float ms=0;C10_CUDA_CHECK(cudaEventElapsedTime(&ms,event->start,event->end));
    auto& row=report[event->label];if(row.is_null())row={{"calls",0},{"cuda_ms",0.0}};
    row["calls"]=row["calls"].get<int>()+1;row["cuda_ms"]=row["cuda_ms"].get<double>()+ms;
  }
  profile_events_.clear();return report;
}
json read_json(const std::string& file) { std::ifstream f(file); TORCH_CHECK(f.good(),"Cannot open ",file); json j; f>>j; return j; }
static Tensor raw(const std::string& dir,const json& d,int device) {
  auto relative=std::filesystem::path(d.at("file").get<std::string>());
  TORCH_CHECK(!relative.is_absolute(),"Absolute tensor path rejected");
  for(auto& part:relative) TORCH_CHECK(part!="..","Path traversal rejected");
  auto shape=d.at("shape").get<std::vector<int64_t>>();
  TORCH_CHECK(!shape.empty(),"Scalar tensors must be exported as [1]");
  int64_t n=1; for(auto x:shape) { TORCH_CHECK(x>0 && n<=std::numeric_limits<int64_t>::max()/x,"Invalid tensor shape"); n*=x; }
  std::string dtype=d.at("dtype"); at::ScalarType type;
  if(dtype=="BF16") type=at::kBFloat16; else if(dtype=="F32") type=at::kFloat;
  else if(dtype=="I64") type=at::kLong; else if(dtype=="U8") type=at::kByte;
  else { TORCH_CHECK(false,"Unsupported dtype ",dtype); }
  auto out=at::empty(shape,at::TensorOptions().dtype(type).device(at::kCPU));
  auto file=std::filesystem::path(dir)/relative;
  TORCH_CHECK(std::filesystem::file_size(file)==out.nbytes(),"Tensor byte count mismatch: ",file.string());
  std::ifstream f(file,std::ios::binary); f.read(static_cast<char*>(out.data_ptr()),out.nbytes());
  TORCH_CHECK(f.good(),"Short tensor read: ",file.string());
  return out.to(at::Device(at::kCUDA,device));
}
Engine::Engine(const std::string& dir,int rank,int world,int device,ncclComm_t comm,int capacity,EngineOptions options)
 :rank_(rank),world_(world),device_(device),capacity_(capacity),comm_(comm),options_(options) {
  TORCH_CHECK(capacity>0,"Invalid capacity");
  TORCH_CHECK(!options_.bf16_tp_reduce||world<=2,"BF16 TP reduction is restricted to TP1/TP2");
  TORCH_CHECK(!std::filesystem::exists(dir+"/INCOMPLETE"),"Incomplete model export");
    auto manifest=read_json(dir+"/manifest.json");
  TORCH_CHECK(manifest.at("format")=="avi-v1" && manifest.at("tp")==world,"Artifact format or TP mismatch");
  config_=manifest.at("config"); text_=config_.at("text_config"); vision_=config_.at("vision_config");
  TORCH_CHECK(config_.at("model_type")=="qwen3_5","Only qwen3_5 dense supported");
  TORCH_CHECK(options_.mtp_tokens>=0&&options_.mtp_tokens<=5,"MTP window must be 0..5");
  TORCH_CHECK(!options_.mtp_verify_graph||options_.mtp_tokens>0,"Verification graph requires MTP");
  TORCH_CHECK(!options_.reuse_verify_graph||options_.mtp_verify_graph,"Graph reuse requires --mtp-verify-graph");
  TORCH_CHECK(options_.gdn_tensor_chunk==32||options_.gdn_tensor_chunk==64,"GDN tensor chunk must be 32 or 64");
  TORCH_CHECK(!options_.gdn_tensor_prefill||(options_.optimized&&!options_.reference_prefill&&text_.at("linear_key_head_dim")==128&&text_.at("linear_value_head_dim")==128),"Tensor GDN requires optimized K=V=128 prefill");
  TORCH_CHECK(!(options_.gdn_fused_solve&&options_.gdn_tilelang),"Choose one fused GDN backend");
  TORCH_CHECK(!(options_.fp8_tensor_small&&options_.tilelang_fp8),"Choose one FP8 Tensor Core backend");
  TORCH_CHECK(options_.fp8_tensor_split==1||options_.fp8_tensor_split==4,"FP8 Tensor Core split must be 1 or 4");
  TORCH_CHECK(!(options_.gdn_fused_solve||options_.gdn_tilelang)||options_.gdn_tensor_prefill,"Fused GDN requires tensor prefill");
  TORCH_CHECK(!(options_.fp8_tensor_small||options_.tilelang_fp8)||options_.optimized,"FP8 Tensor Core requires optimized mode");
  if(options_.gdn_tilelang||options_.tilelang_fp8)configure_tilelang(options_.tilelang_dir);
  if(options_.mtp_tokens){
    TORCH_CHECK(options_.optimized&&manifest.value("native_mtp",false),"MTP requires optimized mode and an artifact imported with --include-mtp");
    TORCH_CHECK(text_.value("mtp_num_hidden_layers",0)==1&&!text_.value("mtp_use_dedicated_embeddings",false),"Only shared-embedding single-layer MTP supported");
    TORCH_CHECK(!options_.cuda_graph,"MTP target verification uses eager multi-token execution; remove --cuda-graph");
    int key=text_.at("linear_key_head_dim");TORCH_CHECK(key==16||key==128,"MTP state tracking requires GDN K=16 or 128");
    options_.prefix_cache_bytes=0;options_.host_prefix_cache_bytes=0;
  }
  TORCH_CHECK(text_.value("attention_bias",false)==false,"Attention bias unsupported");
  TORCH_CHECK(vision_.value("deepstack_visual_indexes",json::array()).empty(),"Deepstack not supported");
  TORCH_CHECK(text_.at("rope_parameters").value("rope_type",std::string("default"))=="default","Only default RoPE supported");
  eps_=text_.at("rms_norm_eps"); states_.resize(text_.at("num_hidden_layers").get<int>());
  auto tensors=manifest.at("ranks").at(rank).at("tensors");
  size_t loaded=0;
  for(auto item=tensors.begin();item!=tensors.end();++item) {
    if(!options_.mtp_tokens&&item.key().rfind("mtp.",0)==0)continue;
    Weight w; w.data=raw(dir,item.value(),device);
    if(item.value().contains("scale")) w.scale=raw(dir,item.value().at("scale"),device);
    weights_.emplace(item.key(),w);
    if(rank==0 && ++loaded%100==0) std::cerr<<"Loaded "<<loaded<<" tensors\n";
  }
  decode_offset_=at::zeros({1},at::TensorOptions().device(at::Device(at::kCUDA,device)).dtype(at::kLong));
  if(options_.tp_lm_head && world_>1) {
    auto& head=weights_.at("lm_head.weight");int64_t vocab=text_.at("vocab_size");
    TORCH_CHECK(vocab%world_==0 && head.data.size(0)==vocab,"LM head vocabulary must divide TP size");
    int64_t rows=vocab/world_,begin=rank_*rows;
    head.data=head.data.narrow(0,begin,rows).clone();
    if(head.scale.defined())head.scale=head.scale.narrow(0,begin,rows).clone();
    auto bias=weights_.find("lm_head.bias");
    if(bias!=weights_.end())bias->second.data=bias->second.data.narrow(0,begin,rows).clone();
  }
  if(options_.optimized) {
    for(size_t i=0;i<states_.size();i++) {
      auto p="model.language_model.layers."+std::to_string(i);
      fuse_weights(p+".mlp.gate_up",{p+".mlp.gate_proj",p+".mlp.up_proj"});
      if(text_.at("layer_types").at(i)=="full_attention") fuse_weights(p+".self_attn.qkv_gate",{p+".self_attn.q_proj",p+".self_attn.k_proj",p+".self_attn.v_proj"});
      else fuse_weights(p+".linear_attn.in_proj_all",{p+".linear_attn.in_proj_qkv",p+".linear_attn.in_proj_z",p+".linear_attn.in_proj_b",p+".linear_attn.in_proj_a"});
    }
  }
  TORCH_CHECK(!options_.cuda_graph || options_.optimized,"CUDA Graph requires optimized mode");
  if(options_.mtp_tokens){
    fuse_weights("mtp.layers.0.mlp.gate_up",{"mtp.layers.0.mlp.gate_proj","mtp.layers.0.mlp.up_proj"});
    fuse_weights("mtp.layers.0.self_attn.qkv_gate",{"mtp.layers.0.self_attn.q_proj","mtp.layers.0.self_attn.k_proj","mtp.layers.0.self_attn.v_proj"});
    verify_recurrent_.resize(states_.size());verify_conv_.resize(states_.size());
  }
  // A deterministic, bounded cache of decoded matrices for multi-token GEMMs.
  // FP8 source codes remain authoritative; no requantization or activation FP8.
  std::vector<std::string> cache_names;
  for(const auto& entry:weights_)if(entry.second.scale.defined()&&
      (entry.first.rfind("model.language_model.layers.",0)==0||entry.first.rfind("mtp.",0)==0||(options_.cache_vision_weights&&entry.first.rfind("model.visual.",0)==0&&entry.second.data.dim()==2)))cache_names.push_back(entry.first);
  std::sort(cache_names.begin(),cache_names.end());
  // Prioritize draft/vision within the same explicit per-rank memory budget.
  if(options_.cache_vision_weights)std::stable_sort(cache_names.begin(),cache_names.end(),[](const auto& a,const auto& b){
    auto priority=[](const std::string& n){return n.rfind("mtp.",0)==0?0:n.rfind("model.visual.",0)==0?1:2;};
    return priority(a)<priority(b);
  });
  for(const auto& name:cache_names){
    const auto& w=weights_.at(name);size_t bytes=w.data.numel()*2;
    if(bytes<=options_.weight_cache_bytes-decoded_bytes_){
      decoded_weights_[name]=fp8_decode(w.data,w.scale);decoded_bytes_+=bytes;
    }
  }
  if(rank_==0&&options_.weight_cache_bytes)std::cerr<<"Decoded GEMM cache bytes: "<<decoded_bytes_<<" / "<<options_.weight_cache_bytes<<"\n";
  C10_CUDA_CHECK(cudaDeviceSynchronize());
}
void Engine::fuse_weights(const std::string& dest,const std::vector<std::string>& source) {
  std::vector<Tensor> data,scales;
  for(auto& name:source) {auto& w=weights_.at(name+".weight");data.push_back(w.data);if(w.scale.defined())scales.push_back(w.scale);}
  if(!scales.empty()&&(scales.size()!=data.size() || std::any_of(scales.begin(),scales.end(),[&](const Tensor& t){return t.dim()!=scales[0].dim() || (t.dim()==2&&t.size(1)!=scales[0].size(1));}))) {
    // Preserve projection order and FP8 codes. Fuse each contiguous compatible run.
    // Qwen GDN commonly becomes [FP8 qkv+z] and [BF16 b+a]: two GEMMs, not four.
    std::vector<std::vector<std::string>> groups;
    auto compatible=[&](const std::string& a,const std::string& b){
      const auto& x=weights_.at(a+".weight");const auto& y=weights_.at(b+".weight");
      if(x.data.scalar_type()!=y.data.scalar_type()||x.scale.defined()!=y.scale.defined())return false;
      return !x.scale.defined()||(x.scale.dim()==y.scale.dim()&&(x.scale.dim()==1||x.scale.size(1)==y.scale.size(1)));
    };
    for(const auto& name:source){if(groups.empty()||!compatible(groups.back().back(),name))groups.emplace_back();groups.back().push_back(name);}
    std::vector<std::string> children;
    for(size_t i=0;i<groups.size();i++){
      if(groups[i].size()==1){children.push_back(groups[i][0]);continue;}
      auto child=dest+".group"+std::to_string(i);fuse_weights(child,groups[i]);children.push_back(child);
    }
    mixed_projections_[dest]=std::move(children);return;
  }
  Weight combined;combined.data=at::cat(data,0);if(!scales.empty())combined.scale=at::cat(scales,0);
  weights_.emplace(dest+".weight",combined);
  int64_t offset=0;
  for(auto& name:source){
    auto& original=weights_.at(name+".weight");auto rows=original.data.size(0);
    // Keep unfused views for reference prefill without a second weight allocation.
    if(options_.reference_prefill){original.data=combined.data.narrow(0,offset,rows);if(combined.scale.defined())original.scale=combined.scale.narrow(0,offset,rows);}
    else weights_.erase(name+".weight");
    offset+=rows;
  }
}
Tensor Engine::tensor(const std::string& name) {
  auto it=weights_.find(name); TORCH_CHECK(it!=weights_.end(),"Missing tensor ",name);
  TORCH_CHECK(!it->second.scale.defined(),"Expected unquantized tensor ",name); return it->second.data;
}
Tensor Engine::read_input(const std::string& dir,const json& d) { return raw(dir,d,device_); }
Tensor Engine::sum(Tensor x) {
  if(world_==1) return x;
  auto timing=profile_begin("tp.reduce");
  if(options_.bf16_tp_reduce&&x.scalar_type()==at::kBFloat16){
    auto output=x.contiguous();
    auto status=ncclAllReduce(output.data_ptr(),output.data_ptr(),output.numel(),ncclBfloat16,ncclSum,comm_,at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(status==ncclSuccess,ncclGetErrorString(status));profile_end(timing);return output;
  }
  auto f=x.to(at::kFloat).contiguous();
  auto r=ncclAllReduce(f.data_ptr(),f.data_ptr(),f.numel(),ncclFloat,ncclSum,comm_,at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r==ncclSuccess,ncclGetErrorString(r));auto output=f.to(x.scalar_type());profile_end(timing);return output;
}
Tensor Engine::linear(Tensor x,const std::string& name,bool reduce) {
  auto mixed=mixed_projections_.find(name);
  if(mixed!=mixed_projections_.end()){std::vector<Tensor> parts;for(auto& source:mixed->second)parts.push_back(linear(x,source,false));auto out=at::cat(parts,-1);return reduce?sum(out):out;}
  auto it=weights_.find(name+".weight"); TORCH_CHECK(it!=weights_.end(),"Missing linear ",name);
  auto& w=it->second; Tensor y;
  TORCH_CHECK(w.data.dim()>=2,"Invalid matrix ",name);
  auto timing=profile_begin(name=="lm_head"?"linear.head":name.rfind("model.visual.",0)==0?"linear.vision":name.rfind("mtp.",0)==0?"linear.mtp":"linear.text");
  auto cached=decoded_weights_.find(name+".weight");
  auto rows=x.numel()/x.size(-1);
  bool tensor_small=(options_.fp8_tensor_small||options_.tilelang_fp8)&&rows>=2&&rows<=8&&w.scale.defined()&&
    w.data.dim()==2&&w.data.size(1)%128==0&&w.scale.dim()==2&&w.scale.size(1)==w.data.size(1)/128&&
    (!options_.tilelang_fp8||w.data.size(0)%16==0);
  if(tensor_small){
    auto input=x.reshape({rows,x.size(-1)}).contiguous();
    if(options_.tilelang_fp8){++tilelang_fp8_calls_;y=tilelang_fp8_small(input,w.data,w.scale,options_.fp8_tensor_split);}
    else{++fp8_tensor_calls_;y=fp8_tensor_small(input,w.data,w.scale,options_.fp8_tensor_split);}
    auto shape=x.sizes().vec();shape.back()=w.data.size(0);y=y.reshape(shape);
  }else if(options_.multi_token_gemv&&rows>1&&rows<=8){
    bool use_cached=cached!=decoded_weights_.end()&&!(options_.shared_gemv_fp8&&w.scale.defined());
    auto matrix=use_cached?cached->second:w.data.reshape({w.data.size(0),-1});
    auto scales=use_cached?Tensor():w.scale;
    y=small_linear_shared(x.reshape({rows,x.size(-1)}).contiguous(),matrix,scales);
    auto shape=x.sizes().vec();shape.back()=matrix.size(0);y=y.reshape(shape);
  } else if(cached!=decoded_weights_.end()&&rows>1){
    y=at::matmul(x,cached->second.t());
  } else if(w.scale.defined() && options_.optimized && !(options_.cublas_prefill && x.numel()/x.size(-1)>8)) {
    y=fp8_linear(x.reshape({-1,x.size(-1)}).contiguous(),w.data,w.scale,options_.vector_gemv);
    auto shape=x.sizes().vec();shape.back()=w.data.size(0);y=y.reshape(shape);
  } else if(w.scale.defined()) {
    TORCH_CHECK(w.data.dim()==2 && x.size(-1)==w.data.size(1),"FP8 dimensions mismatch ",name);
    auto shape=x.sizes().vec(); shape.back()=w.data.size(0); y=at::empty(shape,x.options());
    // Bound BF16 temporary to 2048 output rows; full model remains FP8 resident.
    for(int64_t begin=0;begin<w.data.size(0);begin+=2048) {
      auto count=std::min<int64_t>(2048,w.data.size(0)-begin);
      auto decoded=fp8_decode(w.data.narrow(0,begin,count),w.scale.narrow(0,begin,count));
      y.narrow(-1,begin,count).copy_(at::matmul(x,decoded.t()));
    }
  } else {
    auto matrix=w.data.reshape({w.data.size(0),-1});
    y=at::matmul(x,matrix.t());
  }
  profile_end(timing);
  if(reduce) y=sum(y);
  auto b=weights_.find(name+".bias"); if(b!=weights_.end()) y=y+b->second.data;
  return y;
}
Tensor Engine::norm(Tensor x,const std::string& name,bool one_center) {
  if(options_.optimized) return fused_rms(x,tensor(name+".weight"),eps_,one_center);
  auto f=x.to(at::kFloat); auto w=tensor(name+".weight").to(at::kFloat);
  auto normalized=f*at::rsqrt((f*f).mean(-1,true)+eps_);
  if(one_center) return (normalized*(w+1)).to(x.scalar_type());
  return (normalized.to(x.scalar_type())*w.to(x.scalar_type())).to(x.scalar_type());
}
Tensor Engine::layer_norm(Tensor x,const std::string& name) {
  return at::layer_norm(x,{x.size(-1)},tensor(name+".weight"),tensor(name+".bias"),1e-6);
}
static Tensor rotate(Tensor x,Tensor cos,Tensor sin,int d) {
  auto a=x.narrow(-1,0,d), pass=x.narrow(-1,d,x.size(-1)-d);
  auto swapped=at::cat({-a.narrow(-1,d/2,d/2),a.narrow(-1,0,d/2)},-1);
  return at::cat({a*cos+swapped*sin,pass},-1);
}
Tensor Engine::embed(Tensor ids) {
  auto& w=weights_.at("model.language_model.embed_tokens.weight");
  auto selected=w.data.index_select(0,ids.reshape({-1}));
  return w.scale.defined()?fp8_decode(selected.contiguous(),w.scale.index_select(0,ids.reshape({-1})).contiguous()):selected;
}
Tensor Engine::logits(Tensor x) {
  return project_logits(norm(x,"model.language_model.norm"));
}
Tensor Engine::project_logits(Tensor x) {
  auto local=linear(x,"lm_head").to(at::kFloat).contiguous();
  if(!options_.tp_lm_head || world_==1)return local;
  auto gathered=at::empty({world_,local.size(0),local.size(1)},local.options());
  auto status=ncclAllGather(local.data_ptr(),gathered.data_ptr(),local.numel(),ncclFloat,comm_,at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status==ncclSuccess,ncclGetErrorString(status));
  // NCCL gives [rank,batch,local_vocab]; preserve batch rows when assembling vocabulary.
  return gathered.permute({1,0,2}).reshape({local.size(0),world_*local.size(1)});
}
Tensor Engine::full_attention(Tensor x,Tensor positions,int layer,const std::string& p,Tensor projected,bool finish,State* external) {
  auto T=x.size(0); int H=text_.at("num_attention_heads").get<int>()/world_;
  int HK=text_.at("num_key_value_heads").get<int>()/world_, D=text_.at("head_dim");
  if(options_.optimized && !projected.defined())projected=linear(x,p+".qkv_gate");
  auto qg=(options_.optimized?projected.narrow(-1,0,H*2*D):linear(x,p+".q_proj")).reshape({T,H,2*D});
  auto q=norm(qg.narrow(-1,0,D),p+".q_norm"), gate=qg.narrow(-1,D,D).reshape({T,H*D});
  auto k=norm((options_.optimized?projected.narrow(-1,H*2*D,HK*D):linear(x,p+".k_proj")).reshape({T,HK,D}),p+".k_norm");
  auto v=(options_.optimized?projected.narrow(-1,H*2*D+HK*D,HK*D):linear(x,p+".v_proj")).reshape({T,HK,D});
  auto rope=text_.at("rope_parameters"); int rd=int(D*rope.value("partial_rotary_factor",0.25));
  auto sections=rope.at("mrope_section").get<std::vector<int>>();
  double theta=rope.at("rope_theta");
  if(options_.optimized) {
    q=fused_rope(q,positions,rd,theta,sections[1],sections[2]);
    k=fused_rope(k,positions,rd,theta,sections[1],sections[2]);
  } else {
  std::vector<Tensor> freqs;
  for(int j=0;j<rd/2;j++) {
    int axis=0; if(j%3==1 && j<sections[1]*3) axis=1; if(j%3==2 && j<sections[2]*3) axis=2;
    freqs.push_back(positions.select(0,axis).to(at::kFloat)*std::pow(theta,-2.0*j/rd));
  }
  auto freq=at::stack(freqs,-1); auto angle=at::cat({freq,freq},-1);
  auto cos=angle.cos().to(x.scalar_type()).unsqueeze(1), sin=angle.sin().to(x.scalar_type()).unsqueeze(1);
  q=rotate(q,cos,sin,rd); k=rotate(k,cos,sin,rd);
  }
  auto& state=external?*external:states_.at(layer);
  TORCH_CHECK(decode_mode_ || state.length+T<=session_capacity(),"KV capacity exceeded");
  if(!state.key.defined()) { state.key=at::empty({session_capacity(),HK,D},x.options()); state.value=at::empty_like(state.key); }
  if(verifying_graph_){
    auto out=gqa_chunk_dynamic(q,k,v,state.key,state.value,decode_offset_).reshape({T,H*D});
    out=options_.extra_fusions?fused_sigmoid_gate(out,gate):out*gate.sigmoid();
    return finish?linear(out,p+".o_proj",true):out;
  }
  if(options_.optimized&&!decode_mode_&&T<=8){
    auto out=gqa_chunk(q,k,v,state.key,state.value,state.length).reshape({T,H*D});state.length+=T;
    out=options_.extra_fusions?fused_sigmoid_gate(out,gate):out*gate.sigmoid();
    return finish?linear(out,p+".o_proj",true):out;
  }
  if(decode_mode_) {
    auto out=gqa_decode(q,k,v,state.key,state.value,decode_offset_).reshape({T,H*D});
    out=(options_.optimized&&options_.extra_fusions)?fused_sigmoid_gate(out,gate):out*gate.sigmoid();
    return finish?linear(out,p+".o_proj",true):out;
  }
  int old=state.length; state.key.narrow(0,old,T).copy_(k); state.value.narrow(0,old,T).copy_(v); state.length+=T;
  if(options_.optimized&&options_.flash_prefill){
    auto out=avi::flash_prefill(q,state.key.narrow(0,0,state.length),state.value.narrow(0,0,state.length)).reshape({T,H*D});
    out=options_.extra_fusions?fused_sigmoid_gate(out,gate):out*gate.sigmoid();
    return finish?linear(out,p+".o_proj",true):out;
  }
  auto keys=state.key.narrow(0,0,state.length).repeat_interleave(H/HK,1).transpose(0,1).unsqueeze(0);
  auto values=state.value.narrow(0,0,state.length).repeat_interleave(H/HK,1).transpose(0,1).unsqueeze(0);
  auto qp=q.transpose(0,1).unsqueeze(0);
  // Explicit offset-aware mask: causal=true alone is incorrect for cached chunks.
  auto ints=x.options().dtype(at::kLong);
  auto mask=at::arange(state.length,ints).unsqueeze(0)<=at::arange(old,state.length,ints).unsqueeze(1);
  auto out=at::scaled_dot_product_attention(qp,keys,values,mask,0.0,false).squeeze(0).transpose(0,1).reshape({T,H*D});
  out=(options_.optimized&&options_.extra_fusions)?fused_sigmoid_gate(out,gate):out*gate.sigmoid();return finish?linear(out,p+".o_proj",true):out;
}
Tensor Engine::delta_attention(Tensor x,int layer,const std::string& p,Tensor projected,bool finish) {
  int K=text_.at("linear_key_head_dim"), V=text_.at("linear_value_head_dim");
  int HK=text_.at("linear_num_key_heads").get<int>()/world_, H=text_.at("linear_num_value_heads").get<int>()/world_;
  int kernel=text_.at("linear_conv_kernel_dim"); auto T=x.size(0); auto& s=states_.at(layer);
  int C=2*HK*K+H*V;
  bool tensor_prefill=options_.gdn_tensor_prefill&&!verifying_&&!decode_mode_&&T>=32;
  if(options_.optimized && !projected.defined())projected=linear(x,p+".in_proj_all");
  auto mixed=(options_.optimized?projected.narrow(-1,0,C):linear(x,p+".in_proj_qkv")).t().unsqueeze(0);
  if(!s.conv.defined()) {
    s.conv=at::zeros({1,C,kernel-1},x.options());
    s.recurrent=at::zeros({H,K,V},x.options().dtype(at::kFloat));
  }
  if(options_.optimized&&options_.extra_fusions&&options_.fused_gdn_prepare&&
      !verifying_&&!decode_mode_&&T>=32&&kernel<=32){
    auto preparation=profile_begin("gdn.prepare");
    auto inputs=fused_gdn_prepare(projected.contiguous(),tensor(p+".conv1d.weight"),s.conv,
      tensor(p+".A_log"),tensor(p+".dt_bias"),HK,H,K,V,tensor_prefill);
    profile_end(preparation);
    auto timing=profile_begin("gdn.scan");
    if(tensor_prefill){++gdn_tensor_calls_;if(options_.gdn_tilelang)++gdn_tilelang_calls_;else if(options_.gdn_fused_solve)++gdn_fused_calls_;}
    auto result=tensor_prefill?delta_scan_tensor(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],s.recurrent,options_.gdn_tensor_chunk,options_.gdn_fused_solve,options_.gdn_tilelang):(options_.gdn_wy||options_.gdn_wy_fused)?
      delta_scan_wy(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],s.recurrent,32,options_.gdn_wy_fused):
      delta_scan_fast(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],s.recurrent,{},options_.gdn_cooperative);
    profile_end(timing);
    auto z=projected.narrow(-1,C,H*V).reshape({T,H,V});
    result=fused_rms_gate(result,tensor(p+".norm.weight"),z,eps_).reshape({T,H*V});
    return finish?linear(result,p+".out_proj",true):result;
  }
  Tensor conv;
  if(decode_mode_) conv=conv_decode(mixed.reshape({1,C}),tensor(p+".conv1d.weight"),s.conv);
  else {
    auto input=at::cat({s.conv,mixed},-1);
    if(verifying_)verify_conv_.at(layer)=input;
    s.conv.copy_(input.narrow(-1,input.size(-1)-(kernel-1),kernel-1));
    conv=(options_.optimized&&options_.fused_gdn_conv)?conv_prefill(input,tensor(p+".conv1d.weight")):
      at::silu(at::conv1d(input,tensor(p+".conv1d.weight"),{},at::IntArrayRef{1},at::IntArrayRef{0},at::IntArrayRef{1},C)).squeeze(0).t();
  }
  auto q=conv.narrow(-1,0,HK*K).reshape({T,HK,K});
  auto k=conv.narrow(-1,HK*K,HK*K).reshape({T,HK,K});
  auto v=conv.narrow(-1,2*HK*K,H*V).reshape({T,H,V});
  // Match reference normalization and rounding before the FP32 recurrence.
  if(options_.optimized){q=fused_l2(q);k=fused_l2(k);if(!tensor_prefill){q=q.repeat_interleave(H/HK,1).contiguous();k=k.repeat_interleave(H/HK,1).contiguous();}}
  else {
  q=(q.to(at::kFloat)*at::rsqrt(q.to(at::kFloat).square().sum(-1,true)+1e-6)).to(at::kBFloat16).repeat_interleave(H/HK,1).contiguous();
  k=(k.to(at::kFloat)*at::rsqrt(k.to(at::kFloat).square().sum(-1,true)+1e-6)).to(at::kBFloat16).repeat_interleave(H/HK,1).contiguous();
  }
  auto a=(options_.optimized?projected.narrow(-1,C+H*V+H,H):linear(x,p+".in_proj_a"));
  auto b=(options_.optimized?projected.narrow(-1,C+H*V,H):linear(x,p+".in_proj_b"));
  Tensor g,beta;
  if(options_.optimized&&options_.extra_fusions){auto gates=fused_gdn_gates(a,b,tensor(p+".A_log"),tensor(p+".dt_bias"));g=gates.first;beta=gates.second;}
  else {g=-tensor(p+".A_log").to(at::kFloat).exp()*at::softplus(a.to(at::kFloat)+tensor(p+".dt_bias").to(at::kFloat));beta=b.sigmoid().to(at::kFloat);}
  Tensor trajectory;
  if(verifying_){
    auto& buffer=verify_recurrent_.at(layer);
    if(!buffer.defined()||buffer.size(0)<T)buffer=at::empty({options_.mtp_tokens+1,H,K,V},s.recurrent.options());
    trajectory=buffer.narrow(0,0,T);
  }
  auto scan_timing=profile_begin("gdn.scan");
  Tensor result;
  if(tensor_prefill){++gdn_tensor_calls_;if(options_.gdn_tilelang)++gdn_tilelang_calls_;else if(options_.gdn_fused_solve)++gdn_fused_calls_;result=delta_scan_tensor(q,k,v,g,beta,s.recurrent,options_.gdn_tensor_chunk,options_.gdn_fused_solve,options_.gdn_tilelang);}
  else if((options_.gdn_wy||options_.gdn_wy_fused)&&!verifying_&&!decode_mode_&&T>=32)result=delta_scan_wy(q,k,v,g,beta,s.recurrent,32,options_.gdn_wy_fused);
  else if(gdn_chunk_enabled_&&!verifying_&&!decode_mode_&&T>=32)result=delta_scan_chunked(q,k,v,g,beta,s.recurrent);
  else result=(options_.optimized&&(K==128||K==16))?delta_scan_fast(q,k,v.contiguous(),g.contiguous(),beta.contiguous(),s.recurrent,trajectory,options_.gdn_cooperative):delta_scan(q,k,v.contiguous(),g.contiguous(),beta.contiguous(),s.recurrent);
  profile_end(scan_timing);
  auto z=(options_.optimized?projected.narrow(-1,C,H*V):linear(x,p+".in_proj_z")).reshape({T,H,V});
  result=(options_.optimized&&options_.extra_fusions)?fused_rms_gate(result,tensor(p+".norm.weight"),z,eps_):(norm(result,p+".norm",false).to(at::kFloat)*at::silu(z.to(at::kFloat))).to(at::kBFloat16);
  result=result.reshape({T,H*V});return finish?linear(result,p+".out_proj",true):result;
}
Tensor Engine::step(Tensor x,Tensor positions) {
  auto embeddings=x;
  struct Restore {bool& value;bool saved;~Restore(){value=saved;}} restore{options_.optimized,options_.optimized};
  if(options_.reference_prefill&&!decode_mode_&&!verifying_)options_.optimized=false;
  TORCH_CHECK(x.dim()==2 && positions.dim()==2 && positions.size(0)==3 && positions.size(1)==x.size(0),"Invalid text input shape");
  for(size_t i=0;i<states_.size();i++) {
    std::string p="model.language_model.layers."+std::to_string(i);
    auto n=norm(x,p+".input_layernorm");
    auto update=text_.at("layer_types").at(i)=="full_attention"?full_attention(n,positions,i,p+".self_attn"):delta_attention(n,i,p+".linear_attn");
    if(options_.optimized&&options_.fused_residual_norm){
      auto pair=residual_rms(x,update,tensor(p+".post_attention_layernorm.weight"),eps_);
      x=pair.first;n=pair.second;
    }else{x=x+update;n=norm(x,p+".post_attention_layernorm");}
    auto gated=options_.optimized?fused_swiglu(linear(n,p+".mlp.gate_up")):at::silu(linear(n,p+".mlp.gate_proj"))*linear(n,p+".mlp.up_proj");
    x=x+linear(gated,p+".mlp.down_proj",true);
    if(!trace_prefix_.empty())trace_layer(x,int(i));
  }
  if(options_.mtp_tokens&&!verifying_)advance_draft(embeddings,positions,x);
  return x;
}
void Engine::trace_layer(Tensor hidden,int layer) {
  auto write=[&](const std::string& kind,Tensor value){
    if(!value.defined())return;
    auto path=trace_prefix_+".rank"+std::to_string(rank_)+".layer"+std::to_string(layer)+"."+kind+".f32";
    TORCH_CHECK(!std::filesystem::exists(path),"Trace already exists: ",path);
    value=value.to(at::kFloat).to(at::kCPU).contiguous();
    std::ofstream file(path,std::ios::binary);file.write(static_cast<const char*>(value.data_ptr()),value.nbytes());
    TORCH_CHECK(file.good(),"Cannot write layer trace: ",path);
  };
  write("hidden",hidden.narrow(0,hidden.size(0)-1,1));
  auto& state=states_.at(layer);
  if(state.recurrent.defined()){
    auto flat=state.recurrent.reshape({-1});auto stride=std::max<int64_t>(1,(flat.numel()+4095)/4096);
    write("recurrent_sample",flat.slice(0,0,flat.numel(),stride));
  }
  if(state.conv.defined())write("conv",state.conv);
}
Tensor Engine::vision(const std::string& dir,const json& request) {
  if(!request.contains("images") || request.at("images").empty()) return {};
  std::vector<Tensor> outputs; int H=vision_.at("num_heads"), hidden=vision_.at("hidden_size"), D=hidden/H;
  int merge=vision_.at("spatial_merge_size");
  for(auto& image:request.at("images")) {
    auto cache_key=options_.image_cache_bytes?request_hash(dir,image):std::string();
    auto cached=images_.find(cache_key);
    if(cached!=images_.end()) {cached->second.used=++cache_clock_;++image_hits_;outputs.push_back(cached->second.tensor);continue;}

    auto patches=read_input(dir,image.at("patches")).to(at::kBFloat16); auto T=patches.size(0);
    auto x=linear(patches,"model.visual.patch_embed.proj");
    auto indices=read_input(dir,image.at("position_indices"));
    auto factors=read_input(dir,image.at("position_weights")).to(at::kBFloat16);
    Tensor pos;
    for(int i=0;i<4;i++) {
      auto term=tensor("model.visual.pos_embed.weight").index_select(0,indices.select(0,i))*factors.select(0,i).unsqueeze(-1);
      pos=i==0?term:pos+term;
    }
    x=x+pos;
    auto coords=read_input(dir,image.at("coords")).to(at::kFloat);
    std::vector<Tensor> frequencies;
    for(int axis=0;axis<2;axis++) for(int j=0;j<D/4;j++) frequencies.push_back(coords.select(1,axis)*std::pow(10000.0,-4.0*j/D));
    auto f=at::stack(frequencies,-1); auto angles=at::cat({f,f},-1).unsqueeze(1);
    auto cos=angles.cos(),sin=angles.sin();
    for(int i=0;i<vision_.at("depth").get<int>();i++) {
      auto p="model.visual.blocks."+std::to_string(i); auto n=layer_norm(x,p+".norm1");
      auto qkv=linear(n,p+".attn.qkv").reshape({T,3,H,D});
      auto q=rotate(qkv.select(1,0).to(at::kFloat),cos,sin,D).to(at::kBFloat16);
      auto k=rotate(qkv.select(1,1).to(at::kFloat),cos,sin,D).to(at::kBFloat16);
      auto v=qkv.select(1,2);
      auto out=at::scaled_dot_product_attention(q.transpose(0,1).unsqueeze(0),k.transpose(0,1).unsqueeze(0),v.transpose(0,1).unsqueeze(0),{},0.0,false);
      x=x+linear(out.squeeze(0).transpose(0,1).reshape({T,hidden}),p+".attn.proj");
      n=layer_norm(x,p+".norm2"); x=x+linear(at::gelu(linear(n,p+".mlp.linear_fc1"),"tanh"),p+".mlp.linear_fc2");
    }
    x=layer_norm(x,"model.visual.merger.norm").reshape({-1,hidden*merge*merge});
    x=linear(at::gelu(linear(x,"model.visual.merger.linear_fc1"),"none"),"model.visual.merger.linear_fc2");
    if(options_.image_cache_bytes && x.nbytes()<=options_.image_cache_bytes) {
      while(image_bytes_+x.nbytes()>options_.image_cache_bytes && !images_.empty()) {
        auto oldest=std::min_element(images_.begin(),images_.end(),[](const auto& a,const auto& b){return a.second.used<b.second.used;});
        image_bytes_-=oldest->second.bytes;images_.erase(oldest);
      }
      images_.emplace(cache_key,Image{x,x.nbytes(),++cache_clock_});image_bytes_+=x.nbytes();
    }
    outputs.push_back(x);
  }
  return at::cat(outputs,0);
}

struct DecodeGraph {std::unique_ptr<at::cuda::CUDAGraph> graph;Tensor ids,positions,output;~DecodeGraph(){graph.reset();}};
Engine::~Engine(){
  // NCCL capture retains communicator resources until graph destruction.
  // The owning executable keeps the communicator alive through this destructor.
  cudaDeviceSynchronize();verify_graphs_.clear();verify_graph_pool_.clear();graphs_.clear();draft_graphs_.clear();draft_graph_pool_.clear();
}
int Engine::session_capacity() const {
    auto it=session_capacities_.find(active_);return it==session_capacities_.end()?capacity_:it->second;
  }
  void Engine::reserve_session(int id,int tokens) {
    TORCH_CHECK(tokens>0&&tokens<=capacity_&&!session_capacities_.count(id),"Invalid session reservation");
    session_capacities_[id]=tokens;
  }
  void Engine::activate(int id) {
  TORCH_CHECK(id>=0,"Invalid session ID");if(id==active_)return;
  bool populated=std::any_of(states_.begin(),states_.end(),[](const State& s){return s.key.defined()||s.conv.defined();});
    if(populated)sessions_[active_]=std::move(states_);else sessions_.erase(active_);auto it=sessions_.find(id);
  bool fresh=it==sessions_.end();
  if(fresh)states_=std::vector<State>(text_.at("num_hidden_layers").get<int>());
  else {states_=std::move(it->second);sessions_.erase(it);}active_=id;
  if(fresh)adopt_verify();
}
void Engine::drop(int id) {
  C10_CUDA_CHECK(cudaDeviceSynchronize());release_draft(id);
  release_verify(id);graphs_.erase(id);session_capacities_.erase(id);
  if(id==active_)states_=std::vector<State>(text_.at("num_hidden_layers").get<int>());else sessions_.erase(id);
}
Tensor Engine::decode(int64_t token,int64_t position,int consumed) {
  TORCH_CHECK(consumed>=0 && consumed<session_capacity(),"Decode capacity exceeded");
  auto longs=decode_offset_.options();
  if(!options_.optimized) return logits(step(embed(at::full({1},token,longs)),at::full({3,1},position,longs)));
  decode_mode_=true;decode_offset_.fill_(consumed);
  Tensor result;
  if(!options_.cuda_graph)result=logits(step(embed(at::full({1},token,longs)),at::full({3,1},position,longs)));
  else {
    auto& holder=graphs_[active_];if(!holder)holder=std::make_shared<DecodeGraph>();auto& graph=*holder;
    if(!graph.graph) {
      graph.ids=at::full({1},token,longs);graph.positions=at::full({3,1},position,longs);
      // Warmup mutates recurrence/conv. Snapshot and restore them before capture/replay.
      std::vector<std::pair<Tensor,Tensor>> saved;
      for(auto& st:states_)saved.emplace_back(st.conv.defined()?st.conv.clone():Tensor(),st.recurrent.defined()?st.recurrent.clone():Tensor());
      auto restore=[&](){for(size_t i=0;i<states_.size();i++)if(saved[i].first.defined()){states_[i].conv.copy_(saved[i].first);states_[i].recurrent.copy_(saved[i].second);}};
      C10_CUDA_CHECK(cudaDeviceSynchronize());auto stream=c10::cuda::getStreamFromPool(false,device_);
      {
        c10::cuda::CUDAStreamGuard guard(stream);
        for(int i=0;i<2;i++){graph.output=logits(step(embed(graph.ids),graph.positions));restore();}
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        graph.graph=std::make_unique<at::cuda::CUDAGraph>();graph.graph->capture_begin();
        graph.output=logits(step(embed(graph.ids),graph.positions));graph.graph->capture_end();
      }
    } else {graph.ids.fill_(token);graph.positions.fill_(position);}
    graph.graph->replay();result=graph.output;
  }
  decode_mode_=false;for(auto& st:states_)if(st.key.defined())st.length=consumed+1;return result;
}
Tensor Engine::decode_batch(const std::vector<int>& sessions,const std::vector<int64_t>& tokens,const std::vector<int64_t>& positions,const std::vector<int>& consumed) {
  size_t B=sessions.size();TORCH_CHECK(B>0&&tokens.size()==B&&positions.size()==B&&consumed.size()==B,"Batch metadata mismatch");
  if(B==1){activate(sessions[0]);return decode(tokens[0],positions[0],consumed[0]);}
  if(!options_.optimized) {std::vector<Tensor> out;for(size_t i=0;i<B;i++){activate(sessions[i]);out.push_back(decode(tokens[i],positions[i],consumed[i]));}return at::cat(out);}
  for(size_t i=0;i<B;i++){activate(sessions[i]);TORCH_CHECK(consumed[i]>=0&&consumed[i]<session_capacity(),"Batch exceeds capacity");}
  auto ids=at::empty({int64_t(B)},at::TensorOptions().dtype(at::kLong));std::copy(tokens.begin(),tokens.end(),ids.data_ptr<int64_t>());ids=ids.to(decode_offset_.device());
  auto x=embed(ids);decode_mode_=true;
  // Projection/MLP GEMMs and all-reduces are batched; each request retains independent KV/GDN state.
  for(int layer=0;layer<text_.at("num_hidden_layers").get<int>();layer++) {
    std::string p="model.language_model.layers."+std::to_string(layer);bool full=text_.at("layer_types").at(layer)=="full_attention";
    auto n=norm(x,p+".input_layernorm");auto projected=linear(n,p+(full?".self_attn.qkv_gate":".linear_attn.in_proj_all"));
    std::vector<Tensor> local;
    for(size_t i=0;i<B;i++) {
      activate(sessions[i]);decode_offset_.fill_(consumed[i]);auto row=n.narrow(0,i,1),proj=projected.narrow(0,i,1);
      local.push_back(full?full_attention(row,at::full({3,1},positions[i],decode_offset_.options()),layer,p+".self_attn",proj,false):delta_attention(row,layer,p+".linear_attn",proj,false));
    }
    x=x+linear(at::cat(local,0),p+(full?".self_attn.o_proj":".linear_attn.out_proj"),true);
    n=norm(x,p+".post_attention_layernorm");x=x+linear(fused_swiglu(linear(n,p+".mlp.gate_up")),p+".mlp.down_proj",true);
  }
  decode_mode_=false;for(size_t i=0;i<B;i++){activate(sessions[i]);for(auto& st:states_)if(st.key.defined())st.length=consumed[i]+1;}
  return logits(x);
}
void Engine::demote_prefix(const std::string& key) {
    auto it=prefixes_.find(key);TORCH_CHECK(it!=prefixes_.end(),"Missing checkpoint");
    auto& entry=it->second;
    if(options_.host_prefix_cache_bytes && entry.bytes<=options_.host_prefix_cache_bytes) {
      while(host_prefix_bytes_+entry.bytes>options_.host_prefix_cache_bytes&&!host_prefixes_.empty()) {
        auto old=std::min_element(host_prefixes_.begin(),host_prefixes_.end(),[](const auto& a,const auto& b){return a.second.used<b.second.used;});
        host_prefix_bytes_-=old->second.bytes;host_prefixes_.erase(old);
      }
      auto host=[](const Tensor& x){if(!x.defined())return Tensor();auto y=at::empty(x.sizes(),x.options().device(at::kCPU).pinned_memory(true));y.copy_(x,false);return y;};
      Prefix copy;copy.bytes=entry.bytes;copy.used=entry.used;copy.logits=host(entry.logits);
      for(auto& st:entry.states){State dst;dst.length=st.length;dst.key=host(st.key);dst.value=host(st.value);dst.conv=host(st.conv);dst.recurrent=host(st.recurrent);copy.states.push_back(std::move(dst));}
      host_prefix_bytes_+=copy.bytes;host_prefixes_[key]=std::move(copy);
    }
    prefix_bytes_-=entry.bytes;prefixes_.erase(it);
  }
  void Engine::save_prefix(const std::string& key,Tensor output) {
  if(!options_.prefix_cache_bytes || (prefixes_.count(key)||host_prefixes_.count(key)))return;
  size_t bytes=output.nbytes();for(auto& st:states_){if(st.key.defined())bytes+=2*st.length*st.key.size(1)*st.key.size(2)*st.key.element_size();if(st.conv.defined())bytes+=st.conv.nbytes()+st.recurrent.nbytes();}
  if(bytes>options_.prefix_cache_bytes)return;
  while(prefix_bytes_+bytes>options_.prefix_cache_bytes&&!prefixes_.empty()) {
    auto oldest=std::min_element(prefixes_.begin(),prefixes_.end(),[](const auto& a,const auto& b){return a.second.used<b.second.used;});auto victim=oldest->first;demote_prefix(victim);
  }
  Prefix entry;entry.bytes=bytes;entry.used=++cache_clock_;entry.logits=output.clone();
  for(auto& st:states_) {State copy;copy.length=st.length;if(st.key.defined()){copy.key=st.key.narrow(0,0,st.length).clone();copy.value=st.value.narrow(0,0,st.length).clone();}if(st.conv.defined()){copy.conv=st.conv.clone();copy.recurrent=st.recurrent.clone();}entry.states.push_back(copy);}
  prefixes_.emplace(key,std::move(entry));prefix_bytes_+=bytes;
}
Tensor Engine::restore_prefix(const std::string& key) {
  auto it=prefixes_.find(key);auto host=host_prefixes_.find(key);if(it==prefixes_.end()&&host==host_prefixes_.end())return {};bool from_host=it==prefixes_.end();auto& entry=from_host?host->second:it->second;entry.used=++cache_clock_;++prefix_hits_;if(from_host)++host_prefix_hits_;
  int reserved=session_capacity();drop(active_);session_capacities_[active_]=reserved;for(size_t i=0;i<states_.size();i++){auto& src=entry.states[i];auto& dst=states_[i];dst.length=src.length;
    if(src.key.defined()){dst.key=at::empty({session_capacity(),src.key.size(1),src.key.size(2)},src.key.options().device(at::Device(at::kCUDA,device_)).pinned_memory(false));dst.value=at::empty_like(dst.key);dst.key.narrow(0,0,src.length).copy_(src.key);dst.value.narrow(0,0,src.length).copy_(src.value);}
    if(src.conv.defined()){dst.conv=src.conv.to(at::Device(at::kCUDA,device_),src.conv.scalar_type(),false,true);dst.recurrent=src.recurrent.to(at::Device(at::kCUDA,device_),src.recurrent.scalar_type(),false,true);}}
  return entry.logits.to(at::Device(at::kCUDA,device_),entry.logits.scalar_type(),false,true);
}
json Engine::cache_stats() const {return {{"gdn_fused_calls",gdn_fused_calls_},{"gdn_tilelang_calls",gdn_tilelang_calls_},{"fp8_tensor_calls",fp8_tensor_calls_},{"tilelang_fp8_calls",tilelang_fp8_calls_},{"gdn_tensor_calls",gdn_tensor_calls_},{"verify_graph_pool_tensor_bytes",verify_graph_pool_bytes_},{"verify_graph_reuses",verify_graph_reuses_},{"verify_graph_pool_entries",verify_graph_pool_.size()},{"verify_graph_builds",verify_graph_builds_},{"verify_graph_replays",verify_graph_replays_},{"image_bytes",image_bytes_},{"prefix_bytes",prefix_bytes_},{"image_hits",image_hits_},{"prefix_hits",prefix_hits_},{"host_prefix_bytes",host_prefix_bytes_},{"host_prefix_hits",host_prefix_hits_}};}
std::string request_hash(const std::string& dir,const json& request) {
  std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),EVP_MD_CTX_free);
  TORCH_CHECK(context&&EVP_DigestInit_ex(context.get(),EVP_sha256(),nullptr)==1,"SHA256 initialization failed");
  auto metadata=request.dump();EVP_DigestUpdate(context.get(),metadata.data(),metadata.size());
  auto root=std::filesystem::weakly_canonical(dir);
  std::function<void(const json&)> walk=[&](const json& j){
    if(j.is_object()&&j.contains("file")) {
      auto file=std::filesystem::weakly_canonical(root/j.at("file").get<std::string>());
      auto relative=file.lexically_relative(root);TORCH_CHECK(!relative.empty()&&!relative.is_absolute(),"Invalid cache input path");for(auto& part:relative)TORCH_CHECK(part!="..","Cache path escape");
      std::ifstream f(file,std::ios::binary);TORCH_CHECK(f.good(),"Missing cache input ",file.string());char buffer[65536];
      while(f){f.read(buffer,sizeof(buffer));if(f.gcount())EVP_DigestUpdate(context.get(),buffer,f.gcount());}TORCH_CHECK(f.eof(),"Cache input read failed");
    }
    if(j.is_structured())for(auto& value:j)walk(value);
  };walk(request);unsigned char digest[EVP_MAX_MD_SIZE];unsigned int length;TORCH_CHECK(EVP_DigestFinal_ex(context.get(),digest,&length)==1,"SHA256 finalization failed");
  std::ostringstream output;for(unsigned int i=0;i<length;i++)output<<std::hex<<std::setw(2)<<std::setfill('0')<<int(digest[i]);return output.str();
}
}
