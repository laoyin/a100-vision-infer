#include "avi/engine.h"
#include "avi/json_grammar.h"
#include "avi/memory_budget.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <unordered_map>
#include <filesystem>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/core/InferenceMode.h>
#include <ATen/Parallel.h>
#include <poll.h>
#include <unistd.h>
#include <deque>
#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>
#include <set>
#include <map>
using avi::json;using Clock=std::chrono::steady_clock;
static double age(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static void emit(int rank,const json& j){if(!rank)std::cout<<j.dump()<<"\n"<<std::flush;}
static json broadcast(json j,int rank){std::string s=rank?std::string():j.dump();int size=s.size();MPI_Bcast(&size,1,MPI_INT,0,MPI_COMM_WORLD);s.resize(size);MPI_Bcast(s.data(),size,MPI_CHAR,0,MPI_COMM_WORLD);return rank?json::parse(s):j;}
static std::vector<std::string> vocabulary;
static std::unordered_map<std::string,std::vector<int64_t>> grammar_masks;
struct Job {
 avi::JsonGrammar grammar;
 std::string id,path,key;json req,options;int slot=0,offset=0,consumed=0;int64_t next=0;
 uint64_t reservation=0;int token_capacity=0;
 at::Tensor ids,positions,embedding,logits;std::vector<int64_t> generated;
 bool initialized=false,ready=false,needs_decode=false,cancelled=false;std::string finish;
 Clock::time_point submitted=Clock::now();double first_token=-1;std::mt19937_64 rng;
};
static int64_t sample(Job& job){
 bool constrained=job.options.value("json_object",false);std::vector<unsigned char> allowed;
 if(constrained){
  auto signature=job.grammar.signature();auto it=grammar_masks.find(signature);
  if(it==grammar_masks.end()) {std::vector<int64_t> valid;for(size_t i=0;i<vocabulary.size();i++)if(!vocabulary[i].empty()){auto copy=job.grammar;if(copy.feed(vocabulary[i]))valid.push_back(i);}if(grammar_masks.size()>=32)grammar_masks.clear();it=grammar_masks.emplace(signature,std::move(valid)).first;}
  allowed.resize(vocabulary.size(),0);for(auto id:it->second)allowed[id]=1;
  if(job.grammar.complete())for(auto id:job.req.at("eos_token_ids").get<std::vector<int64_t>>())if(id>=0&&size_t(id)<allowed.size())allowed[id]=1;
 }

 if(!constrained && job.options.value("temperature",0.0)==0.0 && job.options.value("repetition_penalty",1.0)==1.0) {
  auto best=at::max(job.logits.reshape({-1}),0);
  // Vocabulary IDs fit exactly in FP64. Transfer both scalars in one host synchronization.
  auto pair=at::stack({std::get<0>(best).to(at::kDouble),std::get<1>(best).to(at::kDouble)}).to(at::kCPU);
  auto values=pair.data_ptr<double>();TORCH_CHECK(std::isfinite(values[0]),"Nonfinite logits");return int64_t(values[1]);
 }
 auto cpu=job.logits.reshape({-1}).to(at::kCPU).to(at::kFloat).contiguous();auto data=cpu.data_ptr<float>();size_t count=cpu.numel();
 std::vector<std::pair<float,int64_t>> candidates;candidates.reserve(count);
 double temperature=job.options.value("temperature",0.0),top_p=job.options.value("top_p",1.0);int top_k=job.options.value("top_k",0);
 double penalty=job.options.value("repetition_penalty",1.0);std::set<int64_t> repeated(job.generated.begin(),job.generated.end());
 for(size_t i=0;i<count;i++){if(constrained&&!allowed[i])continue;float value=data[i];TORCH_CHECK(std::isfinite(value),"Nonfinite logits");if(repeated.count(i))value=value<0?value*penalty:value/penalty;candidates.emplace_back(value,i);}
 TORCH_CHECK(!candidates.empty(),"JSON grammar has no legal token");count=candidates.size();
 if(temperature==0)return std::max_element(candidates.begin(),candidates.end(),[](auto a,auto b){return a.first<b.first;})->second;
 auto compare=[](auto a,auto b){return a.first==b.first?a.second<b.second:a.first>b.first;};
 if(top_k>0&&top_k<int(count)){std::partial_sort(candidates.begin(),candidates.begin()+top_k,candidates.end(),compare);candidates.resize(top_k);}else std::sort(candidates.begin(),candidates.end(),compare);
 std::vector<double> probabilities;double total=0;for(auto pair:candidates){double p=std::exp((pair.first-candidates[0].first)/temperature);probabilities.push_back(p);total+=p;}
 double cumulative=0;size_t keep=0;for(;keep<probabilities.size();keep++){cumulative+=probabilities[keep]/total;if(cumulative>=top_p){keep++;break;}}probabilities.resize(keep);
 std::discrete_distribution<size_t> distribution(probabilities.begin(),probabilities.end());return candidates[distribution(job.rng)].second;
}
int main(int argc,char** argv){
 MPI_Init(&argc,&argv);int rank,world,local;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&world);
 MPI_Comm host;MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&host);MPI_Comm_rank(host,&local);ncclComm_t comm=nullptr;
 try {
  std::string model;int capacity=20480,concurrency=2,chunk=128,queue_limit=64;avi::EngineOptions options;uint64_t workspace_bytes=8ULL<<30;
  for(int i=1;i<argc;i++){std::string key=argv[i];if(key=="--baseline"){options.optimized=false;continue;}if(key=="--cuda-graph"){options.cuda_graph=true;continue;}
   TORCH_CHECK(i+1<argc,"Missing value for ",key);std::string value=argv[++i];
   if(key=="--model")model=value;else if(key=="--max-context")capacity=std::stoi(value);else if(key=="--max-concurrency")concurrency=std::stoi(value);else if(key=="--prefill-chunk")chunk=std::stoi(value);
   else if(key=="--prefix-cache-bytes")options.prefix_cache_bytes=std::stoull(value);else if(key=="--workspace-mib")workspace_bytes=std::stoull(value)<<20;else if(key=="--host-prefix-cache-mib")options.host_prefix_cache_bytes=std::stoull(value)<<20;
   else if(key=="--image-cache-mib")options.image_cache_bytes=std::stoull(value)<<20;else if(key=="--prefix-cache-mib")options.prefix_cache_bytes=std::stoull(value)<<20;else { TORCH_CHECK(false,"Unknown option ",key); }
  }
  TORCH_CHECK(!model.empty()&&capacity>0&&chunk>0&&concurrency>0&&concurrency<=8,"Invalid worker configuration");
  int nlocal,count;MPI_Comm_size(host,&nlocal);TORCH_CHECK(nlocal==world&&(world==1||world==2||world==4),"Use 1/2/4 local ranks");
  C10_CUDA_CHECK(cudaGetDeviceCount(&count));TORCH_CHECK(count>=world,"All ranks must see the same GPUs");C10_CUDA_CHECK(cudaSetDevice(local));
  cudaDeviceProp prop;C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,local));TORCH_CHECK(prop.major==8&&prop.minor==0,"Requires SM80");
  if(world>1){ncclUniqueId id;if(!rank){ TORCH_CHECK(ncclGetUniqueId(&id)==ncclSuccess,"NCCL id failed"); }MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD);TORCH_CHECK(ncclCommInitRank(&comm,world,id,rank)==ncclSuccess,"NCCL init failed");}
  TORCH_CHECK(!std::filesystem::exists(model+"/INCOMPLETE"),"Incomplete model export");
  { // Scope engine/graphs inside the lifetime of comm.
  c10::InferenceMode guard;at::set_num_threads(1);avi::Engine engine(model,rank,world,local,comm,capacity,options);
  // Reclaim load-time temporary allocator blocks before measuring device headroom.
  C10_CUDA_CHECK(cudaDeviceSynchronize());c10::cuda::CUDACachingAllocator::emptyCache();
  size_t free_bytes,total_bytes;C10_CUDA_CHECK(cudaMemGetInfo(&free_bytes,&total_bytes));
  uint64_t local_free=free_bytes,min_free=0;MPI_Allreduce(&local_free,&min_free,1,MPI_UINT64_T,MPI_MIN,MPI_COMM_WORLD);
  auto overhead=avi::checked_add(workspace_bytes,avi::checked_add(options.image_cache_bytes,options.prefix_cache_bytes));
  TORCH_CHECK(min_free>overhead,"Insufficient memory after weights for workspace/cache reserves");avi::MemoryBudget memory(min_free-overhead);
  if(!rank) {
   auto manifest=avi::read_json(model+"/manifest.json");if(manifest.value("json_grammar_supported",false)) {
    auto table=avi::read_json(model+"/token_bytes.json");for(auto& entry:table){std::string hex=entry.get<std::string>(),bytes;TORCH_CHECK(hex.size()%2==0,"Invalid token byte encoding");for(size_t i=0;i<hex.size();i+=2)bytes.push_back(char(std::stoul(hex.substr(i,2),nullptr,16)));vocabulary.push_back(std::move(bytes));}
   }
  }
  if(!rank&&!vocabulary.empty()){ TORCH_CHECK(vocabulary.size()==engine.config().at("text_config").at("vocab_size").get<size_t>(),"Vocabulary size mismatch"); }
  int grammar_available=!vocabulary.empty();MPI_Bcast(&grammar_available,1,MPI_INT,0,MPI_COMM_WORLD);
  emit(rank,{{"event","ready"},{"tp",world},{"max_context",capacity},{"max_concurrency",concurrency},{"session_budget_bytes",memory.limit()},{"workspace_bytes",workspace_bytes}});
  std::deque<Job> pending;std::map<int,Job> active;std::set<std::string> ids;std::string input;bool stopping=false;int slot_serial=1;size_t turn=0;
  while(!stopping||!pending.empty()||!active.empty()){
   json commands=json::array();
   if(!rank&&!stopping){pollfd descriptor{STDIN_FILENO,POLLIN,0};int wait=active.empty()&&pending.empty()?10:0;
    if(poll(&descriptor,1,wait)>0){char buf[65536];ssize_t read_bytes=read(STDIN_FILENO,buf,sizeof(buf));if(read_bytes<=0)commands.push_back({{"op","shutdown"}});else input.append(buf,read_bytes);}
    TORCH_CHECK(input.size()<4*1024*1024,"IPC command buffer exceeded");size_t end;
    while((end=input.find('\n'))!=std::string::npos){auto line=input.substr(0,end);input.erase(0,end+1);try{commands.push_back(json::parse(line));}catch(const std::exception& e){emit(rank,{{"event","protocol_error"},{"message",e.what()}});}}
   }
   commands=broadcast(commands,rank);
   for(auto& command:commands){auto op=command.value("op",std::string());auto id=command.value("id",std::string());
    if(op=="shutdown"){stopping=true;continue;}
    if(op=="cancel"){for(auto& j:pending)if(j.id==id)j.cancelled=true;for(auto& item:active)if(item.second.id==id)item.second.cancelled=true;continue;}
    if(op=="stats"){emit(rank,{{"event","stats"},{"active",active.size()},{"pending",pending.size()},{"cache",engine.cache_stats()},{"reserved_bytes",memory.used()},{"budget_bytes",memory.limit()}});continue;}
    if(op!="submit"){emit(rank,{{"event","error"},{"id",id},{"message","Unknown operation"}});continue;}
    if(stopping||ids.count(id)||id.empty()||pending.size()>=size_t(queue_limit)){emit(rank,{{"event","error"},{"id",id},{"message","Duplicate ID, shutting down, or queue full"}});continue;}
    try {
    Job job;job.id=id;job.path=command.at("request");job.options=command.value("sampling",json::object());job.rng.seed(job.options.value("seed",uint64_t(42)));
    double timeout=job.options.value("timeout_seconds",300.0);TORCH_CHECK(std::isfinite(timeout)&&timeout>=0,"Invalid timeout");
    double temp=job.options.value("temperature",0.0),p=job.options.value("top_p",1.0),pen=job.options.value("repetition_penalty",1.0);int k=job.options.value("top_k",0);
    if(!std::isfinite(temp)||temp<0||!std::isfinite(p)||p<=0||p>1||k<0||!std::isfinite(pen)||pen<=0){emit(rank,{{"event","error"},{"id",id},{"message","Invalid sampling parameters"}});continue;}
    if(job.options.value("json_object",false)&&!grammar_available){emit(rank,{{"event","error"},{"id",id},{"message","Artifact lacks ByteLevel vocabulary; reconvert with tokenizer.json"}});continue;}
    // Rank zero validates metadata; all ranks receive the same admission decision.
    json metadata;
    if(!rank){try{metadata={{"ok",true},{"request",avi::read_json(job.path+"/request.json")}};}catch(const std::exception& e){metadata={{"ok",false},{"message",e.what()}};}}
    metadata=broadcast(metadata,rank);TORCH_CHECK(metadata.at("ok").get<bool>(),metadata.value("message",std::string("Invalid request")));
    job.req=metadata.at("request");TORCH_CHECK(job.req.at("format")=="avi-request-v1","Invalid request format");
    int64_t prompt=job.req.at("input_ids").at("shape").at(0).get<int64_t>(),output=job.req.at("max_new_tokens");
    TORCH_CHECK(prompt>0&&output>0&&prompt<=capacity&&output<=capacity-prompt,"Invalid token budget");
    job.token_capacity=int(prompt+output);job.reservation=avi::session_bytes(engine.config(),world,job.token_capacity);
    TORCH_CHECK(job.reservation<=memory.limit(),"Request exceeds session memory budget; lower max_tokens/context/cache or workspace reserve");
    ids.insert(id);pending.push_back(std::move(job));emit(rank,{{"event","queued"},{"id",id}});
    }catch(const std::exception& e){emit(rank,{{"event","error"},{"id",id},{"message",e.what()}});}
   }
   json queued_expired=json::array();
   if(!rank)for(auto& j:pending){double timeout=j.options.value("timeout_seconds",300.0);if(timeout>0&&age(j.submitted)>timeout)queued_expired.push_back(j.id);}
   queued_expired=broadcast(queued_expired,rank);
   for(auto it=pending.begin();it!=pending.end();){bool expired=std::find(queued_expired.begin(),queued_expired.end(),json(it->id))!=queued_expired.end();
     if(it->cancelled||expired){emit(rank,{{"event","done"},{"id",it->id},{"finish_reason",it->cancelled?"cancelled":"timeout"},{"input_tokens",0},{"generated_ids",json::array()}});ids.erase(it->id);it=pending.erase(it);}else ++it;
   }
   while(active.size()<size_t(concurrency)&&!pending.empty()){
    if(!pending.front().cancelled&&!memory.fits(pending.front().reservation))break;
    Job job=std::move(pending.front());pending.pop_front();if(job.cancelled){emit(rank,{{"event","done"},{"id",job.id},{"finish_reason","cancelled"},{"input_tokens",0},{"generated_ids",json::array()}});ids.erase(job.id);continue;}
    job.slot=slot_serial++;TORCH_CHECK(memory.reserve(job.slot,job.reservation),"Reservation mismatch");engine.reserve_session(job.slot,job.token_capacity);active.emplace(job.slot,std::move(job));
   }
   // Only rank zero reads wall time; every rank must take identical collective paths.
   json expired=json::array();
   if(!rank)for(auto& item:active){double timeout=item.second.options.value("timeout_seconds",300.0);if(timeout>0&&age(item.second.submitted)>timeout)expired.push_back(item.first);}
   expired=broadcast(expired,rank);
   for(auto& item:active)if(item.second.cancelled)item.second.finish="cancelled";
   for(auto& slot:expired)active.at(slot.get<int>()).finish="timeout";
   // One prefill chunk per iteration, round-robin. Ready requests decode every iteration.
   std::vector<int> prefill;for(auto& item:active)if(!item.second.ready&&item.second.finish.empty())prefill.push_back(item.first);
   if(!prefill.empty()){
    auto& j=active.at(prefill[(turn++)%prefill.size()]);engine.activate(j.slot);
    if(!j.initialized){
     TORCH_CHECK(j.req.at("format")=="avi-request-v1","Invalid request");
     j.ids=engine.read_input(j.path,j.req.at("input_ids"));j.positions=engine.read_input(j.path,j.req.at("positions"));
     int n=j.ids.numel(),max_new=j.req.at("max_new_tokens");TORCH_CHECK(j.ids.dim()==1&&j.ids.scalar_type()==at::kLong&&j.positions.dim()==2&&j.positions.size(0)==3&&j.positions.size(1)==n&&n>0&&max_new>0&&n+max_new<=capacity&&n+max_new<=j.req.at("max_context").get<int>(),"Invalid input/budget");
     int vocab=engine.config().at("text_config").at("vocab_size");TORCH_CHECK(j.ids.min().item<int64_t>()>=0&&j.ids.max().item<int64_t>()<vocab,"Token outside vocabulary");
     j.next=j.req.at("next_position");TORCH_CHECK(j.next==j.positions.max().item<int64_t>()+1,"Invalid continuation position");
     TORCH_CHECK(j.req.at("image_token_id")==engine.config().at("image_token_id"),"Image token mismatch");
     j.key=avi::request_hash(j.path,j.req);j.logits=engine.restore_prefix(j.key);j.initialized=true;
     if(j.logits.defined()){j.ready=true;j.consumed=n;j.offset=n;emit(rank,{{"event","prefix_hit"},{"id",j.id}});}
     else {auto vision=engine.vision(j.path,j.req);j.embedding=engine.embed(j.ids);auto indices=at::nonzero(j.ids==j.req.at("image_token_id").get<int64_t>()).reshape({-1});
      if(vision.defined()){TORCH_CHECK(vision.size(0)==indices.numel(),"Visual token mismatch");j.embedding.index_copy_(0,indices,vision);}else { TORCH_CHECK(indices.numel()==0,"Missing image features"); }}
    }
    if(!j.ready){int n=std::min<int64_t>(chunk,j.ids.numel()-j.offset);auto hidden=engine.step(j.embedding.narrow(0,j.offset,n),j.positions.narrow(1,j.offset,n));j.offset+=n;
     if(j.offset==j.ids.numel()){j.logits=engine.logits(hidden.narrow(0,n-1,1));j.ready=true;j.consumed=j.offset;engine.save_prefix(j.key,j.logits);j.embedding=at::Tensor();}}
   }
   std::vector<int> slots,consumed;std::vector<int64_t> tokens,positions;
   for(auto& item:active){auto& j=item.second;if(j.ready&&j.needs_decode&&j.finish.empty()){slots.push_back(item.first);consumed.push_back(j.consumed);tokens.push_back(j.generated.back());positions.push_back(j.next);}}
   if(!slots.empty()){auto batch=engine.decode_batch(slots,tokens,positions,consumed);for(size_t i=0;i<slots.size();i++){auto& j=active.at(slots[i]);j.logits=batch.narrow(0,i,1);j.needs_decode=false;j.consumed++;j.next++;}}
   for(auto& item:active){auto& j=item.second;if(!j.ready||!j.finish.empty())continue;int64_t token=0;if(!rank)token=sample(j);MPI_Bcast(&token,1,MPI_INT64_T,0,MPI_COMM_WORLD);if(!rank&&j.options.value("json_object",false)&&token>=0&&size_t(token)<vocabulary.size()&&!vocabulary[token].empty()){ TORCH_CHECK(j.grammar.feed(vocabulary[token]),"Grammar state mismatch"); }j.generated.push_back(token);j.needs_decode=true;
    if(j.first_token<0)j.first_token=age(j.submitted);emit(rank,{{"event","token"},{"id",j.id},{"token",token},{"index",j.generated.size()-1}});
    auto eos=j.req.at("eos_token_ids").get<std::vector<int64_t>>();if(std::find(eos.begin(),eos.end(),token)!=eos.end())j.finish="eos";else if(j.generated.size()>=j.req.at("max_new_tokens").get<size_t>())j.finish="length";
   }
   for(auto it=active.begin();it!=active.end();){auto& j=it->second;if(j.finish.empty()){++it;continue;}
    emit(rank,{{"event","done"},{"id",j.id},{"finish_reason",j.finish},{"generated_ids",j.generated},{"input_tokens",j.initialized?j.ids.numel():0},{"ttft_seconds",j.first_token},{"total_seconds",age(j.submitted)},{"cache",engine.cache_stats()},{"json_complete",j.options.value("json_object",false)&&j.grammar.complete()}});
    engine.drop(it->first);memory.release(it->first);ids.erase(j.id);it=active.erase(it);
   }
  }
  std::cerr<<"Rank "<<rank<<": releasing engine and CUDA graphs\n";
  }
  std::cerr<<"Rank "<<rank<<": destroying NCCL communicator\n";
  if(comm) { TORCH_CHECK(ncclCommDestroy(comm)==ncclSuccess,"NCCL destroy failed"); }
  emit(rank,{{"event","stopped"}});MPI_Comm_free(&host);MPI_Finalize();return 0;
 }catch(const std::exception& e){emit(rank,{{"event","fatal"},{"message",e.what()}});std::cerr<<"Rank "<<rank<<": "<<e.what()<<"\n";MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}