#include "avi/engine.h"
#include <mpi.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/core/InferenceMode.h>
#include <ATen/Parallel.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <set>
using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
static void dump(const std::string& path,at::Tensor tensor) {
  tensor=tensor.to(at::kCPU).to(at::kFloat).contiguous();
  TORCH_CHECK(!std::filesystem::exists(path),"Refusing to overwrite ",path);
  std::ofstream f(path,std::ios::binary); f.write(static_cast<const char*>(tensor.data_ptr()),tensor.nbytes());
  TORCH_CHECK(f.good(),"Cannot write ",path);
}
int main(int argc,char** argv) {
  MPI_Init(&argc,&argv); int rank,world,local;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&world);
  MPI_Comm local_comm; MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&local_comm);
  MPI_Comm_rank(local_comm,&local);
  ncclComm_t comm=nullptr;
  try {
    std::string model,request,output; int chunk=128;
    bool trace=false,layer_trace=false,draft_graph=false,gdn_chunk=false; avi::EngineOptions options;
    for(int i=1;i<argc;i++) {
      std::string key=argv[i];
      if(key=="--baseline"){options.optimized=false;continue;}
      if(key=="--mtp-draft-graph"){draft_graph=true;continue;}
      if(key=="--gdn-chunk"){gdn_chunk=true;continue;}
      if(key=="--flash-prefill"){options.flash_prefill=true;continue;}
      if(key=="--gdn-cooperative"){options.gdn_cooperative=true;continue;}
      if(key=="--gdn-wy"){options.gdn_wy=true;continue;}
      if(key=="--audit-logits"){options.audit_logits=true;continue;}
      if(key=="--multi-token-gemv"){options.multi_token_gemv=true;continue;}
      if(key=="--multi-token-gemv-fp8"){options.multi_token_gemv=true;options.shared_gemv_fp8=true;continue;}
      if(key=="--fused-gdn-prepare"){options.fused_gdn_prepare=true;continue;}
      if(key=="--gdn-wy-fused"){options.gdn_wy_fused=true;continue;}
      if(key=="--reuse-verify-graph"){options.reuse_verify_graph=true;continue;}
      if(key=="--mtp-verify-graph"){options.mtp_verify_graph=true;continue;}
      if(key=="--fused-gdn-conv"){options.fused_gdn_conv=true;continue;}
      if(key=="--bf16-tp-reduce"){options.bf16_tp_reduce=true;continue;}
      if(key=="--cache-vision-weights"){options.cache_vision_weights=true;continue;}
      if(key=="--cuda-graph"){options.cuda_graph=true;continue;}
      if(key=="--extra-fusions"){options.extra_fusions=true;continue;}
      if(key=="--cublas-prefill"){options.cublas_prefill=true;continue;}
      if(key=="--tp-lm-head"){options.tp_lm_head=true;continue;}
      if(key=="--reference-prefill"){options.reference_prefill=true;continue;}
      if(key=="--vector-gemv"){options.vector_gemv=true;continue;}
      if(key=="--layer-trace"){layer_trace=true;continue;}
      if(key=="--trace") { trace=true; continue; }
      TORCH_CHECK(i+1<argc,"Missing argument for ",key); std::string value=argv[++i];
      if(key=="--model") model=value; else if(key=="--request") request=value;
      else if(key=="--output") output=value; else if(key=="--prefill-chunk") chunk=std::stoi(value);
      else if(key=="--mtp-tokens") options.mtp_tokens=std::stoi(value);
      else if(key=="--weight-cache-mib") options.weight_cache_bytes=std::stoull(value)<<20;
      else { TORCH_CHECK(false,"Unknown option ",key); }
    }
    TORCH_CHECK(!model.empty() && !request.empty() && !output.empty(),"Usage: avi-infer --model ARTIFACT --request REQUEST_DIR --output result.json [--prefill-chunk 128] [--trace]");
    TORCH_CHECK(world==1 || world==2 || world==4,"Use 1, 2 or 4 ranks on a single node");
    int local_world; MPI_Comm_size(local_comm,&local_world); TORCH_CHECK(local_world==world,"Multi-node is not implemented");
    TORCH_CHECK(chunk>0,"Prefill chunk must be positive");
    TORCH_CHECK(!layer_trace||!options.cuda_graph,"Layer trace requires eager execution; use a separate untraced Graph benchmark");
    TORCH_CHECK(!options.mtp_tokens||(!trace&&!layer_trace),"Use non-MTP runs for per-token traces");
    TORCH_CHECK(!draft_graph||options.mtp_tokens>0,"Draft Graph requires --mtp-tokens");
    TORCH_CHECK(!std::filesystem::exists(output),"Refusing to overwrite output ",output);
    TORCH_CHECK(!std::filesystem::exists(model+"/INCOMPLETE"),"Incomplete model conversion");
    int count; C10_CUDA_CHECK(cudaGetDeviceCount(&count)); TORCH_CHECK(count>=world,"Each rank must see all requested GPUs");
    C10_CUDA_CHECK(cudaSetDevice(local)); cudaDeviceProp prop; C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,local));
    TORCH_CHECK(prop.major==8 && prop.minor==0,"This build targets A100 SM80; got ",prop.name);
    if(world>1) {
      ncclUniqueId id; if(rank==0) { TORCH_CHECK(ncclGetUniqueId(&id)==ncclSuccess,"NCCL unique ID failed"); }
      MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD);
      TORCH_CHECK(ncclCommInitRank(&comm,world,id,rank)==ncclSuccess,"NCCL initialization failed");
    }
    { // Engine and captured graphs must die before their NCCL communicator.
    c10::InferenceMode inference_guard; at::set_num_threads(1);
    auto req=avi::read_json(request+"/request.json"); TORCH_CHECK(req.at("format")=="avi-request-v1","Unsupported request format");
    int capacity=req.at("max_context"), max_new=req.at("max_new_tokens");
    TORCH_CHECK(capacity>0 && max_new>0,"Invalid request budgets");
    if(rank==0) std::cerr<<"Loading native model, TP="<<world<<"; no Python model runtime\n";
    auto start=Clock::now(); avi::Engine engine(model,rank,world,local,comm,capacity,options);
    engine.enable_draft_graph(draft_graph);
    engine.enable_gdn_chunk(gdn_chunk);
    double load_time=seconds(start); auto ids=engine.read_input(request,req.at("input_ids"));
    auto pos=engine.read_input(request,req.at("positions"));
    TORCH_CHECK(ids.dim()==1 && ids.scalar_type()==at::kLong && pos.scalar_type()==at::kLong && pos.dim()==2 && pos.size(0)==3 && pos.size(1)==ids.numel(),"Invalid ids/position layout");
    TORCH_CHECK(ids.numel()>0 && ids.numel()+max_new<=capacity,"Token budget exceeds capacity");
    auto cfg=engine.config(); int vocab=cfg.at("text_config").at("vocab_size");
    TORCH_CHECK(ids.min().item<int64_t>()>=0 && ids.max().item<int64_t>()<vocab,"Token outside vocabulary");
    TORCH_CHECK(req.at("image_token_id")==cfg.at("image_token_id"),"Request/model image token mismatch");
    int64_t next=req.at("next_position"); TORCH_CHECK(next==pos.max().item<int64_t>()+1,"Incorrect continuation position");
    start=Clock::now(); auto vision=engine.vision(request,req); C10_CUDA_CHECK(cudaDeviceSynchronize());
    double vision_time=seconds(start); auto embedding=engine.embed(ids);
    auto image_indices=at::nonzero(ids==cfg.at("image_token_id").get<int64_t>()).reshape({-1});
    if(vision.defined()) {
      TORCH_CHECK(vision.size(0)==image_indices.numel(),"Image features/tokens mismatch");
      embedding.index_copy_(0,image_indices,vision);
      if(trace && rank==0) dump(output+".vision.f32",vision);
    } else { TORCH_CHECK(image_indices.numel()==0,"Image tokens without images"); }
    start=Clock::now(); at::Tensor last;
    for(int64_t i=0;i<ids.numel();i+=chunk) {
      auto n=std::min<int64_t>(chunk,ids.numel()-i);
      if(layer_trace&&i+n==ids.numel())engine.set_trace_prefix(output+".prefill_layers");
      last=engine.step(embedding.narrow(0,i,n),pos.narrow(1,i,n));
      engine.set_trace_prefix("");
      if(rank==0) std::cerr<<"Prefill "<<i+n<<"/"<<ids.numel()<<"\n";
    }
    auto logits=engine.logits(last.narrow(0,last.size(0)-1,1));
    C10_CUDA_CHECK(cudaDeviceSynchronize()); double prefill_time=seconds(start);
    if(trace && rank==0) dump(output+".prefill_logits.f32",logits);
    auto eos_vec=req.at("eos_token_ids").get<std::vector<int64_t>>(); std::set<int64_t> eos(eos_vec.begin(),eos_vec.end());
    std::vector<int64_t> generated; start=Clock::now(); std::string reason="length";
    int proposed=0,accepted=0,rounds=0;
    if(options.mtp_tokens){
      generated.push_back(engine.greedy(logits));
      int consumed=ids.numel();
      while(int(generated.size())<max_new&&!eos.count(generated.back())){
        auto batch=engine.speculate(generated.back(),next,consumed,max_new-int(generated.size()),eos_vec);
        generated.insert(generated.end(),batch.tokens.begin(),batch.tokens.end());
        consumed+=batch.consumed;next+=batch.consumed;proposed+=batch.proposed;accepted+=batch.accepted;++rounds;
      }
      if(eos.count(generated.back()))reason="eos";
    }else for(int i=0;i<max_new;i++) {
      int64_t token=0; if(rank==0) {
        TORCH_CHECK(at::isfinite(logits).all().item<bool>(),"Nonfinite logits; stop and check numerical trace");
        token=logits.argmax(-1).item<int64_t>();
      }
      MPI_Bcast(&token,1,MPI_INT64_T,0,MPI_COMM_WORLD); generated.push_back(token);
      if(eos.count(token)) { reason="eos"; break; }
      if(i+1==max_new) break;
      if(layer_trace&&i+1==74)engine.set_trace_prefix(output+".decode74_layers");
      logits=engine.decode(token,next++,ids.numel()+i);
      engine.set_trace_prefix("");
      if(trace && rank==0) dump(output+".decode_"+std::to_string(i+1)+".f32",logits);
    }
    C10_CUDA_CHECK(cudaDeviceSynchronize()); double decode_time=seconds(start);
    if(rank==0) {
      avi::json result={{"format","avi-result-v1"},{"generated_ids",generated},{"finish_reason",reason},
                        {"input_tokens",ids.numel()},{"tp",world},{"backend","native-libtorch-cuda"},
                        {"optimized",options.optimized},{"cuda_graph",options.cuda_graph},{"tp_lm_head",options.tp_lm_head},{"reference_prefill",options.reference_prefill},{"layer_trace",layer_trace},{"load_seconds",load_time},{"vision_seconds",vision_time},{"prefill_seconds",prefill_time},
                        {"decode_loop_seconds",decode_time},{"note","First generated token comes from prefill; timings exclude Python preprocessing. Experimental, not benchmark-certified."}};
      result["mtp"]={{"window",options.mtp_tokens},{"rounds",rounds},{"proposed",proposed},{"accepted",accepted}};
      result["weight_cache_bytes"]=options.weight_cache_bytes;
      result["mtp_draft_graph"]=draft_graph;
      result["cache"]=engine.cache_stats();
      result["gdn_chunk"]=gdn_chunk;
      std::ofstream f(output); f<<result.dump(2)<<"\n"; TORCH_CHECK(f.good(),"Cannot write output");
      std::cerr<<"Finished "<<generated.size()<<" tokens; output "<<output<<"\n";
    }
    std::cerr<<"Rank "<<rank<<": releasing engine and CUDA graphs\n";
    }
    std::cerr<<"Rank "<<rank<<": destroying NCCL communicator\n";
    if(comm) { TORCH_CHECK(ncclCommDestroy(comm)==ncclSuccess,"NCCL destroy failed"); }
    std::cerr<<"Rank "<<rank<<": finalizing MPI\n";
    MPI_Comm_free(&local_comm); MPI_Finalize(); return 0;
  } catch(const std::exception& e) {
    std::cerr<<"Rank "<<rank<<": "<<e.what()<<"\n";
    MPI_Abort(MPI_COMM_WORLD,1); return 1;
  }
}
