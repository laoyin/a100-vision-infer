#include "avi/ops.h"
#include "avi/tilelang.h"
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/core/InferenceMode.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
struct Event {cudaEvent_t value;Event(){C10_CUDA_CHECK(cudaEventCreate(&value));}~Event(){cudaEventDestroy(value);}};
int main(int argc,char** argv){
 try{
  c10::InferenceMode guard;C10_CUDA_CHECK(cudaSetDevice(0));at::manual_seed(314);
  int repeats=9;std::vector<int> sizes={512,2048,3914};
  for(int i=1;i<argc;i++){
   std::string key=argv[i];TORCH_CHECK(i+1<argc,"Missing value for ",key);std::string argument=argv[++i];
   if(key=="--tilelang-dir"){avi::configure_tilelang(argument);continue;}
   int value=std::stoi(argument);
   if(key=="--tokens")sizes={value};else if(key=="--repeats")repeats=value;else TORCH_CHECK(false,"Unknown option ",key);
  }
  TORCH_CHECK(repeats>0&&repeats<=100,"Invalid repetitions");
  auto f=at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
  nlohmann::json report={{"format","avi-gdn-bench-v1"},{"heads",24},{"key_heads",8},{"K",128},{"V",128},
   {"timing_scope","One GDN scan, resident input; CUDA event interval includes host submission gaps. End-to-end engine testing is separate."},{"profiles",nlohmann::json::array()}};
  bool passed=true;
  for(int T:sizes){
   TORCH_CHECK(T>0&&T<=20480,"Invalid token count");
   auto q=avi::fused_l2(at::randn({T,8,128},f).to(at::kBFloat16));
   auto k=avi::fused_l2(at::randn_like(q)),v=at::randn({T,24,128},f).to(at::kBFloat16);
   auto g=-at::rand({T,24},f)*.01,beta=at::rand({T,24},f),initial=at::randn({24,128,128},f)*.1;
   auto qe=q.repeat_interleave(3,1).contiguous(),ke=k.repeat_interleave(3,1).contiguous();
   auto reference_state=initial.clone();
   auto reference=avi::delta_scan_fast(qe,ke,v,g,beta,reference_state,{},true);
   C10_CUDA_CHECK(cudaDeviceSynchronize());
   std::vector<std::pair<int,int>> cases={{0,0},{1,32},{1,64},{2,32},{2,64}};
   if(avi::tilelang_configured()){cases.push_back({3,32});cases.push_back({3,64});}
   for(auto [backend,C]:cases){
    auto state=initial.clone();
    auto invoke=[&](){return C?avi::delta_scan_tensor(q,k,v,g,beta,state,C,backend==2,backend==3):avi::delta_scan_fast(qe,ke,v,g,beta,state,{},true);};
    auto output=invoke();
    bool ok=at::allclose(output.to(at::kFloat),reference.to(at::kFloat),.01,.001)&&at::allclose(state,reference_state,.0001,.00001);
    float error=(state-reference_state).abs().max().item<float>();
    passed=passed&&ok;
    for(int i=0;i<2;i++){state.copy_(initial);invoke();}C10_CUDA_CHECK(cudaDeviceSynchronize());
    Event begin,end;std::vector<double> intervals,walls;
    for(int i=0;i<repeats;i++){
     state.copy_(initial);C10_CUDA_CHECK(cudaDeviceSynchronize());
     auto start=std::chrono::steady_clock::now();C10_CUDA_CHECK(cudaEventRecord(begin.value));
     output=invoke();C10_CUDA_CHECK(cudaEventRecord(end.value));C10_CUDA_CHECK(cudaEventSynchronize(end.value));
     float ms;C10_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.value,end.value));
     intervals.push_back(ms);walls.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    std::sort(intervals.begin(),intervals.end());std::sort(walls.begin(),walls.end());
    auto row=nlohmann::json{{"backend",backend==3?"tilelang-fused":backend==2?"cuda-fused":backend==1?"tensor-wy":"register-cooperative"},{"chunk",C},{"tokens",T},{"repeats",repeats},
     {"numerical_passed",ok},{"state_max_abs",error},{"event_interval_p50_ms",intervals[intervals.size()/2]},
     {"wall_p50_ms",walls[walls.size()/2]}};
    report["profiles"].push_back(row);std::cerr<<row.dump()<<"\n";
   }
  }
  report["numerical_passed"]=passed;std::cout<<report.dump(2)<<"\n";return passed?0:2;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
