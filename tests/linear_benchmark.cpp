#include "avi/engine.h"
#include "avi/ops.h"
#include "avi/tilelang.h"
#include <c10/cuda/CUDAException.h>
#include <c10/core/InferenceMode.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <functional>
struct Event {cudaEvent_t v;Event(){C10_CUDA_CHECK(cudaEventCreate(&v));}~Event(){cudaEventDestroy(v);}};
int main(int argc,char** argv){
 try{
  c10::InferenceMode guard;C10_CUDA_CHECK(cudaSetDevice(0));int repeats=7;bool head=false;at::manual_seed(915);
  for(int i=1;i<argc;i++){
   std::string key=argv[i];if(key=="--lm-head"){head=true;continue;}
   TORCH_CHECK(i+1<argc,"Missing value for ",key);std::string value=argv[++i];
   if(key=="--tilelang-dir")avi::configure_tilelang(value);
   else if(key=="--repeats")repeats=std::stoi(value);else TORCH_CHECK(false,"Unknown option ",key);
  }
  TORCH_CHECK(repeats>=3&&repeats<=100,"repeats must be 3..100");
  std::vector<std::pair<int,int>> shapes={{5120,5120},{17408,5120},{5120,8704},{8240,5120}};
  if(head)shapes.push_back({124160,5120});
  auto f=at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
  avi::json report={{"format","avi-linear-bench-v1"},{"profiles",avi::json::array()},
   {"scope","Resident rank-local W8A16 linear; cached BF16 decode excluded. Event intervals include host submission gaps. End-to-end speed is tested separately."}};
  bool passed=true;
  for(auto [N,K]:shapes){
   auto codes=at::randint(0,256,{N,K},f.dtype(at::kLong)).to(at::kByte);
   codes.masked_fill_(codes==127,0);codes.masked_fill_(codes==255,0);
   auto scales=at::rand({N,K/128},f)*.001+.0001;
   auto decoded=avi::fp8_decode(codes,scales);
   for(int M:{2,4,8}){
    auto x=(at::randn({M,K},f)*.1).to(at::kBFloat16);
    auto ref=at::matmul(x,decoded.t()).to(at::kFloat);
    std::vector<std::pair<std::string,std::function<at::Tensor()>>> variants={
     {"cached-bf16",[&](){return at::matmul(x,decoded.t());}},
     {"shared-fp8",[&](){return avi::small_linear_shared(x,codes,scales);}},
     {"cuda-tensor-s1",[&](){return avi::fp8_tensor_small(x,codes,scales,1);}},
     {"cuda-tensor-s4",[&](){return avi::fp8_tensor_small(x,codes,scales,4);}}};
    if(avi::tilelang_configured()){
     variants.push_back({"tilelang-s1",[&](){return avi::tilelang_fp8_small(x,codes,scales,1);}});
     variants.push_back({"tilelang-s4",[&](){return avi::tilelang_fp8_small(x,codes,scales,4);}});
    }
    for(auto& entry:variants){
     auto& invoke=entry.second;auto output=invoke().to(at::kFloat);
     bool ok=at::isfinite(output).all().item<bool>()&&at::allclose(output,ref,.01,.001);passed=passed&&ok;
     double error=(output-ref).abs().max().item<double>();
     for(int i=0;i<3;i++)invoke();C10_CUDA_CHECK(cudaDeviceSynchronize());
     Event begin,end;std::vector<double> times,walls;
     for(int i=0;i<repeats;i++){
      C10_CUDA_CHECK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();
      C10_CUDA_CHECK(cudaEventRecord(begin.v));output=invoke();C10_CUDA_CHECK(cudaEventRecord(end.v));
      C10_CUDA_CHECK(cudaEventSynchronize(end.v));float ms;C10_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.v,end.v));
      times.push_back(ms);walls.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
     }
     std::sort(times.begin(),times.end());std::sort(walls.begin(),walls.end());
     avi::json row={{"backend",entry.first},{"M",M},{"N",N},{"K",K},{"numerical_passed",ok},{"max_abs_error",error},
      {"event_interval_p50_ms",times[times.size()/2]},{"wall_p50_ms",walls[walls.size()/2]}};
     report["profiles"].push_back(row);std::cerr<<row.dump()<<"\n";
    }
   }
  }
  report["numerical_passed"]=passed;std::cout<<report.dump(2)<<"\n";return passed?0:2;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
