#include "avi/engine.h"
#include <cuda_runtime.h>
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <cmath>
void run_optimized_tests();
void run_tensor_gdn_tests();
int main() {
 try {
  c10::InferenceMode guard; TORCH_CHECK(cudaSetDevice(0)==cudaSuccess,"No GPU");
  auto opt=at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
  auto codes=at::arange(256,at::TensorOptions().dtype(at::kLong)).to(at::kByte).reshape({1,256}).to(at::kCUDA);
  auto scales=at::ones({1},opt); auto decoded=avi::fp8_decode(codes,scales).to(at::kCPU).to(at::kFloat);
  auto data=decoded.data_ptr<float>();
  for(int b=0;b<256;b++) {
    int e=(b>>3)&15,m=b&7; float expected=e==0?std::ldexp(float(m),-9):std::ldexp(1.f+m/8.f,e-7);
    if(b&128) expected=-expected;
    if(e==15 && m==7) { TORCH_CHECK(std::isnan(data[b]),"NaN decoding mismatch"); }
    else { TORCH_CHECK(data[b]==expected,"FP8 decoding mismatch at ",b); }
  }
  at::manual_seed(42); int T=7,H=2,K=8,V=8;
  auto q=at::randn({T,H,K},opt).to(at::kBFloat16),k=at::randn({T,H,K},opt).to(at::kBFloat16);
  auto v=at::randn({T,H,V},opt).to(at::kBFloat16),g=-at::rand({T,H},opt),b=at::rand({T,H},opt);
  auto state=at::zeros({H,K,V},opt), reference=at::zeros_like(state); std::vector<at::Tensor> outputs;
  for(int t=0;t<T;t++) {
    reference=reference*g[t].exp().unsqueeze(-1).unsqueeze(-1);
    auto kt=k[t].to(at::kFloat); auto delta=(v[t].to(at::kFloat)-(reference*kt.unsqueeze(-1)).sum(1))*b[t].unsqueeze(-1);
    reference=reference+kt.unsqueeze(-1)*delta.unsqueeze(1);
    outputs.push_back(((reference*q[t].to(at::kFloat).unsqueeze(-1)).sum(1)/std::sqrt(float(K))).to(at::kBFloat16));
  }
  auto expected=at::stack(outputs); auto out=avi::delta_scan(q,k,v,g,b,state);
  TORCH_CHECK(at::allclose(out.to(at::kFloat),expected.to(at::kFloat),0.02,0.02),"GDN output mismatch");
  TORCH_CHECK(at::allclose(state,reference,1e-4,1e-4),"GDN state mismatch");
  auto split_state=at::zeros_like(state);
  auto first=avi::delta_scan(q.narrow(0,0,3),k.narrow(0,0,3),v.narrow(0,0,3),g.narrow(0,0,3),b.narrow(0,0,3),split_state);
  auto second=avi::delta_scan(q.narrow(0,3,4),k.narrow(0,3,4),v.narrow(0,3,4),g.narrow(0,3,4),b.narrow(0,3,4),split_state);
  TORCH_CHECK(at::equal(at::cat({first,second}),out) && at::equal(split_state,state),"Chunk continuation mismatch");
  TORCH_CHECK(cudaDeviceSynchronize()==cudaSuccess,"CUDA failure");
  run_optimized_tests();
  run_tensor_gdn_tests();
  std::cout<<"FP8 decoding, GDN recurrence and continuation passed\n"; return 0;
 } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}