#include "avi/ops.h"
#include "avi/tilelang.h"
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <iostream>
namespace {
// Decode on CPU independently, including the BF16 rounding that defines W8A16.
at::Tensor reference(at::Tensor x,at::Tensor codes,at::Tensor scales){
 auto c=codes.cpu().contiguous(),s=scales.cpu().contiguous();
 auto weight=at::empty(c.sizes(),s.options());auto cp=c.data_ptr<unsigned char>();auto sp=s.data_ptr<float>(),wp=weight.data_ptr<float>();
 int N=c.size(0),K=c.size(1);
 for(int n=0;n<N;n++)for(int k=0;k<K;k++){
  int b=cp[n*K+k],e=(b>>3)&15,m=b&7;
  float v=e?std::ldexp(1.f+m/8.f,e-7):std::ldexp(float(m),-9);
  if(e==15&&m==7)v=NAN;if(b&128)v=-v;
  wp[n*K+k]=v*sp[n*(K/128)+k/128];
 }
 return at::matmul(x.cpu().to(at::kDouble),weight.to(at::kBFloat16).to(at::kDouble).t()).to(at::kBFloat16).to(at::kFloat);
}
void check(at::Tensor output,at::Tensor expected){
 TORCH_CHECK(at::isfinite(output).all().item<bool>()&&
  at::allclose(output.cpu().to(at::kFloat),expected,.01,.001),"W8A16 independent FP64 oracle mismatch");
}
}
void run_fp8_tensor_tests(){
 auto f=at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);at::manual_seed(915);
 for(int M=2;M<=8;M++)for(int K:{128,384,5120})for(int N:{17,80,144}){
  auto x=(at::randn({M,K},f)*.1).to(at::kBFloat16);
  auto codes=at::randint(0,256,{N,K},f.dtype(at::kLong)).to(at::kByte);
  codes.masked_fill_(codes==127,0);codes.masked_fill_(codes==255,0);
  auto scales=at::rand({N,K/128},f)*.001+.0001,expected=reference(x,codes,scales);
  for(int split:{1,4}){
   check(avi::fp8_tensor_small(x,codes,scales,split),expected);
   if(avi::tilelang_configured()&&N%16==0)check(avi::tilelang_fp8_small(x,codes,scales,split),expected);
  }
 }
 // Graph replay sees updated activations, with no Python or compilation.
 for(bool plugin:{false,true}){
  if(plugin&&!avi::tilelang_configured())continue;
  auto x=(at::randn({4,384},f)*.1).to(at::kBFloat16);
  auto codes=at::full({144,384},24,f.dtype(at::kByte)),scales=at::rand({144,3},f)*.1;
  auto invoke=[&](){return plugin?avi::tilelang_fp8_small(x,codes,scales,4):avi::fp8_tensor_small(x,codes,scales,4);};
  auto stream=c10::cuda::getStreamFromPool(false);at::cuda::CUDAGraph graph;at::Tensor out;
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  {c10::cuda::CUDAStreamGuard guard(stream);out=invoke();C10_CUDA_CHECK(cudaStreamSynchronize(stream));
   graph.capture_begin();out=invoke();graph.capture_end();}
  for(int i=0;i<2;i++){if(i)x.add_(.015625);C10_CUDA_CHECK(cudaDeviceSynchronize());
   graph.replay();C10_CUDA_CHECK(cudaDeviceSynchronize());check(out,reference(x,codes,scales));}
 }
 std::cout<<"FP8 W8A16: FP64 oracle, 2..8 rows, tails, empty split partitions and graph replay passed\n";
}
