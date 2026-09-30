#include "avi/ops.h"
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <iostream>
#include <vector>
namespace {
// Independent scalar FP64 recurrence on the CPU; no WY or Tensor Core code.
std::pair<at::Tensor,at::Tensor> oracle(at::Tensor q,at::Tensor k,at::Tensor v,
 at::Tensor g,at::Tensor beta,at::Tensor state){
 q=q.to(at::kCPU).to(at::kDouble).contiguous();k=k.to(at::kCPU).to(at::kDouble).contiguous();
 v=v.to(at::kCPU).to(at::kDouble).contiguous();g=g.to(at::kCPU).to(at::kDouble).contiguous();
 beta=beta.to(at::kCPU).to(at::kDouble).contiguous();state=state.to(at::kCPU).to(at::kDouble).contiguous().clone();
 int T=q.size(0),HQ=q.size(1),H=v.size(1),K=128,V=v.size(2);
 auto out=at::empty({T,H,V},state.options());
 auto qp=q.data_ptr<double>(),kp=k.data_ptr<double>(),vp=v.data_ptr<double>();
 auto gp=g.data_ptr<double>(),bp=beta.data_ptr<double>(),sp=state.data_ptr<double>(),yp=out.data_ptr<double>();
 for(int t=0;t<T;t++)for(int h=0;h<H;h++){
  int hq=h/(H/HQ);double decay=std::exp(gp[t*H+h]);
  for(int j=0;j<V;j++){
   double memory=0;
   for(int i=0;i<K;i++){auto index=(h*K+i)*V+j;sp[index]*=decay;memory+=sp[index]*kp[(t*HQ+hq)*K+i];}
   double delta=bp[t*H+h]*(vp[(t*H+h)*V+j]-memory),y=0;
   for(int i=0;i<K;i++){auto index=(h*K+i)*V+j;sp[index]+=kp[(t*HQ+hq)*K+i]*delta;y+=sp[index]*qp[(t*HQ+hq)*K+i];}
   yp[(t*H+h)*V+j]=y/std::sqrt(double(K));
  }
 }
 return {out.to(at::kBFloat16).to(at::kFloat),state.to(at::kFloat)};
}
void check(at::Tensor out,at::Tensor state,const std::pair<at::Tensor,at::Tensor>& ref,int T,int C){
 TORCH_CHECK(at::isfinite(out).all().item<bool>()&&at::isfinite(state).all().item<bool>(),"Tensor GDN nonfinite result");
 TORCH_CHECK(at::allclose(out.to(at::kCPU).to(at::kFloat),ref.first,.01,.001),
  "Tensor GDN FP64 output mismatch T=",T," chunk=",C);
 TORCH_CHECK(at::allclose(state.to(at::kCPU),ref.second,.0001,.00001),
  "Tensor GDN FP64 state mismatch T=",T," chunk=",C,
  " max_abs=",(state.to(at::kCPU)-ref.second).abs().max().item<float>());
}
}
void run_tensor_gdn_tests(){
 auto f=at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);at::manual_seed(314);
 for(int T:{1,31,32,33,63,64,65,129,513})for(int C:{32,64}){
  int HQ=1,H=3,V=128;
  // Strided head/feature views cover the exact layouts coming from conv splits.
  auto source=at::randn({T,HQ,384},f).to(at::kBFloat16);
  auto q=avi::fused_l2(source.narrow(2,0,128)),k=avi::fused_l2(source.narrow(2,128,128));
  auto qp=at::empty({T,HQ,144},q.options()),kp=at::empty_like(qp);
  qp.narrow(2,0,128).copy_(q);kp.narrow(2,0,128).copy_(k);
  q=qp.narrow(2,0,128);k=kp.narrow(2,0,128);
  auto v=at::randn({T,H,V+16},f).to(at::kBFloat16).narrow(2,0,V);
  auto gates=at::rand({T,H,2},f);
  auto g=-gates.select(2,0)*.08,beta=gates.select(2,1);
  auto initial=at::randn({H,128,V},f)*.1,state=initial.clone();
  auto ref=oracle(q,k,v,g,beta,initial);
  auto actual=avi::delta_scan_tensor(q,k,v,g,beta,state,C);check(actual,state,ref,T,C);
  if(T>=65){
   auto split=initial.clone();
   auto a=avi::delta_scan_tensor(q.narrow(0,0,37),k.narrow(0,0,37),v.narrow(0,0,37),
    g.narrow(0,0,37),beta.narrow(0,0,37),split,C);
   auto z=avi::delta_scan_tensor(q.narrow(0,37,T-37),k.narrow(0,37,T-37),v.narrow(0,37,T-37),
    g.narrow(0,37,T-37),beta.narrow(0,37,T-37),split,C);
   check(at::cat({a,z}),split,ref,T,C);
  }
 }
 // Actual TP2 grouped-head geometry and zero/strong decay (underflow tails).
 for(float decay:{0.f,.001f,30.f}){
  int T=129,HQ=8,H=24;
  auto q=avi::fused_l2(at::randn({T,HQ,128},f).to(at::kBFloat16));
  auto k=avi::fused_l2(at::randn_like(q)),v=at::randn({T,H,128},f).to(at::kBFloat16);
  auto g=at::full({T,H},-decay,f),beta=at::rand({T,H},f),initial=at::randn({H,128,128},f)*.1;
  auto ref=oracle(q,k,v,g,beta,initial);
  for(int C:{32,64}){auto state=initial.clone();check(avi::delta_scan_tensor(q,k,v,g,beta,state,C),state,ref,T,C);}
 }
 // Capture/replay must use live input buffers and restore the caller's state.
 {
  int T=65,H=3;
  auto q=avi::fused_l2(at::randn({T,1,128},f).to(at::kBFloat16)),k=avi::fused_l2(at::randn_like(q));
  auto v=at::randn({T,H,128},f).to(at::kBFloat16),g=-at::rand({T,H},f)*.05,beta=at::rand({T,H},f);
  auto initial=at::randn({H,128,128},f)*.1,state=initial.clone();
  at::cuda::CUDAGraph graph;at::Tensor out;
  auto stream=c10::cuda::getStreamFromPool(false);
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  {
   c10::cuda::CUDAStreamGuard guard(stream);
   out=avi::delta_scan_tensor(q,k,v,g,beta,state,64);
   state.copy_(initial);C10_CUDA_CHECK(cudaStreamSynchronize(stream));
   graph.capture_begin();out=avi::delta_scan_tensor(q,k,v,g,beta,state,64);graph.capture_end();
  }
  for(int i=0;i<2;i++){
   if(i)v.add_(.125);
   state.copy_(initial);graph.replay();check(out,state,oracle(q,k,v,g,beta,initial),T,64);
  }
 }
 std::cout<<"Tensor GDN FP64 oracle, TP2 grouped heads, partial chunks, continuation and graph replay passed\n";
}
