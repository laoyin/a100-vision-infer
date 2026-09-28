#include "avi/engine.h"
#include "avi/ops.h"
#include <iostream>
void run_optimized_tests(){
 auto opt=at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);at::manual_seed(17);
 for(int M:{1,4,17,33})for(int K:{32,70}){
  int N=79;auto x=at::randn({M,K},opt).to(at::kBFloat16);
  auto codes=at::randint(0,126,{N,K},opt.dtype(at::kByte));auto scales=at::rand({N},opt)*0.01;
  auto reference=at::matmul(x,avi::fp8_decode(codes,scales).t()).to(at::kFloat);
  auto actual=avi::fp8_linear(x,codes,scales).to(at::kFloat);
  TORCH_CHECK(at::allclose(reference,actual,0.025,0.05),"FP8 GEMV/WMMA mismatch: M=",M," K=",K);
 }
 auto x=at::randn({3,512},opt).to(at::kBFloat16),w=at::randn({512},opt).to(at::kBFloat16);
 auto f=x.to(at::kFloat);auto expected=(f*at::rsqrt((f*f).mean(-1,true)+1e-6)*(1+w.to(at::kFloat))).to(at::kBFloat16);
 TORCH_CHECK(at::allclose(avi::fused_rms(x,w,1e-6,true).to(at::kFloat),expected.to(at::kFloat),0.02,0.02),"Fused RMS mismatch");
 auto gu=at::randn({4,128},opt).to(at::kBFloat16);auto sg=at::silu(gu.narrow(1,0,64))*gu.narrow(1,64,64);
 TORCH_CHECK(at::allclose(avi::fused_swiglu(gu).to(at::kFloat),sg.to(at::kFloat),0.02,0.02),"SwiGLU mismatch");
 for(int K:{16,128}){
  auto q=at::randn({5,2,K},opt).to(at::kBFloat16),k=at::randn_like(q),v=at::randn({5,2,32},opt).to(at::kBFloat16);
  q=avi::fused_l2(q);k=avi::fused_l2(k);auto g=-at::rand({5,2},opt),b=at::rand({5,2},opt);
  auto s1=at::zeros({2,K,32},opt),s2=at::zeros_like(s1);
  auto a=avi::delta_scan(q,k,v,g,b,s1),c=avi::delta_scan_fast(q,k,v,g,b,s2);
  TORCH_CHECK(at::allclose(a.to(at::kFloat),c.to(at::kFloat),0.02,0.02)&&at::allclose(s1,s2,1e-4,1e-4),"Register scan mismatch");
 }
 for(int D:{16,256})for(int L:{1,255,257}){
  int H=4,HK=2,capacity=512;auto q=at::randn({1,H,D},opt).to(at::kBFloat16);
  auto k=at::randn({1,HK,D},opt).to(at::kBFloat16),v=at::randn_like(k);
  auto keys=at::randn({capacity,HK,D},opt).to(at::kBFloat16),values=at::randn_like(keys);
  auto offset=at::full({1},L-1,opt.dtype(at::kLong));auto actual=avi::gqa_decode(q,k,v,keys,values,offset);
  auto kr=keys.narrow(0,0,L).repeat_interleave(H/HK,1).transpose(0,1).unsqueeze(0);
  auto vr=values.narrow(0,0,L).repeat_interleave(H/HK,1).transpose(0,1).unsqueeze(0);
  auto expected=at::scaled_dot_product_attention(q.transpose(0,1).unsqueeze(0),kr,vr,{},0.0,false).squeeze(0).transpose(0,1);
  TORCH_CHECK(at::allclose(actual.to(at::kFloat),expected.to(at::kFloat),0.03,0.03),"Split-K GQA mismatch at length ",L);
 }
 auto hist=at::randn({1,8,3},opt).to(at::kBFloat16),cx=at::randn({1,8},opt).to(at::kBFloat16),cw=at::randn({8,1,4},opt).to(at::kBFloat16);
 auto all=at::cat({hist,cx.unsqueeze(-1)},-1);auto conv=at::silu(at::conv1d(all,cw,{},at::IntArrayRef{1},at::IntArrayRef{0},at::IntArrayRef{1},8)).reshape({1,8});
 auto co=avi::conv_decode(cx,cw,hist);TORCH_CHECK(at::allclose(co.to(at::kFloat),conv.to(at::kFloat),0.03,0.03)&&at::equal(hist,all.narrow(-1,1,3)),"Conv state mismatch");
 // Zero position must be identity, including the unrotated tail.
 auto rx=at::randn({3,2,64},opt).to(at::kBFloat16);
 auto rp=at::zeros({3,3},opt.dtype(at::kLong));
 TORCH_CHECK(at::equal(avi::fused_rope(rx,rp,32,10000.,4,4),rx),"Zero-position RoPE mismatch");
 std::cout<<"Optimized GEMV/WMMA, RMS, SwiGLU, register GDN, GQA and convolution passed\n";
}