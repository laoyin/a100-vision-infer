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
 // Original block-FP8 codes: independent per-column-block scale reference.
 for(int M:{1,4,17}) {
  int N=137,K=257;auto x=at::randn({M,K},opt).to(at::kBFloat16);
  auto codes=at::randint(0,126,{N,K},opt.dtype(at::kByte));auto scales=at::rand({N,3},opt)*0.01+0.001;
  auto unit=avi::fp8_decode(codes,at::ones({N},opt)).to(at::kFloat);
  auto expanded=scales.repeat_interleave(128,1).narrow(1,0,K);
  auto decoded=(unit*expanded).to(at::kBFloat16);
  TORCH_CHECK(at::equal(avi::fp8_decode(codes,scales),decoded),"Block FP8 dequant mismatch");
  auto expected=at::matmul(x,decoded.t()).to(at::kFloat);
  TORCH_CHECK(at::allclose(avi::fp8_linear(x,codes,scales).to(at::kFloat),expected,0.025,0.05),"Block FP8 GEMV/WMMA mismatch");
 }
 // GDN mixed precision fusion: preserve qkv/z/b/a order across two GEMMs.
 for(int M:{1,17}) {
  auto input=at::randn({M,256},opt).to(at::kBFloat16);
  auto qkv=at::randint(0,126,{384,256},opt.dtype(at::kByte)),z=at::randint(0,126,{128,256},opt.dtype(at::kByte));
  auto qs=at::rand({384,2},opt)*.01,zs=at::rand({128,2},opt)*.01;
  auto b=at::randn({4,256},opt).to(at::kBFloat16),a=at::randn_like(b);
  auto separate=at::cat({avi::fp8_linear(input,qkv,qs),avi::fp8_linear(input,z,zs),at::matmul(input,b.t()),at::matmul(input,a.t())},1);
  auto grouped=at::cat({avi::fp8_linear(input,at::cat({qkv,z},0),at::cat({qs,zs},0)),at::matmul(input,at::cat({b,a},0).t())},1);
  TORCH_CHECK(at::allclose(separate.to(at::kFloat),grouped.to(at::kFloat),.025,.05),"Mixed projection fusion order/numerics mismatch");
 }
 auto x=at::randn({3,512},opt).to(at::kBFloat16),w=at::randn({512},opt).to(at::kBFloat16);
 auto f=x.to(at::kFloat);auto expected=(f*at::rsqrt((f*f).mean(-1,true)+1e-6)*(1+w.to(at::kFloat))).to(at::kBFloat16);
 TORCH_CHECK(at::allclose(avi::fused_rms(x,w,1e-6,true).to(at::kFloat),expected.to(at::kFloat),0.02,0.02),"Fused RMS mismatch");
 for(int D:{16,128,257})for(int T:{1,17}){
  auto input=at::randn({T,2,D},opt).to(at::kBFloat16),gate=(at::randn({T,2,D},opt)*8).to(at::kBFloat16);
  auto weight=at::randn({D},opt).to(at::kBFloat16);
  auto ref=(avi::fused_rms(input,weight,1e-6,false).to(at::kFloat)*at::silu(gate.to(at::kFloat))).to(at::kBFloat16);
  TORCH_CHECK(at::allclose(avi::fused_rms_gate(input,weight,gate,1e-6).to(at::kFloat),ref.to(at::kFloat),.01,.01),"Fused RMS gate mismatch");
  TORCH_CHECK(at::allclose(avi::fused_sigmoid_gate(input,gate).to(at::kFloat),(input*gate.sigmoid()).to(at::kFloat),.01,.001),"Fused sigmoid gate mismatch");
 }
 {
  auto a=(at::randn({17,24},opt)*30).to(at::kBFloat16),b=at::randn_like(a);
  auto decay=at::randn({24},opt),bias=at::randn({24},opt);
  auto gates=avi::fused_gdn_gates(a,b,decay,bias);
  TORCH_CHECK(at::allclose(gates.first,-decay.exp()*at::softplus(a.to(at::kFloat)+bias),1e-5,1e-6),"GDN decay gate mismatch");
  TORCH_CHECK(at::allclose(gates.second,b.sigmoid().to(at::kFloat),.005,1e-5),"GDN beta gate mismatch");
 }
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