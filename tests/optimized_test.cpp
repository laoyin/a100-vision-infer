#include "avi/engine.h"
#include "avi/ops.h"
#include <iostream>
void run_optimized_tests(){
 auto opt=at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);at::manual_seed(17);
 // Shared weights across query rows, odd tails and original FP8 block scales.
 for(int M:{1,2,4,6,8})for(int K:{17,128,257,5120}){
  int N=19;auto x=at::randn({M,K},opt).to(at::kBFloat16);
  auto codes=at::randint(0,126,{N,K},opt.dtype(at::kByte));auto scales=at::rand({N,(K+127)/128},opt)*.01+.001;
  auto decoded=avi::fp8_decode(codes,scales);
  auto actual=avi::small_linear_shared(x,codes,scales),bf16=avi::small_linear_shared(x,decoded);
  TORCH_CHECK(at::equal(actual,bf16),"Shared GEMV FP8/BF16 rounding mismatch");
  TORCH_CHECK(at::allclose(actual.to(at::kFloat),at::matmul(x,decoded.t()).to(at::kFloat),.025,.05),"Shared GEMV reference mismatch");
 }
 // Packed loads must fall back for otherwise contiguous unaligned views.
 {
  int M=2,N=19,K=132;
  auto x=at::randn({M*K+1},opt).to(at::kBFloat16).narrow(0,1,M*K).reshape({M,K});
  auto codes=at::randint(0,126,{N*K+1},opt.dtype(at::kByte)).narrow(0,1,N*K).reshape({N,K});
  auto scales=at::ones({N,(K+127)/128},opt)*.01;
  TORCH_CHECK(at::allclose(avi::small_linear_shared(x,codes,scales).to(at::kFloat),
    at::matmul(x,avi::fp8_decode(codes,scales).t()).to(at::kFloat),.025,.05),"Unaligned shared GEMV fallback mismatch");
 }
 // Nonzero convolution history, repeated key heads and BF16 rounding boundaries.
 for(int T:{32,33,129})for(int K:{16,128})for(int V:{17,128}){
  int HK=2,H=6,C=2*HK*K+H*V,width=4;
  auto projected=at::randn({T,C+H*V+2*H},opt).to(at::kBFloat16);
  auto weight=at::randn({C,1,width},opt).to(at::kBFloat16);
  auto history=at::randn({1,C,width-1},opt).to(at::kBFloat16),original=history.clone();
  auto log_decay=at::randn({H},opt).to(at::kBFloat16),bias=at::randn({H},opt).to(at::kBFloat16);
  auto mixed=projected.narrow(1,0,C).t().unsqueeze(0);
  auto input=at::cat({original,mixed},-1),conv=avi::conv_prefill(input,weight);
  auto q=avi::fused_l2(conv.narrow(1,0,HK*K).reshape({T,HK,K})).repeat_interleave(H/HK,1);
  auto k=avi::fused_l2(conv.narrow(1,HK*K,HK*K).reshape({T,HK,K})).repeat_interleave(H/HK,1);
  auto v=conv.narrow(1,2*HK*K,H*V).reshape({T,H,V});
  auto gates=avi::fused_gdn_gates(projected.narrow(1,C+H*V+H,H),projected.narrow(1,C+H*V,H),log_decay,bias);
  auto actual=avi::fused_gdn_prepare(projected,weight,history,log_decay,bias,HK,H,K,V);
  TORCH_CHECK(at::allclose(actual[0].to(at::kFloat),q.to(at::kFloat),.02,.003)&&
    at::allclose(actual[1].to(at::kFloat),k.to(at::kFloat),.02,.003)&&
    at::allclose(actual[2].to(at::kFloat),v.to(at::kFloat),.02,.003),"Fused GDN convolution/L2 mismatch");
  TORCH_CHECK(at::allclose(actual[3],gates.first,.0001,.00001)&&at::equal(actual[4],gates.second),"Fused GDN gates mismatch");
  TORCH_CHECK(at::equal(history,input.narrow(-1,T,width-1)),"Fused GDN history update mismatch");
 }
 // Packed GEMV: signed/subnormal FP8, block boundaries, batch and scalar fallback.
 for(int M:{1,2,8})for(int K:{4,128,132,257,5120}){
  int N=19;auto input=at::randn({M,K},opt).to(at::kBFloat16);
  auto codes=(at::randint(0,127,{N,K},opt.dtype(at::kLong))+128*at::randint(0,2,{N,K},opt.dtype(at::kLong))).to(at::kByte);
  auto scales=at::rand({N,(K+127)/128},opt)*.01+.001;
  auto reference=at::matmul(input,avi::fp8_decode(codes,scales).t()).to(at::kFloat);
  TORCH_CHECK(at::allclose(avi::fp8_linear(input,codes,scales,true).to(at::kFloat),reference,.025,.05),"Vector FP8 GEMV mismatch");
 }
 // Contiguous views can still be unaligned: they must use the scalar fallback.
 {
  auto input=at::randn({133},opt).to(at::kBFloat16).narrow(0,1,132).reshape({1,132});
  auto codes=at::randint(0,126,{19,132},opt.dtype(at::kByte));auto scales=at::ones({19},opt)*.01;
  TORCH_CHECK(at::equal(avi::fp8_linear(input,codes,scales,true),avi::fp8_linear(input,codes,scales)),"Unaligned GEMV fallback mismatch");
 }
 // AllGather layout must concatenate vocabulary, never request rows.
 for(int B:{1,2,3})for(int TP:{2,4}){
  auto input=at::randn({B,32},opt).to(at::kBFloat16),weight=at::randn({128,32},opt).to(at::kBFloat16);
  std::vector<at::Tensor> shards;
  for(int rank=0;rank<TP;rank++)shards.push_back(at::matmul(input,weight.narrow(0,rank*(128/TP),128/TP).t()).to(at::kFloat));
  auto gathered=at::stack(shards,0).permute({1,0,2}).reshape({B,128});
  TORCH_CHECK(at::allclose(gathered,at::matmul(input,weight.t()).to(at::kFloat),.02,.02),"TP output head gather layout mismatch");
 }
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
 // Every saved recurrent state must match executing exactly that accepted prefix.
 for(int T:{1,31,32,33,67})for(int K:{16,128}){
  auto q=avi::fused_l2(at::randn({T,2,K},opt).to(at::kBFloat16));
  auto k=avi::fused_l2(at::randn_like(q)),v=at::randn({T,2,32},opt).to(at::kBFloat16);
  auto g=-at::rand({T,2},opt)*.1,b=at::rand({T,2},opt);
  auto reference=at::randn({2,K,32},opt),state=reference.clone();
  auto expected=avi::delta_scan_fast(q,k,v,g,b,reference);
  auto actual=avi::delta_scan_chunked(q,k,v,g,b,state);
  TORCH_CHECK(at::allclose(actual.to(at::kFloat),expected.to(at::kFloat),.02,.003),"Chunked GDN output mismatch");
  TORCH_CHECK(at::allclose(state,reference,.001,.0001),"Chunked GDN state mismatch");
 }
 for(int K:{16,128}) {
  auto q=avi::fused_l2(at::randn({6,2,K},opt).to(at::kBFloat16));
  auto k=avi::fused_l2(at::randn_like(q)),v=at::randn({6,2,32},opt).to(at::kBFloat16);
  auto g=-at::rand({6,2},opt),b=at::rand({6,2},opt),initial=at::randn({2,K,32},opt);
  auto state=initial.clone(),history=at::empty({6,2,K,32},opt);
  auto all=avi::delta_scan_fast(q,k,v,g,b,state,history);
  auto sequential=initial.clone();
  for(int i=0;i<6;++i){
   auto row=avi::delta_scan_fast(q.narrow(0,i,1),k.narrow(0,i,1),v.narrow(0,i,1),g.narrow(0,i,1),b.narrow(0,i,1),sequential);
   TORCH_CHECK(at::allclose(row.to(at::kFloat),all.narrow(0,i,1).to(at::kFloat),1e-5,1e-5),"GDN verification output");
   TORCH_CHECK(at::allclose(sequential,history.select(0,i),1e-5,1e-5),"GDN accepted-prefix state");
   // Resume after rollback; discarded future state must have no effect.
   if(i<5){
    auto restored=history.select(0,i).clone();
    avi::delta_scan_fast(q.narrow(0,i+1,5-i),k.narrow(0,i+1,5-i),v.narrow(0,i+1,5-i),g.narrow(0,i+1,5-i),b.narrow(0,i+1,5-i),restored);
    TORCH_CHECK(at::allclose(restored,state,1e-5,1e-5),"GDN rollback/resume");
   }
  }
 }
 for(int T:{1,2,4,6})for(int old:{0,255,257}){
  int H=4,HK=2,D=128;
  auto q=at::randn({T,H,D},opt).to(at::kBFloat16),k=at::randn({T,HK,D},opt).to(at::kBFloat16),v=at::randn_like(k);
  auto keys=at::randn({512,HK,D},opt).to(at::kBFloat16),vals=at::randn_like(keys);
  auto refk=keys.clone(),refv=vals.clone();auto actual=avi::gqa_chunk(q,k,v,keys,vals,old);
  for(int i=0;i<T;++i){
   auto expected=avi::gqa_decode(q.narrow(0,i,1),k.narrow(0,i,1),v.narrow(0,i,1),refk,refv,at::full({1},old+i,opt.dtype(at::kLong)));
   TORCH_CHECK(at::allclose(actual.narrow(0,i,1).to(at::kFloat),expected.to(at::kFloat),.01,.01),"Causal multi-query GQA mismatch");
  }
  TORCH_CHECK(at::equal(keys.narrow(0,0,old+T),refk.narrow(0,0,old+T)),"Chunk KV append mismatch");
 }
 // WY batching must preserve recurrence across full/partial chunks and calls.
 for(int K:{16,128})for(int T:{1,31,32,33,129})for(float decay:{.1f,20.f}){
  int H=2,V=17;
  auto q=avi::fused_l2(at::randn({T,H,K},opt).to(at::kBFloat16));
  auto k=avi::fused_l2(at::randn_like(q)),v=at::randn({T,H,V},opt).to(at::kBFloat16);
  auto g=-at::rand({T,H},opt)*decay,b=at::rand({T,H},opt);
  auto state=at::randn({H,K,V},opt),reference=state.clone(),fused_state=state.clone();
  auto expected=avi::delta_scan_fast(q,k,v,g,b,reference);
  auto actual=avi::delta_scan_wy(q,k,v,g,b,state);
  auto fused=avi::delta_scan_wy(q,k,v,g,b,fused_state,32,true);
  TORCH_CHECK(at::allclose(fused.to(at::kFloat),expected.to(at::kFloat),.02,.003)&&
    at::allclose(fused_state,reference,.001,.0001),"CUDA WY output/state mismatch");
  TORCH_CHECK(at::allclose(actual.to(at::kFloat),expected.to(at::kFloat),.02,.003),"Batched WY output mismatch");
  TORCH_CHECK(at::allclose(state,reference,.001,.0001),"Batched WY final state mismatch");
 }
 for(int T:{1,2,4,6})for(int old:{0,3,255,257}){
  int H=6,HK=2,D=128;
  auto q=at::randn({T,H,D},opt).to(at::kBFloat16),k=at::randn({T,HK,D},opt).to(at::kBFloat16),v=at::randn_like(k);
  auto keys=at::randn({521,HK,D},opt).to(at::kBFloat16),values=at::randn_like(keys);
  auto refk=keys.clone(),refv=values.clone();
  auto offset=at::full({1},old,opt.dtype(at::kLong));
  auto actual=avi::gqa_chunk_dynamic(q,k,v,keys,values,offset);
  auto expected=avi::gqa_chunk(q,k,v,refk,refv,old);
  TORCH_CHECK(at::allclose(actual.to(at::kFloat),expected.to(at::kFloat),.01,.01),"Dynamic prefix GQA output mismatch");
  TORCH_CHECK(at::equal(keys,refk)&&at::equal(values,refv),"Dynamic prefix GQA writes outside appended range");
 }
 // Nonzero recurrent states, long prefixes, partial value tiles and rollback.
 for(int K:{16,128})for(int T:{1,4,33,513})for(int V:{17,32,128}){
  auto q=avi::fused_l2(at::randn({T,2,K},opt).to(at::kBFloat16));
  auto k=avi::fused_l2(at::randn_like(q)),v=at::randn({T,2,V},opt).to(at::kBFloat16);
  auto g=-at::rand({T,2},opt)*.1,b=at::rand({T,2},opt);
  auto state=at::randn({2,K,V},opt),ref=state.clone();
  auto trajectory=at::empty({T,2,K,V},opt),reference_history=at::empty_like(trajectory);
  auto expected=avi::delta_scan_fast(q,k,v,g,b,ref,reference_history);
  auto actual=avi::delta_scan_fast(q,k,v,g,b,state,trajectory,true);
  TORCH_CHECK(at::allclose(actual.to(at::kFloat),expected.to(at::kFloat),.02,.003),"Cooperative GDN output mismatch");
  TORCH_CHECK(at::allclose(trajectory,reference_history,.001,.0001)&&at::allclose(state,ref,.001,.0001),"Cooperative GDN state/trajectory mismatch");
 }
 for(int C:{7,64})for(int T:{1,4,33,513}){
  auto input=at::randn({1,C,T+3},opt).to(at::kBFloat16),weight=at::randn({C,1,4},opt).to(at::kBFloat16);
  auto expected=at::silu(at::conv1d(input,weight,{},at::IntArrayRef{1},at::IntArrayRef{0},at::IntArrayRef{1},C)).squeeze(0).t();
  TORCH_CHECK(at::allclose(avi::conv_prefill(input,weight).to(at::kFloat),expected.to(at::kFloat),.02,.01),"Fused GDN convolution mismatch");
 }
 // Exercise bottom-right causal alignment, including unequal lengths and GQA.
 for(int D:{16,128,256})for(int T:{1,9,33})for(int old:{0,17,257}){
  const int H=6,HK=2,L=old+T;
  auto q=at::randn({T,H,D},opt).to(at::kBFloat16);
  auto k=at::randn({L,HK,D},opt).to(at::kBFloat16),v=at::randn_like(k);
  auto actual=avi::flash_prefill(q,k,v);
  auto mask=at::arange(L,opt.dtype(at::kLong)).unsqueeze(0)<=at::arange(old,L,opt.dtype(at::kLong)).unsqueeze(1);
  auto expected=at::scaled_dot_product_attention(q.transpose(0,1).unsqueeze(0),
      k.repeat_interleave(H/HK,1).transpose(0,1).unsqueeze(0),
      v.repeat_interleave(H/HK,1).transpose(0,1).unsqueeze(0),mask,0.0,false).squeeze(0).transpose(0,1);
  TORCH_CHECK(at::allclose(actual.to(at::kFloat),expected.to(at::kFloat),.03,.01),
              "Flash prefill causal/GQA mismatch: D=",D," T=",T," prefix=",old);
 }
 std::cout<<"Optimized kernels, speculative GDN rollback, Flash prefill and multi-query GQA passed\n";
}
