#include "avi/ops.h"
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <iostream>
void run_dual_path_tests(){
 auto f=at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);at::manual_seed(104);
 for(int M:{1,4,33,512})for(int D:{16,128,5120})for(bool centered:{false,true}){
  auto x=at::randn({M,D},f).to(at::kBFloat16),update=at::randn({M,D},f).to(at::kBFloat16),w=at::randn({D},f).to(at::kBFloat16);
  auto output=avi::residual_rms(x,update,w,1e-6,centered);auto sum=x+update;
  TORCH_CHECK(at::equal(output.first,sum),"Residual BF16 rounding changed");
  auto ref=sum.to(at::kFloat);ref=ref*at::rsqrt((ref*ref).mean(-1,true)+1e-6);
  if(!centered)ref=ref.to(at::kBFloat16).to(at::kFloat);
  ref=(ref*(w.to(at::kFloat)+(centered?1.f:0.f))).to(at::kBFloat16);
  TORCH_CHECK(at::allclose(output.second.to(at::kFloat),ref.to(at::kFloat),.01,.001),"Fused RMS reference mismatch");
  TORCH_CHECK(at::equal(output.second,avi::fused_rms(sum,w,1e-6,centered)),"Fused RMS changed existing reduction/rounding");
 }
 for(int M:{1,4,6})for(int N:{1,257,4096,4097,124160}){
  auto logits=at::randn({M,N},f).to(at::kBFloat16);
  logits.select(1,0).fill_(8);logits.select(1,N-1).fill_(8); // ties across chunks
  auto candidates=avi::vocabulary_candidates(logits,124160);auto top=at::max(logits.to(at::kFloat),-1);
  TORCH_CHECK(at::equal(candidates.select(1,0),std::get<0>(top).to(at::kDouble)),"Candidate value mismatch");
  TORCH_CHECK(at::equal(candidates.select(1,1), (std::get<1>(top)+124160).to(at::kDouble)),"Candidate tie/offset mismatch");
  auto lower=candidates.clone();lower.select(1,1).sub_(124160);
  auto ids=avi::merge_candidates(at::stack({candidates,lower}));
  TORCH_CHECK(at::equal(ids,std::get<1>(top)),"TP candidate tie mismatch");
 }
 for(double invalid:{NAN,INFINITY,-INFINITY}){
  auto logits=at::full({1,4097},invalid,f).to(at::kBFloat16);
  auto ids=avi::merge_candidates(avi::vocabulary_candidates(logits).unsqueeze(0));
  TORCH_CHECK(ids.item<int64_t>()==-1,"Nonfinite vocabulary not rejected");
 }
 // A losing -inf is valid; a NaN anywhere must invalidate the row.
 auto logits=at::zeros({2,4097},f).to(at::kBFloat16);
 logits[0][4096].fill_(-INFINITY);logits[1][4096].fill_(NAN);
 auto ids=avi::merge_candidates(avi::vocabulary_candidates(logits).unsqueeze(0)).cpu();
 TORCH_CHECK(ids[0].item<int64_t>()==0&&ids[1].item<int64_t>()==-1,"Nonfinite propagation mismatch");
 auto x=at::randn({4,128},f).to(at::kBFloat16),u=at::randn_like(x),w=at::zeros({128},x.options());
 auto stream=c10::cuda::getStreamFromPool(false);at::cuda::CUDAGraph graph;at::Tensor output;
 auto invoke=[&](){return avi::merge_candidates(avi::vocabulary_candidates(avi::residual_rms(x,u,w,1e-6).second).unsqueeze(0));};
 C10_CUDA_CHECK(cudaDeviceSynchronize());
 {c10::cuda::CUDAStreamGuard guard(stream);output=invoke();C10_CUDA_CHECK(cudaStreamSynchronize(stream));graph.capture_begin();output=invoke();graph.capture_end();}
 for(int i=0;i<2;i++){
  if(i)x.select(1,37).fill_(100);
  graph.replay();auto expected=avi::fused_rms(x+u,w,1e-6,true).argmax(-1);
  TORCH_CHECK(at::equal(output,expected),"Graph replay candidate data stale");
 }
 std::cout<<"Residual RMS, vocabulary ties/NaNs, TP merge and graph replay passed\n";
}
