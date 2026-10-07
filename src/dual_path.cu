#include "avi/ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <climits>
#include <cmath>
namespace avi {
namespace {
using bf=__nv_bfloat16;
__device__ float sum_warp(float x){for(int d=16;d;d/=2)x+=__shfl_down_sync(0xffffffff,x,d);return x;}
__global__ void add_norm(const bf* x,const bf* update,const bf* w,bf* residual,bf* out,int D,float eps,bool centered){
 __shared__ float sums[8];int row=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
 int64_t base=int64_t(row)*D;float sum=0;
 for(int i=threadIdx.x;i<D;i+=256){
  // Match the existing BF16 residual boundary before RMS statistics.
  bf rounded=__float2bfloat16_rn(__bfloat162float(x[base+i])+__bfloat162float(update[base+i]));
  residual[base+i]=rounded;float v=__bfloat162float(rounded);sum+=v*v;
 }
 sum=sum_warp(sum);if(!lane)sums[warp]=sum;__syncthreads();
 sum=threadIdx.x<8?sums[lane]:0.f;sum=sum_warp(sum);
 if(!threadIdx.x)sums[0]=rsqrtf(sum/D+eps);__syncthreads();
 for(int i=threadIdx.x;i<D;i+=256){
  float v=__bfloat162float(residual[base+i])*sums[0];
  if(!centered)v=__bfloat162float(__float2bfloat16_rn(v));
  out[base+i]=__float2bfloat16_rn(v*(__bfloat162float(w[i])+(centered?1.f:0.f)));
 }
}
__device__ void pick(float& value,int& id,float other,int other_id){
 if(other>value||(other==value&&other_id<id)){value=other;id=other_id;}
}
__device__ void warp_pick(float& value,int& id){
 for(int d=16;d;d/=2){float v=__shfl_down_sync(0xffffffff,value,d);int i=__shfl_down_sync(0xffffffff,id,d);pick(value,id,v,i);}
}
// Chunk vocabulary to expose enough CTAs for small M. Output is tiny, rather
// than a full FP32 copy of the vocabulary plus several conversion kernels.
__global__ void vocab_partial(const bf* logits,double* partial,int N,int chunks,int offset){
 __shared__ float values[8];__shared__ int ids[8];
 int row=blockIdx.y,chunk=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
 float best=-INFINITY;int id=INT_MAX;
 int64_t limit=int64_t(chunk+1)*4096;int end=limit<N?int(limit):N;
 for(int i=chunk*4096+threadIdx.x;i<end;i+=256){
  float v=__bfloat162float(logits[int64_t(row)*N+i]);
  // torch.max propagates NaN; preserve rejection using a nonfinite maximum.
  if(isnan(v))v=INFINITY;pick(best,id,v,i+offset);
 }
 warp_pick(best,id);if(!lane){values[warp]=best;ids[warp]=id;}__syncthreads();
 if(warp==0){best=lane<8?values[lane]:-INFINITY;id=lane<8?ids[lane]:INT_MAX;warp_pick(best,id);
  if(!lane){auto p=(int64_t(row)*chunks+chunk)*2;partial[p]=best;partial[p+1]=id;}}
}
__global__ void vocab_finish(const double* partial,double* out,int chunks){
 int row=blockIdx.x;float best=-INFINITY;int id=INT_MAX;
 for(int c=threadIdx.x;c<chunks;c+=32){auto p=(int64_t(row)*chunks+c)*2;pick(best,id,float(partial[p]),int(partial[p+1]));}
 warp_pick(best,id);if(!threadIdx.x){out[row*2]=best;out[row*2+1]=id;}
}
__global__ void candidate_merge(const double* gathered,int64_t* out,int ranks,int rows){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
 double best=-INFINITY;int64_t id=INT64_MAX;bool invalid=false;
 for(int rank=0;rank<ranks;rank++){
  auto p=(int64_t(rank)*rows+row)*2;double v=gathered[p],raw=gathered[p+1];
  // IDs are exactly representable in doubles; reject corrupt candidates.
  if(!isfinite(v)||!isfinite(raw)||raw<0||raw>INT_MAX||raw!=floor(raw)){invalid=true;continue;}
  int64_t next=int64_t(raw);if(v>best||(v==best&&next<id)){best=v;id=next;}
 }
 out[row]=invalid?-1:id;
}
}
std::pair<at::Tensor,at::Tensor> residual_rms(at::Tensor x,at::Tensor update,at::Tensor weight,double eps,bool centered){
 TORCH_CHECK(x.is_cuda()&&x.dim()>=2&&x.scalar_type()==at::kBFloat16&&x.is_contiguous()&&x.numel()>0&&
 update.device()==x.device()&&update.scalar_type()==x.scalar_type()&&update.sizes()==x.sizes()&&update.is_contiguous()&&
 weight.device()==x.device()&&weight.scalar_type()==x.scalar_type()&&weight.is_contiguous()&&weight.numel()==x.size(-1)&&
 x.size(-1)>0&&x.size(-1)<=INT_MAX&&x.numel()/x.size(-1)<=INT_MAX&&std::isfinite(eps)&&eps>0,"Invalid fused residual RMS input");
 c10::cuda::CUDAGuard guard(x.device());auto residual=at::empty_like(x),out=at::empty_like(x);
 add_norm<<<x.numel()/x.size(-1),256,0,at::cuda::getCurrentCUDAStream()>>>(
  reinterpret_cast<const bf*>(x.data_ptr<at::BFloat16>()),reinterpret_cast<const bf*>(update.data_ptr<at::BFloat16>()),
  reinterpret_cast<const bf*>(weight.data_ptr<at::BFloat16>()),reinterpret_cast<bf*>(residual.data_ptr<at::BFloat16>()),
  reinterpret_cast<bf*>(out.data_ptr<at::BFloat16>()),x.size(-1),eps,centered);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {residual,out};
}
at::Tensor vocabulary_candidates(at::Tensor logits,int64_t offset){
 TORCH_CHECK(logits.is_cuda()&&logits.scalar_type()==at::kBFloat16&&logits.dim()==2&&logits.is_contiguous()&&
 logits.size(0)>0&&logits.size(0)<=65535&&logits.size(1)>0&&offset>=0&&offset<=INT_MAX-logits.size(1),"Invalid vocabulary candidates");
 c10::cuda::CUDAGuard guard(logits.device());int M=logits.size(0),N=logits.size(1),chunks=(N+4095LL)/4096;
 auto out=at::empty({M,2},logits.options().dtype(at::kDouble)),partial=at::empty({M,chunks,2},out.options());
 auto stream=at::cuda::getCurrentCUDAStream();
 vocab_partial<<<dim3(chunks,M),256,0,stream>>>(reinterpret_cast<const bf*>(logits.data_ptr<at::BFloat16>()),partial.data_ptr<double>(),N,chunks,offset);
 vocab_finish<<<M,32,0,stream>>>(partial.data_ptr<double>(),out.data_ptr<double>(),chunks);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
at::Tensor merge_candidates(at::Tensor gathered){
 TORCH_CHECK(gathered.is_cuda()&&gathered.scalar_type()==at::kDouble&&gathered.is_contiguous()&&gathered.dim()==3&&
 gathered.size(0)>0&&gathered.size(0)<=8&&gathered.size(1)>0&&gathered.size(1)<=65535&&gathered.size(2)==2,"Invalid gathered candidates");
 c10::cuda::CUDAGuard guard(gathered.device());auto out=at::empty({gathered.size(1)},gathered.options().dtype(at::kLong));
 candidate_merge<<<(out.numel()+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(gathered.data_ptr<double>(),out.data_ptr<int64_t>(),gathered.size(0),gathered.size(1));
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
}
