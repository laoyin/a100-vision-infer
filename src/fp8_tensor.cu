// Ampere W8A16: original E4M3FN bytes and row-expanded block128 multipliers.
// Decode each tile directly into BF16 shared memory; no full BF16 matrix.
#include "avi/ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
namespace avi {
namespace {
using bf=__nv_bfloat16;namespace wm=nvcuda::wmma;
__device__ float value(unsigned char b){
 int e=(b>>3)&15,m=b&7;float x=e?ldexpf(1.f+m*.125f,e-7):ldexpf(float(m),-9);
 if(e==15&&m==7)x=nanf("");return (b&128)?-x:x;
}
template<int SPLIT> __global__ void w8a16(const bf* x,const unsigned char* w,const float* scales,
 bf* output,float* partial,int M,int N,int K){
 __shared__ __align__(32) bf a[16*128],b[64*128];
 __shared__ __align__(32) float tile[4][256];
 int warp=threadIdx.x/32,lane=threadIdx.x%32,n0=blockIdx.x*64,split=blockIdx.y;
 int total=K/128,steps=(total+SPLIT-1)/SPLIT,begin=split*steps,end=min(total,begin+steps);
 wm::fragment<wm::accumulator,16,16,16,float> acc;wm::fill_fragment(acc,0.f);
 for(int block=begin;block<end;block++){
  for(int i=threadIdx.x;i<16*128;i+=128)a[i]=i/128<M?x[(i/128)*K+block*128+i%128]:__float2bfloat16_rn(0.f);
  for(int i=threadIdx.x;i<64*128;i+=128){
   int n=n0+i/128;b[i]=n<N?__float2bfloat16_rn(value(w[int64_t(n)*K+block*128+i%128])*scales[n*total+block]):__float2bfloat16_rn(0.f);
  }
  __syncthreads();
  for(int kk=0;kk<128;kk+=16){
   wm::fragment<wm::matrix_a,16,16,16,bf,wm::row_major> af;
   wm::fragment<wm::matrix_b,16,16,16,bf,wm::col_major> bf_;
   wm::load_matrix_sync(af,a+kk,128);wm::load_matrix_sync(bf_,b+warp*16*128+kk,128);
   wm::mma_sync(acc,af,bf_,acc);
  }
  __syncthreads();
 }
 wm::store_matrix_sync(tile[warp],acc,16,wm::mem_row_major);__syncwarp();
 for(int i=lane;i<256;i+=32){
  int m=i/16,n=n0+warp*16+i%16;
  if(m<M&&n<N){
   if(SPLIT==1)output[m*N+n]=__float2bfloat16_rn(tile[warp][i]);
   else partial[(split*M+m)*N+n]=tile[warp][i];
  }
 }
}
__global__ void reduce4(const float* p,bf* output,int elements){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<elements;i+=blockDim.x*gridDim.x)
  output[i]=__float2bfloat16_rn(((p[i]+p[elements+i])+p[2*elements+i])+p[3*elements+i]);
}
}
at::Tensor fp8_tensor_small(at::Tensor x,at::Tensor codes,at::Tensor scales,int split){
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&x.dim()==2&&x.is_contiguous()&&
 codes.device()==x.device()&&codes.scalar_type()==at::kByte&&codes.dim()==2&&codes.is_contiguous()&&
 scales.device()==x.device()&&scales.scalar_type()==at::kFloat&&scales.dim()==2&&scales.is_contiguous()&&
 x.size(0)>=2&&x.size(0)<=8&&x.size(1)>0&&x.size(1)==codes.size(1)&&x.size(1)%128==0&&
 codes.size(0)>0&&scales.size(0)==codes.size(0)&&scales.size(1)==x.size(1)/128&&
 (split==1||split==4),"Invalid Ampere FP8 W8A16 geometry/dtype");
 c10::cuda::CUDAGuard guard(x.device());int M=x.size(0),N=codes.size(0),K=x.size(1);
 auto out=at::empty({M,N},x.options());auto stream=at::cuda::getCurrentCUDAStream();
 auto xp=reinterpret_cast<const bf*>(x.data_ptr<at::BFloat16>());auto yp=reinterpret_cast<bf*>(out.data_ptr<at::BFloat16>());
 if(split==1)w8a16<1><<<dim3((N+63)/64,1),128,0,stream>>>(xp,codes.data_ptr<unsigned char>(),scales.data_ptr<float>(),yp,nullptr,M,N,K);
 else{
  auto partial=at::empty({4,M,N},x.options().dtype(at::kFloat));
  w8a16<4><<<dim3((N+63)/64,4),128,0,stream>>>(xp,codes.data_ptr<unsigned char>(),scales.data_ptr<float>(),yp,partial.data_ptr<float>(),M,N,K);
  reduce4<<<std::min(65535,(M*N+255)/256),256,0,stream>>>(partial.data_ptr<float>(),yp,M*N);
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
}
