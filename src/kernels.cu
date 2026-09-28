#include "avi/engine.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
namespace avi {
// E4M3FN: exp=15 is finite except mantissa=7; max magnitude is 448.
__device__ float decode_e4m3(unsigned char b) {
  int e=(b>>3)&15, m=b&7;
  float v = e==0 ? ldexpf(float(m),-9) : ldexpf(1.0f+float(m)/8.0f,e-7);
  if(e==15 && m==7) v=NAN;
  return (b&128) ? -v : v;
}
__global__ void dequant(const unsigned char* q,const float* s,__nv_bfloat16* y,long n,int cols) {
  for(long i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=long(blockDim.x)*gridDim.x)
    y[i]=__float2bfloat16_rn(decode_e4m3(q[i])*s[i/cols]);
}
Tensor fp8_decode(Tensor q, Tensor s) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kByte && q.dim()==2 && q.is_contiguous(),"FP8 codes must be contiguous CUDA uint8 matrix");
  TORCH_CHECK(s.is_cuda() && s.device()==q.device() && s.scalar_type()==at::kFloat && s.numel()==q.size(0) && s.is_contiguous(),"Invalid FP8 row scales");
  c10::cuda::CUDAGuard guard(q.device());
  auto y=at::empty(q.sizes(),q.options().dtype(at::kBFloat16));
  if(q.numel()) {
    dequant<<<std::min<int64_t>(65535,(q.numel()+255)/256),256,0,at::cuda::getCurrentCUDAStream()>>>(q.data_ptr<unsigned char>(),s.data_ptr<float>(),reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),q.numel(),q.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return y;
}
// One block per value head. Threads own value columns, state is FP32 [H,K,V].
// Sequential scan is a correctness-first implementation; chunk algebra/fusion is future work.
__global__ void scan(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,
                     const float* g,const float* b,float* state,__nv_bfloat16* out,int T,int H,int K,int V) {
  int h=blockIdx.x, j=threadIdx.x;
  if(j>=V) return;
  float* st=state+(h*K)*V+j;
  for(int t=0;t<T;t++) {
    int base=(t*H+h)*K;
    float decay=expf(g[t*H+h]), memory=0;
    for(int i=0;i<K;i++) { st[i*V]*=decay; memory+=st[i*V]*__bfloat162float(k[base+i]); }
    float delta=(__bfloat162float(v[(t*H+h)*V+j])-memory)*b[t*H+h];
    float result=0;
    for(int i=0;i<K;i++) {
      st[i*V]+=__bfloat162float(k[base+i])*delta;
      result+=st[i*V]*__bfloat162float(q[base+i]);
    }
    out[(t*H+h)*V+j]=__float2bfloat16_rn(result*rsqrtf(float(K)));
  }
}
Tensor delta_scan(Tensor q,Tensor k,Tensor v,Tensor g,Tensor b,Tensor state) {
  TORCH_CHECK(q.dim()==3 && k.sizes()==q.sizes() && v.dim()==3,"Invalid GDN shapes");
  auto T=q.size(0), H=q.size(1), K=q.size(2), V=v.size(2);
  TORCH_CHECK(T>0 && H>0 && K>0 && V>0 && V<=1024 && v.size(0)==T && v.size(1)==H,"Invalid GDN dimensions");
  for(auto x:{q,k,v}) TORCH_CHECK(x.is_cuda() && x.device()==q.device() && x.scalar_type()==at::kBFloat16 && x.is_contiguous(),"GDN q/k/v must be CUDA BF16 contiguous");
  for(auto x:{g,b,state}) TORCH_CHECK(x.is_cuda() && x.device()==q.device() && x.scalar_type()==at::kFloat && x.is_contiguous(),"GDN state/gates must be CUDA FP32 contiguous");
  TORCH_CHECK(g.numel()==T*H && b.numel()==T*H && state.dim()==3 && state.size(0)==H && state.size(1)==K && state.size(2)==V,"Invalid GDN state/gates");
  c10::cuda::CUDAGuard guard(q.device());
  auto out=at::empty_like(v);
  scan<<<H,((V+31)/32)*32,0,at::cuda::getCurrentCUDAStream()>>>(
    reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
    reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),g.data_ptr<float>(),b.data_ptr<float>(),state.data_ptr<float>(),
    reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),T,H,K,V);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
}