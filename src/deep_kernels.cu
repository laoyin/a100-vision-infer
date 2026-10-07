#include "avi/fp8_codec.cuh"
#include "avi/ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
namespace avi {
namespace {
using bf=__nv_bfloat16;
const bf* read_bf(const at::Tensor& x){return reinterpret_cast<const bf*>(x.data_ptr<at::BFloat16>());}
bf* write_bf(at::Tensor& x){return reinterpret_cast<bf*>(x.data_ptr<at::BFloat16>());}
__device__ float reduce_warp(float x){for(int i=16;i;i/=2)x+=__shfl_down_sync(0xffffffff,x,i);return x;}
__device__ float reduce_block(float x){
 __shared__ float values[8];int lane=threadIdx.x%32,warp=threadIdx.x/32;
 x=reduce_warp(x);if(!lane)values[warp]=x;__syncthreads();
 x=threadIdx.x<8?values[lane]:0.f;x=reduce_warp(x);
 if(!threadIdx.x)values[0]=x;__syncthreads();float result=values[0];__syncthreads();return result;
}
__device__ float decode_e4m3(unsigned char code){return fp8_e4m3_value(code);}
// One warp per output row; all query rows share the weight/scale loads.
template<bool Quantized> __global__ void shared_gemv(const bf* x,const void* weight,
 const float* scales,bf* y,int M,int N,int K,int S){
 int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;
 if(row>=N)return;float accum[8]={0.f};
 for(int k=lane;k<K;k+=32){
  float w;
  if constexpr(Quantized)w=__bfloat162float(__float2bfloat16_rn(
     decode_e4m3(static_cast<const unsigned char*>(weight)[int64_t(row)*K+k])*scales[int64_t(row)*S+(S==1?0:k/128)]));
  else w=__bfloat162float(static_cast<const bf*>(weight)[int64_t(row)*K+k]);
  #pragma unroll
  for(int m=0;m<8;m++)if(m<M)accum[m]+=__bfloat162float(x[int64_t(m)*K+k])*w;
 }
 #pragma unroll
 for(int m=0;m<8;m++)if(m<M){float sum=reduce_warp(accum[m]);if(!lane)y[int64_t(m)*N+row]=__float2bfloat16_rn(sum);}
}
template<bool Quantized> __global__ void shared_gemv_packed(const bf* x,const void* weight,
 const float* scales,bf* y,int M,int N,int K,int S){
 int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;
 if(row>=N)return;float accum[8]={0.f};
 for(int k=lane*4;k<K;k+=128){
  float w[4];
  if constexpr(Quantized){
   unsigned packed=*reinterpret_cast<const unsigned*>(static_cast<const unsigned char*>(weight)+int64_t(row)*K+k);
   float scale=scales[int64_t(row)*S+(S==1?0:k/128)];
   #pragma unroll
   for(int j=0;j<4;j++)w[j]=__bfloat162float(__float2bfloat16_rn(decode_e4m3((packed>>(8*j))&255)*scale));
  }else{
   uint2 packed=*reinterpret_cast<const uint2*>(static_cast<const bf*>(weight)+int64_t(row)*K+k);
   #pragma unroll
   for(int j=0;j<4;j++){unsigned bits=j<2?packed.x:packed.y;w[j]=__bfloat162float(__ushort_as_bfloat16((bits>>(16*(j%2)))&65535));}
  }
  #pragma unroll
  for(int m=0;m<8;m++)if(m<M){
   uint2 activations=*reinterpret_cast<const uint2*>(x+int64_t(m)*K+k);
   #pragma unroll
   for(int j=0;j<4;j++){unsigned bits=j<2?activations.x:activations.y;accum[m]+=__bfloat162float(__ushort_as_bfloat16((bits>>(16*(j%2)))&65535))*w[j];}
  }
 }
 #pragma unroll
 for(int m=0;m<8;m++)if(m<M){float sum=reduce_warp(accum[m]);if(!lane)y[int64_t(m)*N+row]=__float2bfloat16_rn(sum);}
}

__device__ float convolved(const bf* input,const bf* weight,const bf* history,
 int t,int c,int stride,int width){
 float sum=0.f;
 for(int i=0;i<width;i++){
  int position=t+i-(width-1);
  bf v=position<0?history[c*(width-1)+position+width-1]:input[int64_t(position)*stride+c];
  sum+=__bfloat162float(v)*__bfloat162float(weight[c*width+i]);
 }
 float rounded=__bfloat162float(__float2bfloat16_rn(sum));
 return __bfloat162float(__float2bfloat16_rn(rounded/(1.f+expf(-rounded))));
}
// Conv/SiLU/L2/head expansion/gates without materializing [T,C] conv output.
__global__ void prepare_gdn(const bf* input,const bf* weight,const bf* history,
 const float* log_decay,const float* bias,bf* q,bf* k,bf* v,float* g,float* beta,
 int T,int H,int HK,int K,int V,int stride,int width,int QH){
 int t=blockIdx.x,h=blockIdx.y,hk=h/(H/HK),d=threadIdx.x;
 float qv=d<K?convolved(input,weight,history,t,hk*K+d,stride,width):0.f;
 float kv=d<K?convolved(input,weight,history,t,HK*K+hk*K+d,stride,width):0.f;
 float qsum=reduce_block(qv*qv),ksum=reduce_block(kv*kv);
 if(d<K&&(QH==H||h%(H/HK)==0)){q[(int64_t(t)*QH+(QH==H?h:hk))*K+d]=__float2bfloat16_rn(qv*rsqrtf(qsum+1e-6f));
         k[(int64_t(t)*QH+(QH==H?h:hk))*K+d]=__float2bfloat16_rn(kv*rsqrtf(ksum+1e-6f));}
 for(int j=d;j<V;j+=256)v[(int64_t(t)*H+h)*V+j]=__float2bfloat16_rn(
   convolved(input,weight,history,t,2*HK*K+h*V+j,stride,width));
 if(!d){
  int C=2*HK*K+H*V;float a=__bfloat162float(input[int64_t(t)*stride+C+H*V+H+h])+bias[h];
  g[t*H+h]=-expf(log_decay[h])*(a>20.f?a:log1pf(expf(a)));
  float b=__bfloat162float(input[int64_t(t)*stride+C+H*V+h]);
  beta[t*H+h]=__bfloat162float(__float2bfloat16_rn(1.f/(1.f+expf(-b))));
 }
}
__global__ void update_history(const bf* input,bf* history,int T,int C,int stride,int width){
 int i=blockIdx.x*256+threadIdx.x;if(i<C*(width-1))
 history[i]=input[int64_t(T-(width-1)+i%(width-1))*stride+i/(width-1)];
}
// Each value column keeps FP32 state in registers across every WY chunk.
// No per-chunk ATen launches, state copies or temporary update tensors.
template<int K> __global__ void propagate_wy(const float* W,const float* U,const float* Q,
 const float* A,const float* keys,const float* last,float* state,bf* output,
 int blocks,int H,int C,int V,int T){
 int h=blockIdx.x,j=blockIdx.y*32+threadIdx.x;if(j>=V)return;
 float s[K];__shared__ float update[64*32];
 #pragma unroll
 for(int k=0;k<K;k++)s[k]=state[(h*K+k)*V+j];
 for(int b=0;b<blocks;b++){
  int64_t base=int64_t(b*H+h)*C;
  for(int t=0;t<C;t++){
   float dot=0.f;
   #pragma unroll
   for(int k=0;k<K;k++)dot+=W[(base+t)*K+k]*s[k];
   update[t*32+threadIdx.x]=U[(base+t)*V+j]-dot;
  }
  for(int t=0;t<C&&b*C+t<T;t++){
   float value=0.f;
   #pragma unroll
   for(int k=0;k<K;k++)value+=Q[(base+t)*K+k]*s[k];
   for(int i=0;i<=t;i++)value+=A[(base+t)*C+i]*update[i*32+threadIdx.x];
   output[(int64_t(b*C+t)*H+h)*V+j]=__float2bfloat16_rn(value*rsqrtf(float(K)));
  }
  float decay=expf(last[b*H+h]);
  #pragma unroll
  for(int k=0;k<K;k++){
   float add=0.f;for(int t=0;t<C;t++)add+=keys[(base+t)*K+k]*update[t*32+threadIdx.x];
   s[k]=s[k]*decay+add;
  }
 }
 #pragma unroll
 for(int k=0;k<K;k++)state[(h*K+k)*V+j]=s[k];
}
}
at::Tensor small_linear_shared(at::Tensor x,at::Tensor weight,at::Tensor scales){
 TORCH_CHECK(x.is_cuda()&&x.dim()==2&&x.is_contiguous()&&x.scalar_type()==at::kBFloat16&&
   weight.dim()==2&&weight.is_contiguous()&&weight.device()==x.device()&&x.size(1)==weight.size(1),
   "Invalid shared GEMV inputs");
 int M=x.size(0),N=weight.size(0),K=x.size(1);
 TORCH_CHECK(M>=1&&M<=8&&N>0&&K>0,"Shared GEMV requires 1..8 rows");
 auto output=at::empty({M,N},x.options());auto stream=at::cuda::getCurrentCUDAStream();
 if(scales.defined()){
  TORCH_CHECK(weight.scalar_type()==at::kByte&&scales.device()==x.device()&&
    scales.scalar_type()==at::kFloat&&scales.is_contiguous()&&
    ((scales.dim()==1&&scales.numel()==N)||(scales.dim()==2&&scales.size(0)==N&&scales.size(1)==(K+127)/128)),
    "Invalid shared FP8 scales");
  int S=scales.dim()==1?1:scales.size(1);
  if(K%4==0&&reinterpret_cast<std::uintptr_t>(x.data_ptr())%8==0&&reinterpret_cast<std::uintptr_t>(weight.data_ptr())%4==0)
    shared_gemv_packed<true><<<(N+7)/8,256,0,stream>>>(read_bf(x),weight.data_ptr(),scales.data_ptr<float>(),write_bf(output),M,N,K,S);
  else shared_gemv<true><<<(N+7)/8,256,0,stream>>>(read_bf(x),weight.data_ptr(),scales.data_ptr<float>(),write_bf(output),M,N,K,S);
 }else{
  TORCH_CHECK(weight.scalar_type()==at::kBFloat16,"Shared GEMV expects BF16 or FP8 weights");
  if(K%4==0&&reinterpret_cast<std::uintptr_t>(x.data_ptr())%8==0&&reinterpret_cast<std::uintptr_t>(weight.data_ptr())%8==0)
    shared_gemv_packed<false><<<(N+7)/8,256,0,stream>>>(read_bf(x),weight.data_ptr(),nullptr,write_bf(output),M,N,K,1);
  else shared_gemv<false><<<(N+7)/8,256,0,stream>>>(read_bf(x),weight.data_ptr(),nullptr,write_bf(output),M,N,K,1);
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
std::vector<at::Tensor> fused_gdn_prepare(at::Tensor projected,at::Tensor weight,at::Tensor history,
 at::Tensor log_decay,at::Tensor bias,int HK,int H,int K,int V,bool grouped_qk){
 int64_t C=2*HK*K+H*V;
 TORCH_CHECK(projected.is_cuda()&&projected.dim()==2&&projected.scalar_type()==at::kBFloat16&&
   projected.is_contiguous()&&H>0&&HK>0&&H%HK==0&&K>0&&K<=256&&V>0&&
   projected.size(1)==C+H*V+2*H&&weight.dim()==3&&weight.size(0)==C&&weight.size(1)==1&&
   weight.size(2)>1&&history.sizes()==at::IntArrayRef({1,C,weight.size(2)-1})&&
   projected.size(0)>=weight.size(2)-1&&weight.scalar_type()==at::kBFloat16&&history.scalar_type()==at::kBFloat16&&
   weight.device()==projected.device()&&history.device()==projected.device()&&weight.is_contiguous()&&history.is_contiguous(),
   "Invalid fused GDN preparation");
 log_decay=log_decay.to(at::kFloat).contiguous();bias=bias.to(at::kFloat).contiguous();
 TORCH_CHECK(log_decay.device()==projected.device()&&bias.device()==projected.device()&&log_decay.numel()==H&&bias.numel()==H,
   "Invalid fused GDN gate parameters");
 int T=projected.size(0);
 int QH=grouped_qk?HK:H;
 auto q=at::empty({T,QH,K},projected.options()),k=at::empty_like(q),v=at::empty({T,H,V},projected.options());
 auto g=at::empty({T,H},projected.options().dtype(at::kFloat)),beta=at::empty_like(g);
 auto stream=at::cuda::getCurrentCUDAStream();
 prepare_gdn<<<dim3(T,H),256,0,stream>>>(read_bf(projected),read_bf(weight),read_bf(history),
   log_decay.data_ptr<float>(),bias.data_ptr<float>(),write_bf(q),write_bf(k),write_bf(v),
   g.data_ptr<float>(),beta.data_ptr<float>(),T,H,HK,K,V,projected.size(1),weight.size(2),QH);
 update_history<<<(C*(weight.size(2)-1)+255)/256,256,0,stream>>>(read_bf(projected),write_bf(history),T,C,projected.size(1),weight.size(2));
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {q,k,v,g,beta};
}
at::Tensor wy_propagate(at::Tensor W,at::Tensor U,at::Tensor Q,at::Tensor A,at::Tensor keys,
 at::Tensor last,at::Tensor state,int T){
 W=W.contiguous();U=U.contiguous();Q=Q.contiguous();A=A.contiguous();keys=keys.contiguous();last=last.contiguous();
 TORCH_CHECK(W.is_cuda()&&W.dim()==4&&W.scalar_type()==at::kFloat&&U.dim()==4&&U.device()==W.device()&&
   U.scalar_type()==at::kFloat&&state.device()==W.device()&&state.scalar_type()==at::kFloat&&state.is_contiguous(),
   "Invalid WY propagation");
 int B=W.size(0),H=W.size(1),C=W.size(2),K=W.size(3),V=U.size(3);
 TORCH_CHECK((K==16||K==128)&&C>0&&C<=64&&T>0&&T<=B*C&&U.sizes()==at::IntArrayRef({B,H,C,V})&&
   Q.sizes()==W.sizes()&&keys.sizes()==W.sizes()&&A.sizes()==at::IntArrayRef({B,H,C,C})&&
   last.sizes()==at::IntArrayRef({B,H})&&state.sizes()==at::IntArrayRef({H,K,V}),"Invalid WY geometry");
 TORCH_CHECK(Q.device()==W.device()&&A.device()==W.device()&&keys.device()==W.device()&&last.device()==W.device()&&
   Q.scalar_type()==at::kFloat&&A.scalar_type()==at::kFloat&&keys.scalar_type()==at::kFloat&&last.scalar_type()==at::kFloat,"Invalid WY transforms");
 auto output=at::empty({T,H,V},W.options().dtype(at::kBFloat16));auto stream=at::cuda::getCurrentCUDAStream();
 #define AVI_WY(KDIM) propagate_wy<KDIM><<<dim3(H,(V+31)/32),32,0,stream>>>(W.data_ptr<float>(),U.data_ptr<float>(),Q.data_ptr<float>(),A.data_ptr<float>(),keys.data_ptr<float>(),last.data_ptr<float>(),state.data_ptr<float>(),write_bf(output),B,H,C,V,T)
 if(K==128){AVI_WY(128);}else{AVI_WY(16);}
 #undef AVI_WY
 C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
}
