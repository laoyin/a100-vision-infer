#include "avi/ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
namespace avi {
using at::Tensor; using bf=__nv_bfloat16;
static const bf* ptr(const Tensor& x){return reinterpret_cast<const bf*>(x.data_ptr<at::BFloat16>());}
static bf* outptr(Tensor& x){return reinterpret_cast<bf*>(x.data_ptr<at::BFloat16>());}
__device__ float fp8(unsigned char b){int e=(b>>3)&15,m=b&7;float f=e?ldexpf(1.f+m/8.f,e-7):ldexpf(float(m),-9);return b&128?-f:f;}
__device__ float warp_sum(float x){for(int d=16;d;d/=2)x+=__shfl_down_sync(0xffffffff,x,d);return x;}
__device__ float warp_max(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_down_sync(0xffffffff,x,d));return x;}
__device__ float block_sum(float x){__shared__ float buf[8];int lane=threadIdx.x%32,w=threadIdx.x/32;x=warp_sum(x);if(!lane)buf[w]=x;__syncthreads();x=threadIdx.x<8?buf[lane]:0;x=warp_sum(x);if(!threadIdx.x)buf[0]=x;__syncthreads();return buf[0];}
__device__ float block_max(float x){__shared__ float buf[8];int lane=threadIdx.x%32,w=threadIdx.x/32;x=warp_max(x);if(!lane)buf[w]=x;__syncthreads();x=threadIdx.x<8?buf[lane]:-INFINITY;x=warp_max(x);if(!threadIdx.x)buf[0]=x;__syncthreads();return buf[0];}
__global__ void gemv(const bf* x,const unsigned char* w,const float* scale,bf* y,int N,int K){
 int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32,batch=blockIdx.y;float sum=0;
 if(row<N){for(int k=lane;k<K;k+=32){float weight=__bfloat162float(__float2bfloat16_rn(fp8(w[row*K+k])*scale[row]));sum+=__bfloat162float(x[batch*K+k])*weight;}sum=warp_sum(sum);if(!lane)y[batch*N+row]=__float2bfloat16_rn(sum);}
}
// Four warps compute [16,64] output. FP8 is decoded only into shared BF16 tiles.
__global__ void gemm(const bf* x,const unsigned char* w,const float* scale,bf* y,int M,int N,int K){
 using namespace nvcuda;
 __shared__ __align__(32) bf a[16*16];__shared__ __align__(32) bf b[64*16];__shared__ __align__(32) float c[4*16*16];
 int warp=threadIdx.x/32,mi=blockIdx.y*16,ni=blockIdx.x*64;
 wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
 for(int base=0;base<K;base+=16){
  for(int i=threadIdx.x;i<256;i+=128){int r=i/16,k=i%16;a[i]=(mi+r<M&&base+k<K)?x[(mi+r)*K+base+k]:__float2bfloat16_rn(0);}
  for(int i=threadIdx.x;i<1024;i+=128){int n=i/16,k=i%16;b[i]=(ni+n<N&&base+k<K)?__float2bfloat16_rn(fp8(w[(ni+n)*K+base+k])*scale[ni+n]):__float2bfloat16_rn(0);}
  __syncthreads();
  wmma::fragment<wmma::matrix_a,16,16,16,bf,wmma::row_major> af;
  wmma::fragment<wmma::matrix_b,16,16,16,bf,wmma::col_major> bfmat;
  wmma::load_matrix_sync(af,a,16);wmma::load_matrix_sync(bfmat,b+warp*256,16);wmma::mma_sync(acc,af,bfmat,acc);__syncthreads();
 }
 wmma::store_matrix_sync(c+warp*256,acc,16,wmma::mem_row_major);__syncthreads();
 for(int i=threadIdx.x;i<1024;i+=128){int tile=i/256,r=(i%256)/16,col=i%16;if(mi+r<M&&ni+tile*16+col<N)y[(mi+r)*N+ni+tile*16+col]=__float2bfloat16_rn(c[i]);}
}
Tensor fp8_linear(Tensor x,Tensor w,Tensor s){
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&x.dim()==2&&x.is_contiguous(),"FP8 linear expects CUDA BF16 [M,K]");
 TORCH_CHECK(w.is_cuda()&&w.device()==x.device()&&w.scalar_type()==at::kByte&&w.dim()==2&&w.is_contiguous()&&x.size(1)==w.size(1),"Invalid FP8 weights");
 TORCH_CHECK(s.is_cuda()&&s.device()==x.device()&&s.scalar_type()==at::kFloat&&s.is_contiguous()&&s.numel()==w.size(0),"Invalid FP8 scales");
 int M=x.size(0),N=w.size(0),K=w.size(1);TORCH_CHECK(M>0&&N>0&&K>0,"Empty GEMM");auto y=at::empty({M,N},x.options());auto stream=at::cuda::getCurrentCUDAStream();
 if(M<=8)gemv<<<dim3((N+7)/8,M),256,0,stream>>>(ptr(x),w.data_ptr<unsigned char>(),s.data_ptr<float>(),outptr(y),N,K);
 else gemm<<<dim3((N+63)/64,(M+15)/16),128,0,stream>>>(ptr(x),w.data_ptr<unsigned char>(),s.data_ptr<float>(),outptr(y),M,N,K);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
__global__ void rms(const bf* x,const bf* w,bf* y,int D,float eps,bool centered){int r=blockIdx.x;float s=0;for(int i=threadIdx.x;i<D;i+=256){float v=__bfloat162float(x[r*D+i]);s+=v*v;}float inv=rsqrtf(block_sum(s)/D+eps);for(int i=threadIdx.x;i<D;i+=256){float v=__bfloat162float(x[r*D+i])*inv,weight=__bfloat162float(w[i]);if(!centered)v=__bfloat162float(__float2bfloat16_rn(v));y[r*D+i]=__float2bfloat16_rn(v*(weight+(centered?1.f:0.f)));}}
Tensor fused_rms(Tensor x,Tensor w,double eps,bool centered){x=x.contiguous();w=w.contiguous();TORCH_CHECK(x.is_cuda()&&w.device()==x.device()&&x.scalar_type()==at::kBFloat16&&w.scalar_type()==at::kBFloat16&&w.numel()==x.size(-1),"Invalid RMS input");auto y=at::empty_like(x);rms<<<x.numel()/x.size(-1),256,0,at::cuda::getCurrentCUDAStream()>>>(ptr(x),ptr(w),outptr(y),x.size(-1),eps,centered);C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
__global__ void swiglu(const bf* x,bf* y,int M,int D){for(int i=blockIdx.x*256+threadIdx.x;i<M*D;i+=gridDim.x*256){int r=i/D,j=i%D;float a=__bfloat162float(x[r*2*D+j]);float gate=__bfloat162float(__float2bfloat16_rn(a/(1+expf(-a))));y[i]=__float2bfloat16_rn(gate*__bfloat162float(x[r*2*D+D+j]));}}
Tensor fused_swiglu(Tensor x){TORCH_CHECK(x.is_contiguous()&&x.dim()==2&&x.scalar_type()==at::kBFloat16&&x.is_cuda()&&x.size(1)%2==0,"Invalid SwiGLU input");auto y=at::empty({x.size(0),x.size(1)/2},x.options());swiglu<<<std::min<int64_t>(65535,(y.numel()+255)/256),256,0,at::cuda::getCurrentCUDAStream()>>>(ptr(x),outptr(y),x.size(0),y.size(1));C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
__global__ void l2(const bf* x,bf* y,int D){int r=blockIdx.x;float s=0;for(int i=threadIdx.x;i<D;i+=256){float v=__bfloat162float(x[r*D+i]);s+=v*v;}float inv=rsqrtf(block_sum(s)+1e-6f);for(int i=threadIdx.x;i<D;i+=256)y[r*D+i]=__float2bfloat16_rn(__bfloat162float(x[r*D+i])*inv);}
Tensor fused_l2(Tensor x){x=x.contiguous();TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16,"Invalid L2 input");auto y=at::empty_like(x);l2<<<x.numel()/x.size(-1),256,0,at::cuda::getCurrentCUDAStream()>>>(ptr(x),outptr(y),x.size(-1));C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
__global__ void rope_kernel(const bf* x,const int64_t* pos,bf* y,int T,int H,int D,int R,float theta,int hs,int ws){for(int i=blockIdx.x*256+threadIdx.x;i<T*H*D;i+=gridDim.x*256){int d=i%D,t=i/(H*D);if(d>=R){y[i]=x[i];continue;}int f=d%(R/2),axis=(f%3==1&&f<hs*3)?1:((f%3==2&&f<ws*3)?2:0);float angle=float(pos[axis*T+t])*powf(theta,-2.f*f/R);float co=__bfloat162float(__float2bfloat16_rn(cosf(angle))),si=__bfloat162float(__float2bfloat16_rn(sinf(angle)));int other=d<R/2?i+R/2:i-R/2;float rotated=__bfloat162float(x[other])*(d<R/2?-1.f:1.f);float a=__bfloat162float(__float2bfloat16_rn(__bfloat162float(x[i])*co));float b=__bfloat162float(__float2bfloat16_rn(rotated*si));y[i]=__float2bfloat16_rn(a+b);}}
Tensor fused_rope(Tensor x,Tensor pos,int r,double theta,int hs,int ws){x=x.contiguous();pos=pos.contiguous();TORCH_CHECK(x.dim()==3&&x.is_cuda()&&x.scalar_type()==at::kBFloat16&&pos.device()==x.device()&&pos.scalar_type()==at::kLong&&pos.size(0)==3&&pos.size(1)==x.size(0)&&r>0&&r%2==0&&r<=x.size(2),"Invalid RoPE input");auto y=at::empty_like(x);rope_kernel<<<std::min<int64_t>(65535,(x.numel()+255)/256),256,0,at::cuda::getCurrentCUDAStream()>>>(ptr(x),pos.data_ptr<int64_t>(),outptr(y),x.size(0),x.size(1),x.size(2),r,theta,hs,ws);C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
__global__ void conv_one(const bf* x,const bf* w,bf* hist,bf* y,int C,int K){int c=blockIdx.x*256+threadIdx.x;if(c>=C)return;float sum=0;for(int j=0;j<K-1;j++)sum+=__bfloat162float(hist[c*(K-1)+j])*__bfloat162float(w[c*K+j]);sum+=__bfloat162float(x[c])*__bfloat162float(w[c*K+K-1]);for(int j=0;j<K-2;j++)hist[c*(K-1)+j]=hist[c*(K-1)+j+1];hist[c*(K-1)+K-2]=x[c];float rounded=__bfloat162float(__float2bfloat16_rn(sum));y[c]=__float2bfloat16_rn(rounded/(1+expf(-rounded)));}
Tensor conv_decode(Tensor x,Tensor w,Tensor history){x=x.contiguous();TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&w.device()==x.device()&&history.device()==x.device()&&w.scalar_type()==at::kBFloat16&&history.scalar_type()==at::kBFloat16&&w.dim()==3&&w.size(0)==x.numel()&&w.size(2)>=2&&history.numel()==x.numel()*(w.size(2)-1),"Invalid conv state");auto y=at::empty_like(x);conv_one<<<(x.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(ptr(x),ptr(w),outptr(history),outptr(y),x.numel(),w.size(2));C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
__global__ void append_kv(const bf* k,const bf* v,bf* keys,bf* vals,const int64_t* offset,int HK,int D){int i=blockIdx.x*256+threadIdx.x;if(i<HK*D){keys[offset[0]*HK*D+i]=k[i];vals[offset[0]*HK*D+i]=v[i];}}
__global__ void attention_parts(const bf* q,const bf* keys,const bf* vals,const int64_t* offset,float* partial,float* stats,int H,int HK,int D,int P){
 int h=blockIdx.x,p=blockIdx.y,lane=threadIdx.x%32,warp=threadIdx.x/32,base=p*256,L=int(offset[0])+1,hk=h/(H/HK);__shared__ float scores[256];
 for(int i=warp;i<256;i+=8){float dot=0;if(base+i<L){for(int d=lane;d<D;d+=32)dot+=__bfloat162float(q[h*D+d])*__bfloat162float(keys[((base+i)*HK+hk)*D+d]);dot=warp_sum(dot)*rsqrtf(float(D));}else dot=-INFINITY;if(!lane)scores[i]=dot;}
 __syncthreads();float mx=block_max(scores[threadIdx.x]);float prob=isfinite(mx)?expf(scores[threadIdx.x]-mx):0;float sm=block_sum(prob);scores[threadIdx.x]=prob;__syncthreads();
 if(!threadIdx.x){stats[(h*P+p)*2]=mx;stats[(h*P+p)*2+1]=sm;}
 for(int d=threadIdx.x;d<D;d+=256){float value=0;for(int i=0;i<256&&base+i<L;i++)value+=scores[i]*__bfloat162float(vals[((base+i)*HK+hk)*D+d]);partial[(h*P+p)*D+d]=value;}
}
__global__ void attention_merge(const float* partial,const float* stats,bf* y,int D,int P){int h=blockIdx.x;float m=-INFINITY;for(int p=threadIdx.x;p<P;p+=256)m=fmaxf(m,stats[(h*P+p)*2]);m=block_max(m);float sum=0;for(int p=threadIdx.x;p<P;p+=256)sum+=stats[(h*P+p)*2+1]*expf(stats[(h*P+p)*2]-m);sum=block_sum(sum);for(int d=threadIdx.x;d<D;d+=256){float result=0;for(int p=0;p<P;p++)result+=partial[(h*P+p)*D+d]*expf(stats[(h*P+p)*2]-m);y[h*D+d]=__float2bfloat16_rn(result/sum);}}
Tensor gqa_decode(Tensor q,Tensor k,Tensor v,Tensor keys,Tensor values,Tensor offset){q=q.contiguous();k=k.contiguous();v=v.contiguous();TORCH_CHECK(q.dim()==3&&q.size(0)==1&&q.scalar_type()==at::kBFloat16&&q.is_cuda()&&keys.is_contiguous()&&values.is_contiguous(),"Invalid decode attention");int H=q.size(1),D=q.size(2),HK=k.size(1),P=(keys.size(0)+255)/256;TORCH_CHECK(HK>0&&H%HK==0&&D<=512&&k.size(2)==D&&v.sizes()==k.sizes()&&offset.is_cuda()&&offset.scalar_type()==at::kLong&&offset.numel()==1,"Invalid GQA geometry");auto out=at::empty_like(q);auto partial=at::empty({H,P,D},q.options().dtype(at::kFloat)),stats=at::empty({H,P,2},q.options().dtype(at::kFloat));auto stream=at::cuda::getCurrentCUDAStream();append_kv<<<(HK*D+255)/256,256,0,stream>>>(ptr(k),ptr(v),outptr(keys),outptr(values),offset.data_ptr<int64_t>(),HK,D);attention_parts<<<dim3(H,P),256,0,stream>>>(ptr(q),ptr(keys),ptr(values),offset.data_ptr<int64_t>(),partial.data_ptr<float>(),stats.data_ptr<float>(),H,HK,D,P);attention_merge<<<H,256,0,stream>>>(partial.data_ptr<float>(),stats.data_ptr<float>(),outptr(out),D,P);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}
// Value-column tiling increases the number of blocks; recurrent state stays in registers across time.
template<int K> __global__ void register_scan(const bf* q,const bf* k,const bf* v,const float* g,const float* beta,float* state,bf* out,int T,int H,int V){
 int h=blockIdx.x,j=blockIdx.y*32+threadIdx.x;if(j>=V)return;float s[K];
 #pragma unroll
 for(int i=0;i<K;i++)s[i]=state[(h*K+i)*V+j];
 for(int t=0;t<T;t++){int base=(t*H+h)*K;float decay=expf(g[t*H+h]),memory=0;
 #pragma unroll
 for(int i=0;i<K;i++){s[i]*=decay;memory+=s[i]*__bfloat162float(k[base+i]);}
 float delta=(__bfloat162float(v[(t*H+h)*V+j])-memory)*beta[t*H+h],result=0;
 #pragma unroll
 for(int i=0;i<K;i++){s[i]+=__bfloat162float(k[base+i])*delta;result+=s[i]*__bfloat162float(q[base+i]);}
 out[(t*H+h)*V+j]=__float2bfloat16_rn(result*rsqrtf(float(K)));}
 #pragma unroll
 for(int i=0;i<K;i++)state[(h*K+i)*V+j]=s[i];
}
Tensor delta_scan_fast(Tensor q,Tensor k,Tensor v,Tensor g,Tensor beta,Tensor state){TORCH_CHECK(q.is_contiguous()&&k.is_contiguous()&&v.is_contiguous()&&g.is_contiguous()&&beta.is_contiguous()&&state.is_contiguous()&&q.is_cuda()&&q.scalar_type()==at::kBFloat16&&state.scalar_type()==at::kFloat,"Invalid register GDN inputs");auto y=at::empty_like(v);int K=q.size(2),T=q.size(0),H=q.size(1),V=v.size(2);auto stream=at::cuda::getCurrentCUDAStream();if(K==128)register_scan<128><<<dim3(H,(V+31)/32),32,0,stream>>>(ptr(q),ptr(k),ptr(v),g.data_ptr<float>(),beta.data_ptr<float>(),state.data_ptr<float>(),outptr(y),T,H,V);else if(K==16)register_scan<16><<<dim3(H,(V+31)/32),32,0,stream>>>(ptr(q),ptr(k),ptr(v),g.data_ptr<float>(),beta.data_ptr<float>(),state.data_ptr<float>(),outptr(y),T,H,V);else { TORCH_CHECK(false,"Register GDN supports K=16 or 128"); }C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
}