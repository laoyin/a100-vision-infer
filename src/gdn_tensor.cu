// Fixed K=128 GDN prefill for Ampere. Algorithm: chunked gated delta / WY.
// References: fla-org/flash-linear-attention (MIT), Gated Delta Networks
// arXiv:2412.06464. This CUDA implementation is original, not copied source.
#include "avi/ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <algorithm>
#include <climits>
namespace avi {
namespace {
using bf=__nv_bfloat16;
namespace wm=nvcuda::wmma;
const bf* rd(const at::Tensor& x){return reinterpret_cast<const bf*>(x.data_ptr<at::BFloat16>());}
bf* wr(at::Tensor& x){return reinterpret_cast<bf*>(x.data_ptr<at::BFloat16>());}

// Grouped Q/K are packed once per key head, without expanded input tensors.
// All padded positions have zero vectors/beta and unchanged cumulative gates.
template<int C> __global__ void pack_gdn(const bf* q,const bf* k,const bf* v,
 const float* g,const float* beta,bf* Q,bf* K,bf* V,float* G,float* B,
 int T,int H,int HQ,int VD,int64_t qt,int64_t qh,int64_t kt,int64_t kh,
 int64_t vt,int64_t vh,int64_t gt,int64_t gh,int64_t bt,int64_t bh){
 int h=blockIdx.x%H,b=blockIdx.x/H,hq=h/(H/HQ),d=threadIdx.x;
 int64_t base=int64_t(b*H+h)*C,qbase=int64_t(b*HQ+hq)*C;
 float sum=0.f;
 for(int i=0;i<C;i++){
  int t=b*C+i;bool live=t<T;
  if(d==0){sum+=live?g[int64_t(t)*gt+h*gh]:0.f;G[base+i]=sum;B[base+i]=live?beta[int64_t(t)*bt+h*bh]:0.f;}
  if(d<128&&h%(H/HQ)==0){Q[(qbase+i)*128+d]=live?q[int64_t(t)*qt+hq*qh+d]:__float2bfloat16_rn(0.f);
             K[(qbase+i)*128+d]=live?k[int64_t(t)*kt+hq*kh+d]:__float2bfloat16_rn(0.f);}
  for(int j=d;j<VD;j+=128)V[(base+i)*VD+j]=live?v[int64_t(t)*vt+h*vh+j]:__float2bfloat16_rn(0.f);
 }
}

// Four warps calculate 16x16 KKT / QKT tiles using BF16 Tensor Cores,
// then apply causal gates in FP32. No dense mask/decay intermediate.
template<int C> __global__ void intra_gdn(const bf* Q,const bf* K,
 const float* G,const float* beta,float* L,float* A,int H,int HQ){
 __shared__ __align__(32) bf left[4][256],right[4][256];
 __shared__ __align__(32) float result[4][256];
 int warp=threadIdx.x/32,lane=threadIdx.x%32;
 int tile=blockIdx.y*4+warp,rows=C/16;
 int r0=(tile/rows)*16,c0=(tile%rows)*16;
 int64_t qbase=int64_t(blockIdx.x)*C;
 int block=blockIdx.x/HQ,hq=blockIdx.x%HQ;
 for(int kind=0;kind<2;kind++){
  wm::fragment<wm::accumulator,16,16,16,float> acc;wm::fill_fragment(acc,0.f);
  for(int kk=0;kk<128;kk+=16){
   for(int i=lane;i<256;i+=32){
    left[warp][i]=(kind?Q:K)[(qbase+r0+i/16)*128+kk+i%16];
    // Column-major B: B[k,j] = K[j,k].
    right[warp][i]=K[(qbase+c0+i/16)*128+kk+i%16];
   }
   __syncwarp();
   wm::fragment<wm::matrix_a,16,16,16,bf,wm::row_major> a;
   wm::fragment<wm::matrix_b,16,16,16,bf,wm::col_major> b;
   wm::load_matrix_sync(a,left[warp],16);wm::load_matrix_sync(b,right[warp],16);
   wm::mma_sync(acc,a,b,acc);__syncwarp();
  }
  wm::store_matrix_sync(result[warp],acc,16,wm::mem_row_major);__syncwarp();
  for(int i=lane;i<256;i+=32){
   int r=r0+i/16,c=c0+i%16;
   // KKT/QKT depend only on the shared Q/K head. Reuse the dot product
   // across all value heads; only the gates and beta differ.
   for(int h=hq*(H/HQ);h<(hq+1)*(H/HQ);h++){
    int64_t base=int64_t(block*H+h)*C;float x=0.f;
    if(c<=r)x=result[warp][i]*expf(G[base+r]-G[base+c]);
    if(kind)A[(base+r)*C+c]=x;
    else L[(base+r)*C+c]=c<r?x*beta[base+r]:0.f;
   }
  }
  __syncwarp();
 }
}

// Unit-lower triangular substitution: one lane owns an RHS column.
// Shared-memory rows avoid register-array spills and vendor solver launches.
template<int C> __global__ void solve_gdn(const bf* Q,const bf* K,const bf* V,
 const float* G,const float* beta,const float* L,float* W,float* U,
 float* SQ,float* WK,float* last,int VD,int H,int HQ){
 __shared__ float lower[C*C],solved[C*64];
 int64_t base=int64_t(blockIdx.x)*C;
 int h=blockIdx.x%H,block=blockIdx.x/H;
 int64_t qbase=int64_t(block*HQ+h/(H/HQ))*C;
 int d=blockIdx.y*64+threadIdx.x;
 for(int i=threadIdx.x;i<C*C;i+=64)lower[i]=L[base*C+i];
 __syncthreads();
 float end=G[base+C-1];
 if(threadIdx.x==0&&blockIdx.y==0)last[blockIdx.x]=end;
 for(int r=0;r<C;r++){
  float value=0.f;
  if(d<128)value=beta[base+r]*expf(G[base+r])*__bfloat162float(K[(qbase+r)*128+d]);
  else if(d<128+VD)value=beta[base+r]*__bfloat162float(V[(base+r)*VD+d-128]);
  for(int c=0;c<r;c++)value-=lower[r*C+c]*solved[c*64+threadIdx.x];
  // Each lane consumes only its own previously written column.
  solved[r*64+threadIdx.x]=value;
  if(d<128){
   W[(base+r)*128+d]=value;
   SQ[(base+r)*128+d]=__bfloat162float(Q[(qbase+r)*128+d])*expf(G[base+r]);
   WK[(base+r)*128+d]=__bfloat162float(K[(qbase+r)*128+d])*expf(end-G[base+r]);
  }else if(d<128+VD)U[(base+r)*VD+d-128]=value;
 }
}

// TF32x4: retain FP32 state storage and accumulators; add the low-part product and both cross
// products instead of rounding the entire recurrent state to one TF32 operand.
// This is a distinct finite-precision path; the GPU tests qualify it against an FP64 oracle.
// TF32 has ten explicit mantissa bits. Explicit truncation also avoids a
// toolkit-specific WMMA conversion intrinsic; splitting tests use this format.
__device__ float tf32_part(float x){return __uint_as_float(__float_as_uint(x)&0xffffe000U);}
struct __align__(32) FloatTile {float ah[128],al[128],bh[128],bl[128],out[256];};
__device__ void product(const float* lhs,int lr,int lc,const float* rhs,int rr,int rc,
 int reduction,FloatTile& tmp){
 int lane=threadIdx.x%32;
 wm::fragment<wm::accumulator,16,16,8,float> acc;wm::fill_fragment(acc,0.f);
 for(int kk=0;kk<reduction;kk+=8){
  for(int i=lane;i<128;i+=32){
   float a=lhs[(i/8)*lr+(kk+i%8)*lc],b=rhs[(kk+i/16)*rr+(i%16)*rc];
   tmp.ah[i]=tf32_part(a);tmp.al[i]=tf32_part(a-tmp.ah[i]);
   tmp.bh[i]=tf32_part(b);tmp.bl[i]=tf32_part(b-tmp.bh[i]);
  }
  __syncwarp();
  wm::fragment<wm::matrix_a,16,16,8,wm::precision::tf32,wm::row_major> ah,al;
  wm::fragment<wm::matrix_b,16,16,8,wm::precision::tf32,wm::row_major> bh,bl;
  wm::load_matrix_sync(ah,tmp.ah,8);wm::load_matrix_sync(al,tmp.al,8);
  wm::load_matrix_sync(bh,tmp.bh,16);wm::load_matrix_sync(bl,tmp.bl,16);
  wm::mma_sync(acc,al,bl,acc);wm::mma_sync(acc,al,bh,acc);wm::mma_sync(acc,ah,bl,acc);wm::mma_sync(acc,ah,bh,acc);
  __syncwarp();
 }
 wm::store_matrix_sync(tmp.out,acc,16,wm::mem_row_major);__syncwarp();
}

// Each CTA owns 16 value columns for one head for the entire sequence.
// State remains FP32 in shared memory; no per-chunk host loop or state copies.
template<int C> __global__ void propagate_gdn(const float* W,const float* U,
 const float* Q,const float* A,const float* keys,const float* last,
 float* state,bf* output,int blocks,int H,int VD,int T){
 __shared__ __align__(32) float s[128*16],update[C*16],ys[C*16];
 __shared__ FloatTile tmp[4];
 int h=blockIdx.x,j0=blockIdx.y*16,warp=threadIdx.x/32,lane=threadIdx.x%32;
 for(int i=threadIdx.x;i<128*16;i+=128)s[i]=state[(h*128+i/16)*VD+j0+i%16];
 __syncthreads();
 for(int b=0;b<blocks;b++){
  int64_t base=int64_t(b*H+h)*C;
  for(int r0=warp*16;r0<C;r0+=64){
   product(W+(base+r0)*128,128,1,s,16,1,128,tmp[warp]);
   for(int i=lane;i<256;i+=32)update[(r0+i/16)*16+i%16]=U[(base+r0+i/16)*VD+j0+i%16]-tmp[warp].out[i];
   __syncwarp();
   product(Q+(base+r0)*128,128,1,s,16,1,128,tmp[warp]);
   for(int i=lane;i<256;i+=32)ys[(r0+i/16)*16+i%16]=tmp[warp].out[i];
   __syncwarp();
  }
  __syncthreads();
  for(int r0=warp*16;r0<C;r0+=64){
   product(A+(base+r0)*C,C,1,update,16,1,C,tmp[warp]);
   for(int i=lane;i<256;i+=32){
    int t=b*C+r0+i/16;
    if(t<T)output[(int64_t(t)*H+h)*VD+j0+i%16]=__float2bfloat16_rn((ys[(r0+i/16)*16+i%16]+tmp[warp].out[i])*rsqrtf(128.f));
   }
   __syncwarp();
  }
  // All update/old-state readers complete before writing the new state.
  __syncthreads();
  float decay=expf(last[b*H+h]);
  // Products below consume update/keys, not old shared state. Each warp
  // owns disjoint state rows, so the stores cannot affect another product.
  for(int r0=warp*16;r0<128;r0+=64){
   product(keys+base*128+r0,1,128,update,16,1,C,tmp[warp]);
   for(int i=lane;i<256;i+=32){
    int at=(r0+i/16)*16+i%16;s[at]=s[at]*decay+tmp[warp].out[i];
   }
   __syncwarp();
  }
  __syncthreads();
 }
 for(int i=threadIdx.x;i<128*16;i+=128)state[(h*128+i/16)*VD+j0+i%16]=s[i];
}
template<int C> at::Tensor launch_gdn(at::Tensor q,at::Tensor k,at::Tensor v,
 at::Tensor g,at::Tensor beta,at::Tensor state){
 int T=q.size(0),HQ=q.size(1),H=v.size(1),VD=v.size(2),blocks=(T+C-1)/C;
 auto f=q.options().dtype(at::kFloat);
 auto Q=at::empty({blocks,HQ,C,128},q.options()),K=at::empty_like(Q);
 auto V=at::empty({blocks,H,C,VD},q.options());
 auto G=at::empty({blocks,H,C},f),B=at::empty_like(G);
 auto L=at::empty({blocks,H,C,C},f),A=at::empty_like(L);
 auto W=at::empty({blocks,H,C,128},f),SQ=at::empty_like(W),WK=at::empty_like(W);
 auto U=at::empty({blocks,H,C,VD},f),last=at::empty({blocks,H},f),out=at::empty_like(v);
 auto stream=at::cuda::getCurrentCUDAStream();
 pack_gdn<C><<<blocks*H,128,0,stream>>>(rd(q),rd(k),rd(v),g.data_ptr<float>(),beta.data_ptr<float>(),
  wr(Q),wr(K),wr(V),G.data_ptr<float>(),B.data_ptr<float>(),T,H,HQ,VD,
  q.stride(0),q.stride(1),k.stride(0),k.stride(1),v.stride(0),v.stride(1),
  g.stride(0),g.stride(1),beta.stride(0),beta.stride(1));
 intra_gdn<C><<<dim3(blocks*HQ,C*C/1024),128,0,stream>>>(rd(Q),rd(K),G.data_ptr<float>(),B.data_ptr<float>(),L.data_ptr<float>(),A.data_ptr<float>(),H,HQ);
 solve_gdn<C><<<dim3(blocks*H,(128+VD+63)/64),64,0,stream>>>(rd(Q),rd(K),rd(V),G.data_ptr<float>(),B.data_ptr<float>(),L.data_ptr<float>(),
  W.data_ptr<float>(),U.data_ptr<float>(),SQ.data_ptr<float>(),WK.data_ptr<float>(),last.data_ptr<float>(),VD,H,HQ);
 propagate_gdn<C><<<dim3(H,VD/16),128,0,stream>>>(W.data_ptr<float>(),U.data_ptr<float>(),SQ.data_ptr<float>(),A.data_ptr<float>(),
  WK.data_ptr<float>(),last.data_ptr<float>(),state.data_ptr<float>(),wr(out),blocks,H,VD,T);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
}
at::Tensor delta_scan_tensor(at::Tensor q,at::Tensor k,at::Tensor v,
 at::Tensor g,at::Tensor beta,at::Tensor state,int chunk){
 TORCH_CHECK(q.is_cuda()&&q.dim()==3&&q.scalar_type()==at::kBFloat16&&q.size(2)==128&&q.stride(2)==1&&
  k.device()==q.device()&&k.sizes()==q.sizes()&&k.scalar_type()==q.scalar_type()&&k.stride(2)==1&&
  v.device()==q.device()&&v.dim()==3&&v.scalar_type()==q.scalar_type()&&v.stride(2)==1,
  "Tensor GDN expects CUDA BF16 Q/K with K=128 and BF16 V");
 auto T=q.size(0),H=v.size(1),HQ=q.size(1),VD=v.size(2);
 TORCH_CHECK(T>0&&T<=65536&&H>0&&H<=64&&HQ>0&&H%HQ==0&&VD>0&&VD<=128&&VD%16==0&&
  v.size(0)==T&&g.device()==q.device()&&beta.device()==q.device()&&
  g.sizes()==at::IntArrayRef({T,H})&&beta.sizes()==g.sizes()&&
  g.scalar_type()==at::kFloat&&beta.scalar_type()==at::kFloat&&
  state.device()==q.device()&&state.scalar_type()==at::kFloat&&
  state.sizes()==at::IntArrayRef({H,128,VD})&&state.is_contiguous()&&
  (chunk==32||chunk==64),"Invalid tensor GDN geometry/dtypes");
 c10::cuda::CUDAGuard guard(q.device());
 // The physical output uses contiguous token/head/value strides.
 // Inputs may be strided views; the pack kernel consumes their actual strides.
 v=v.contiguous();
 return chunk==64?launch_gdn<64>(q,k,v,g,beta,state):launch_gdn<32>(q,k,v,g,beta,state);
}
}
