#include "avi/ops.h"
#include <algorithm>
#include <cmath>
namespace avi {
// State-independent WY transforms are batched across all chunks and heads.
at::Tensor delta_scan_wy(at::Tensor q,at::Tensor k,at::Tensor v,
    at::Tensor g,at::Tensor beta,at::Tensor state,int chunk){
  TORCH_CHECK(chunk>0&&chunk<=64&&q.is_cuda()&&q.dim()==3&&q.sizes()==k.sizes()&&
      v.dim()==3&&v.size(0)==q.size(0)&&v.size(1)==q.size(1)&&state.scalar_type()==at::kFloat,"Invalid WY inputs");
  int64_t T=q.size(0),H=q.size(1),K=q.size(2),V=v.size(2),blocks=(T+chunk-1)/chunk;
  TORCH_CHECK(T>0&&state.sizes()==at::IntArrayRef({H,K,V})&&g.sizes()==at::IntArrayRef({T,H})&&beta.sizes()==g.sizes(),"Invalid WY state");
  auto pack=[&](at::Tensor x){
    x=x.to(at::kFloat);
    if(blocks*chunk>T){auto shape=x.sizes().vec();shape[0]=blocks*chunk-T;x=at::cat({x,at::zeros(shape,x.options())},0);}
    if(x.dim()==3)return x.reshape({blocks,chunk,H,x.size(2)}).permute({0,2,1,3}).contiguous();
    return x.reshape({blocks,chunk,H}).permute({0,2,1}).contiguous();
  };
  auto Q=pack(q),keys=pack(k),values=pack(v),B=pack(beta).unsqueeze(-1),G=pack(g).cumsum(-1);
  auto exponential=G.exp().unsqueeze(-1);
  auto lower=at::ones({chunk,chunk},q.options().dtype(at::kBool)).tril();
  auto decay=(G.unsqueeze(-1)-G.unsqueeze(-2)).masked_fill(lower.logical_not(),0).exp()*lower;
  auto kt=keys.transpose(-1,-2);
  auto L=(at::matmul(keys,kt)*decay*B).tril(-1)+at::eye(chunk,G.options());
  auto solved=at::linalg_solve_triangular(L,at::cat({B*exponential*keys,B*values},-1),false,true,true);
  auto W=solved.narrow(-1,0,K),U=solved.narrow(-1,K,V);
  auto attention=at::matmul(Q,kt)*decay,scaled_q=Q*exponential;
  auto last=G.select(-1,chunk-1);
  auto weighted_keys=keys*(last.unsqueeze(-1)-G).exp().unsqueeze(-1);
  auto output=at::empty({blocks,H,chunk,V},v.options());
  for(int64_t block=0;block<blocks;++block){
    auto update=U[block]-at::matmul(W[block],state);
    auto y=(at::matmul(scaled_q[block],state)+at::matmul(attention[block],update))/std::sqrt(double(K));
    output[block].copy_(y.to(v.scalar_type()));
    state.copy_(last[block].exp().reshape({H,1,1})*state+at::matmul(weighted_keys[block].transpose(-1,-2),update));
  }
  return output.permute({0,2,1,3}).reshape({blocks*chunk,H,V}).narrow(0,0,T).contiguous();
}
// Blockwise delta recurrence expressed as a unit-lower triangular solve.
// For a block, U_i = beta_i (v_i - exp(G_i) k_i^T S0
//                  - sum_{j<i} exp(G_i-G_j) (k_i^T k_j) U_j).
// This parallelizes time within each block; S remains FP32.
at::Tensor delta_scan_chunked(at::Tensor q,at::Tensor k,at::Tensor v,
    at::Tensor g,at::Tensor beta,at::Tensor state,int chunk){
  TORCH_CHECK(chunk>0&&chunk<=64&&q.is_cuda()&&q.sizes()==k.sizes()&&q.dim()==3&&
      v.dim()==3&&q.size(0)==v.size(0)&&q.size(1)==v.size(1)&&
      state.scalar_type()==at::kFloat,"Invalid chunked GDN geometry");
  const auto T=q.size(0),H=q.size(1),K=q.size(2),V=v.size(2);
  TORCH_CHECK(state.sizes()==at::IntArrayRef({H,K,V})&&g.sizes()==at::IntArrayRef({T,H})&&beta.sizes()==g.sizes(),"Invalid chunked GDN state");
  auto output=at::empty_like(v);
  for(int64_t start=0;start<T;start+=chunk){
    int64_t n=std::min<int64_t>(chunk,T-start);
    auto Q=q.narrow(0,start,n).transpose(0,1).to(at::kFloat);
    auto keys=k.narrow(0,start,n).transpose(0,1).to(at::kFloat);
    auto values=v.narrow(0,start,n).transpose(0,1).to(at::kFloat);
    auto B=beta.narrow(0,start,n).t().to(at::kFloat).unsqueeze(-1);
    auto G=g.narrow(0,start,n).t().to(at::kFloat).cumsum(-1);
    auto lower=at::ones({n,n},q.options().dtype(at::kBool)).tril();
    // Mask before exp: upper-triangle positive differences could overflow.
    auto decay=(G.unsqueeze(-1)-G.unsqueeze(-2)).masked_fill(lower.logical_not(),0).exp()*lower;
    auto L=(at::matmul(keys,keys.transpose(-1,-2))*decay*B).tril(-1);
    L=L+at::eye(n,L.options()).unsqueeze(0);
    auto rhs=B*(values-G.exp().unsqueeze(-1)*at::matmul(keys,state));
    auto U=at::linalg_solve_triangular(L,rhs,false,true,true);
    auto local=at::matmul(at::matmul(Q,keys.transpose(-1,-2))*decay,U);
    auto result=(G.exp().unsqueeze(-1)*at::matmul(Q,state)+local)/std::sqrt(double(K));
    output.narrow(0,start,n).copy_(result.transpose(0,1).to(v.scalar_type()));
    auto last=G.select(-1,n-1);
    auto weighted=(last.unsqueeze(-1)-G).exp().unsqueeze(-1)*U;
    state.copy_(last.exp().reshape({H,1,1})*state+at::matmul(keys.transpose(-1,-2),weighted));
  }
  return output;
}
}
