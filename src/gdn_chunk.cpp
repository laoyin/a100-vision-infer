#include "avi/ops.h"
#include <algorithm>
#include <cmath>
namespace avi {
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
