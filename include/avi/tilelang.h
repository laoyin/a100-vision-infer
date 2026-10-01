#pragma once
#include <ATen/ATen.h>
#include <cuda_runtime_api.h>
#include <string>
namespace avi {
void configure_tilelang(const std::string& directory);
bool tilelang_configured();
void tilelang_gdn_prepare(const at::Tensor& Q,const at::Tensor& K,const at::Tensor& V,
 const at::Tensor& G,const at::Tensor& B,at::Tensor& A,at::Tensor& W,at::Tensor& U,
 at::Tensor& SQ,at::Tensor& WK,at::Tensor& last,int chunk,int blocks,int H,int HQ,cudaStream_t stream);
at::Tensor tilelang_fp8_small(at::Tensor x,at::Tensor codes,at::Tensor scales,int split);
}
