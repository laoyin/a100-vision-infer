#pragma once
#include <ATen/ATen.h>
namespace avi {
at::Tensor fp8_linear(at::Tensor x,at::Tensor codes,at::Tensor scales);
at::Tensor fused_rms(at::Tensor x,at::Tensor weight,double eps,bool one_center);
at::Tensor fused_swiglu(at::Tensor gate_up);
at::Tensor fused_l2(at::Tensor x);
at::Tensor fused_rope(at::Tensor x,at::Tensor positions,int rotary,double theta,int height_section,int width_section);
at::Tensor conv_decode(at::Tensor x,at::Tensor weight,at::Tensor history);
at::Tensor gqa_decode(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor keys,at::Tensor values,at::Tensor offset);
at::Tensor delta_scan_fast(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor g,at::Tensor beta,at::Tensor state);
}