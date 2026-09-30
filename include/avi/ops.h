#pragma once
#include <ATen/ATen.h>
#include <utility>
#include <vector>
namespace avi {
at::Tensor small_linear_shared(at::Tensor x,at::Tensor weight,at::Tensor scales={});
std::vector<at::Tensor> fused_gdn_prepare(at::Tensor projected,at::Tensor weight,at::Tensor history,at::Tensor log_decay,at::Tensor bias,int HK,int H,int K,int V,bool grouped_qk=false);
at::Tensor wy_propagate(at::Tensor W,at::Tensor U,at::Tensor Q,at::Tensor A,at::Tensor keys,at::Tensor last,at::Tensor state,int T);
at::Tensor delta_scan_tensor(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor g,at::Tensor beta,at::Tensor state,int chunk=64);
at::Tensor delta_scan_wy(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor g,at::Tensor beta,at::Tensor state,int chunk=32,bool fused=false);
at::Tensor gqa_chunk_dynamic(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor keys,at::Tensor values,at::Tensor offset);
at::Tensor flash_prefill(at::Tensor q,at::Tensor keys,at::Tensor values);
at::Tensor fp8_linear(at::Tensor x,at::Tensor codes,at::Tensor scales,bool vector_gemv=false);
at::Tensor fused_rms(at::Tensor x,at::Tensor weight,double eps,bool one_center);
at::Tensor fused_swiglu(at::Tensor gate_up);
std::pair<at::Tensor,at::Tensor> fused_gdn_gates(at::Tensor a,at::Tensor b,at::Tensor log_decay,at::Tensor bias);
at::Tensor fused_rms_gate(at::Tensor x,at::Tensor weight,at::Tensor gate,double eps);
at::Tensor fused_sigmoid_gate(at::Tensor x,at::Tensor gate);
at::Tensor fused_l2(at::Tensor x);
at::Tensor fused_rope(at::Tensor x,at::Tensor positions,int rotary,double theta,int height_section,int width_section);
at::Tensor conv_decode(at::Tensor x,at::Tensor weight,at::Tensor history);
at::Tensor conv_prefill(at::Tensor input,at::Tensor weight);
at::Tensor gqa_decode(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor keys,at::Tensor values,at::Tensor offset);
at::Tensor gqa_chunk(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor keys,at::Tensor values,int old);
at::Tensor delta_scan_fast(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor g,at::Tensor beta,at::Tensor state,at::Tensor trajectory={},bool cooperative=false);
at::Tensor delta_scan_chunked(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor g,at::Tensor beta,at::Tensor state,int chunk=32);
}
