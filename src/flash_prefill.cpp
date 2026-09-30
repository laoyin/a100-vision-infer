#include "avi/ops.h"
#include <ATen/ops/_flash_attention_forward.h>
#include <optional>

namespace avi {
at::Tensor flash_prefill(at::Tensor q,at::Tensor keys,at::Tensor values) {
  TORCH_CHECK(q.is_cuda()&&q.dim()==3&&keys.dim()==3&&values.sizes()==keys.sizes(),
              "Invalid FlashAttention prefill tensors");
  TORCH_CHECK(q.scalar_type()==at::kBFloat16&&keys.scalar_type()==q.scalar_type()&&
              values.scalar_type()==q.scalar_type()&&q.device()==keys.device()&&
              q.device()==values.device(),"Flash prefill requires same-device BF16");
  TORCH_CHECK(q.size(0)>0&&keys.size(0)>=q.size(0)&&keys.size(1)>0&&
              q.size(1)%keys.size(1)==0&&q.size(2)==keys.size(2)&&
              q.size(2)>0&&q.size(2)<=256&&q.size(2)%8==0,"Invalid Flash prefill geometry");
  // The low-level FA2 operation uses [B,T,H,D], native GQA and bottom-right
  // causal alignment: query i attends through KV length - query length + i.
  // Do not replace with SDPA is_causal=true (top-left semantics for unequal T).
  auto result=at::_flash_attention_forward(q.unsqueeze(0),keys.unsqueeze(0),values.unsqueeze(0),
      std::nullopt,std::nullopt,q.size(0),keys.size(0),0.0,true,false);
  return std::get<0>(result).squeeze(0);
}
}
