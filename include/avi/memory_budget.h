#pragma once
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <limits>
#include <nlohmann/json.hpp>
namespace avi {
inline uint64_t checked_product(uint64_t a,uint64_t b){if(b&&a>std::numeric_limits<uint64_t>::max()/b)throw std::overflow_error("Memory size overflow");return a*b;}
inline uint64_t checked_add(uint64_t a,uint64_t b){if(a>std::numeric_limits<uint64_t>::max()-b)throw std::overflow_error("Memory sum overflow");return a+b;}
inline uint64_t session_bytes(const nlohmann::json& config,int tp,int tokens){
 if(tp<=0||tokens<=0)throw std::invalid_argument("Invalid reservation geometry");
 auto& t=config.at("text_config");uint64_t total=0;
 auto get=[&](const char* name){auto n=t.at(name).get<int64_t>();if(n<=0)throw std::invalid_argument("Invalid model dimension");return uint64_t(n);};
 auto shard=[&](const char* name){auto n=get(name);if(n%tp)throw std::invalid_argument("Non-divisible TP");return n/tp;};
 for(auto& layer:t.at("layer_types")) {
   if(layer=="full_attention")total=checked_add(total,checked_product(checked_product(4ULL*tokens,shard("num_key_value_heads")),get("head_dim")));
   else if(layer=="linear_attention"){
     auto k=get("linear_key_head_dim"),v=get("linear_value_head_dim"),h=shard("linear_num_value_heads"),hk=shard("linear_num_key_heads");
     total=checked_add(total,checked_product(checked_product(4*h,k),v));
     total=checked_add(total,checked_product(2*(2*hk*k+h*v),get("linear_conv_kernel_dim")-1));
   }else throw std::invalid_argument("Unsupported layer type");
 }
 // Full prompt embeddings, positions/ids and logits remain live across prefill slices.
 total=checked_add(total,checked_product(tokens,2*get("hidden_size")+32));
 return checked_add(total,4*get("vocab_size"));
}
class MemoryBudget {
 uint64_t limit_,used_=0;std::unordered_map<int,uint64_t> reservations_;
 public:
 explicit MemoryBudget(uint64_t limit):limit_(limit){}
 bool fits(uint64_t bytes)const{return bytes<=limit_-used_;}
 bool reserve(int id,uint64_t bytes){if(reservations_.count(id))throw std::logic_error("Duplicate reservation");if(!fits(bytes))return false;reservations_[id]=bytes;used_+=bytes;return true;}
 void release(int id){auto it=reservations_.find(id);if(it==reservations_.end())throw std::logic_error("Unknown reservation");used_-=it->second;reservations_.erase(it);}
 uint64_t used()const{return used_;}uint64_t limit()const{return limit_;}
};
}
