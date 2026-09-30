#pragma once
#include <ATen/ATen.h>
#include <nlohmann/json.hpp>
#include <nccl.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <functional>
namespace avi {
using at::Tensor;
using json = nlohmann::json;
Tensor fp8_decode(Tensor codes, Tensor scales);
Tensor delta_scan(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor state);
struct Weight { Tensor data, scale; };
struct EngineOptions { bool optimized=true; bool extra_fusions=false; bool cublas_prefill=false; bool cuda_graph=false; bool tp_lm_head=false; bool reference_prefill=false; bool vector_gemv=false; bool flash_prefill=false; bool cache_vision_weights=false; bool gdn_cooperative=false; bool fused_gdn_conv=false; bool bf16_tp_reduce=false; int mtp_tokens=0; size_t weight_cache_bytes=0; size_t image_cache_bytes=256ULL<<20; size_t prefix_cache_bytes=512ULL<<20; size_t host_prefix_cache_bytes=0; };
struct SpeculativeResult { std::vector<int64_t> tokens; int consumed=0,proposed=0,accepted=0; };
struct DecodeGraph;
struct DraftGraph;
class Engine {
 public:
  Engine(const std::string& model_dir, int rank, int world, int device, ncclComm_t comm, int capacity, EngineOptions options={});
  ~Engine();
  void activate(int session);
  void reserve_session(int session,int tokens);
  void drop(int session);
  Tensor decode(int64_t token,int64_t position,int consumed);
  Tensor decode_batch(const std::vector<int>& sessions,const std::vector<int64_t>& tokens,const std::vector<int64_t>& positions,const std::vector<int>& consumed);
  void save_prefix(const std::string& key,Tensor logits);
  Tensor restore_prefix(const std::string& key);
  json cache_stats() const;
  Tensor vision(const std::string& request_dir, const json& request);
  Tensor step(Tensor embeddings, Tensor positions);
  void set_trace_prefix(const std::string& prefix) { trace_prefix_=prefix; }
  Tensor embed(Tensor ids);
  Tensor logits(Tensor hidden);
  SpeculativeResult speculate(int64_t pending,int64_t position,int consumed,int budget,const std::vector<int64_t>& eos,
      const std::function<int64_t(Tensor)>& select={});
  int64_t greedy(Tensor logits);
  void enable_draft_graph(bool enabled) { draft_graph_enabled_=enabled; }
  void enable_gdn_chunk(bool enabled) { gdn_chunk_enabled_=enabled; }
  Tensor read_input(const std::string& dir, const json& desc);
  const json& config() const { return config_; }
 private:
  struct State { Tensor key, value, conv, recurrent; int length=0; };
  Tensor linear(Tensor x, const std::string& prefix, bool reduce=false);
  Tensor norm(Tensor x, const std::string& prefix, bool one_center=true);
  Tensor layer_norm(Tensor x, const std::string& prefix);
  Tensor full_attention(Tensor x, Tensor positions, int layer, const std::string& prefix, Tensor projected={}, bool finish=true, State* external=nullptr);
  Tensor delta_attention(Tensor x, int layer, const std::string& prefix, Tensor projected={}, bool finish=true);
  Tensor tensor(const std::string& name);
  Tensor sum(Tensor x);
  std::vector<State> states_;
  struct Draft { State state; Tensor hidden; };
  std::unordered_map<int,Draft> drafts_;
  Tensor mtp_step(Tensor embeddings,Tensor positions,Tensor hidden,bool single_decode=false);
  Tensor draft_one(int64_t token,int64_t position,Tensor hidden);
  Tensor local_candidates(Tensor normalized);
  std::vector<int64_t> gather_candidates(Tensor candidates);
  bool draft_graph_enabled_=false;
  std::unordered_map<int,std::shared_ptr<DraftGraph>> draft_graphs_;
  std::unordered_map<int,std::shared_ptr<DraftGraph>> draft_graph_pool_;
  void release_draft(int id);
  void advance_draft(Tensor embeddings,Tensor positions,Tensor target_hidden);
  Tensor project_logits(Tensor normalized);
  bool verifying_=false;
  bool gdn_chunk_enabled_=false;
  std::vector<Tensor> verify_recurrent_,verify_conv_;
  std::unordered_map<std::string,Tensor> decoded_weights_;
  size_t decoded_bytes_=0;
  std::unordered_map<std::string,Weight> weights_;
  std::unordered_map<std::string,std::vector<std::string>> mixed_projections_;
  json config_, text_, vision_;
  int rank_, world_, device_, capacity_;
  ncclComm_t comm_;
  double eps_;
  EngineOptions options_;
  std::string trace_prefix_;
  void trace_layer(Tensor hidden,int layer);
  bool decode_mode_=false;
  int active_=0;
  std::unordered_map<int,int> session_capacities_;
  int session_capacity() const;
  Tensor decode_offset_;
  std::unordered_map<int,std::vector<State>> sessions_;
  std::unordered_map<int,std::shared_ptr<DecodeGraph>> graphs_;
  struct Prefix {std::vector<State> states; Tensor logits; size_t bytes=0; uint64_t used=0;};
  struct Image {Tensor tensor; size_t bytes=0; uint64_t used=0;};
  std::unordered_map<std::string,Prefix> prefixes_,host_prefixes_;
  size_t host_prefix_bytes_=0;
  uint64_t host_prefix_hits_=0;
  void demote_prefix(const std::string& key);
  std::unordered_map<std::string,Image> images_;
  size_t prefix_bytes_=0,image_bytes_=0;
  uint64_t cache_clock_=0,prefix_hits_=0,image_hits_=0;
  void fuse_weights(const std::string& dest,const std::vector<std::string>& source);

};
json read_json(const std::string& file);
std::string request_hash(const std::string& dir,const json& request);
}
