#pragma once
#include <ATen/ATen.h>
#include <nlohmann/json.hpp>
#include <nccl.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
namespace avi {
using at::Tensor;
using json = nlohmann::json;
Tensor fp8_decode(Tensor codes, Tensor scales);
Tensor delta_scan(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor state);
struct Weight { Tensor data, scale; };
struct EngineOptions { bool optimized=true; bool cuda_graph=false; size_t image_cache_bytes=256ULL<<20; size_t prefix_cache_bytes=512ULL<<20; size_t host_prefix_cache_bytes=0; };
struct DecodeGraph;
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
  Tensor embed(Tensor ids);
  Tensor logits(Tensor hidden);
  Tensor read_input(const std::string& dir, const json& desc);
  const json& config() const { return config_; }
 private:
  Tensor linear(Tensor x, const std::string& prefix, bool reduce=false);
  Tensor norm(Tensor x, const std::string& prefix, bool one_center=true);
  Tensor layer_norm(Tensor x, const std::string& prefix);
  Tensor full_attention(Tensor x, Tensor positions, int layer, const std::string& prefix, Tensor projected={}, bool finish=true);
  Tensor delta_attention(Tensor x, int layer, const std::string& prefix, Tensor projected={}, bool finish=true);
  Tensor tensor(const std::string& name);
  Tensor sum(Tensor x);
  struct State { Tensor key, value, conv, recurrent; int length=0; };
  std::vector<State> states_;
  std::unordered_map<std::string,Weight> weights_;
  std::unordered_map<std::string,std::vector<std::string>> mixed_projections_;
  json config_, text_, vision_;
  int rank_, world_, device_, capacity_;
  ncclComm_t comm_;
  double eps_;
  EngineOptions options_;
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