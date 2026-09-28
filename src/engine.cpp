#include "avi/engine.h"
#include "avi/ops.h"
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <openssl/evp.h>
#include <iomanip>
#include <sstream>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <cmath>
#include <limits>
#include <functional>
#include <algorithm>
namespace avi {
json read_json(const std::string& file) { std::ifstream f(file); TORCH_CHECK(f.good(),"Cannot open ",file); json j; f>>j; return j; }
static Tensor raw(const std::string& dir,const json& d,int device) {
  auto relative=std::filesystem::path(d.at("file").get<std::string>());
  TORCH_CHECK(!relative.is_absolute(),"Absolute tensor path rejected");
  for(auto& part:relative) TORCH_CHECK(part!="..","Path traversal rejected");
  auto shape=d.at("shape").get<std::vector<int64_t>>();
  TORCH_CHECK(!shape.empty(),"Scalar tensors must be exported as [1]");
  int64_t n=1; for(auto x:shape) { TORCH_CHECK(x>0 && n<=std::numeric_limits<int64_t>::max()/x,"Invalid tensor shape"); n*=x; }
  std::string dtype=d.at("dtype"); at::ScalarType type;
  if(dtype=="BF16") type=at::kBFloat16; else if(dtype=="F32") type=at::kFloat;
  else if(dtype=="I64") type=at::kLong; else if(dtype=="U8") type=at::kByte;
  else { TORCH_CHECK(false,"Unsupported dtype ",dtype); }
  auto out=at::empty(shape,at::TensorOptions().dtype(type).device(at::kCPU));
  auto file=std::filesystem::path(dir)/relative;
  TORCH_CHECK(std::filesystem::file_size(file)==out.nbytes(),"Tensor byte count mismatch: ",file.string());
  std::ifstream f(file,std::ios::binary); f.read(static_cast<char*>(out.data_ptr()),out.nbytes());
  TORCH_CHECK(f.good(),"Short tensor read: ",file.string());
  return out.to(at::Device(at::kCUDA,device));
}
Engine::Engine(const std::string& dir,int rank,int world,int device,ncclComm_t comm,int capacity,EngineOptions options)
 :rank_(rank),world_(world),device_(device),capacity_(capacity),comm_(comm),options_(options) {
  TORCH_CHECK(capacity>0,"Invalid capacity");
  TORCH_CHECK(!std::filesystem::exists(dir+"/INCOMPLETE"),"Incomplete model export");
    auto manifest=read_json(dir+"/manifest.json");
  TORCH_CHECK(manifest.at("format")=="avi-v1" && manifest.at("tp")==world,"Artifact format or TP mismatch");
  config_=manifest.at("config"); text_=config_.at("text_config"); vision_=config_.at("vision_config");
  TORCH_CHECK(config_.at("model_type")=="qwen3_5","Only qwen3_5 dense supported");
  TORCH_CHECK(text_.value("attention_bias",false)==false,"Attention bias unsupported");
  TORCH_CHECK(vision_.value("deepstack_visual_indexes",json::array()).empty(),"Deepstack not supported");
  TORCH_CHECK(text_.at("rope_parameters").value("rope_type",std::string("default"))=="default","Only default RoPE supported");
  eps_=text_.at("rms_norm_eps"); states_.resize(text_.at("num_hidden_layers").get<int>());
  auto tensors=manifest.at("ranks").at(rank).at("tensors");
  size_t loaded=0;
  for(auto item=tensors.begin();item!=tensors.end();++item) {
    Weight w; w.data=raw(dir,item.value(),device);
    if(item.value().contains("scale")) w.scale=raw(dir,item.value().at("scale"),device);
    weights_.emplace(item.key(),w);
    if(rank==0 && ++loaded%100==0) std::cerr<<"Loaded "<<loaded<<" tensors\n";
  }
  decode_offset_=at::zeros({1},at::TensorOptions().device(at::Device(at::kCUDA,device)).dtype(at::kLong));
  if(options_.optimized) {
    for(size_t i=0;i<states_.size();i++) {
      auto p="model.language_model.layers."+std::to_string(i);
      fuse_weights(p+".mlp.gate_up",{p+".mlp.gate_proj",p+".mlp.up_proj"});
      if(text_.at("layer_types").at(i)=="full_attention") fuse_weights(p+".self_attn.qkv_gate",{p+".self_attn.q_proj",p+".self_attn.k_proj",p+".self_attn.v_proj"});
      else fuse_weights(p+".linear_attn.in_proj_all",{p+".linear_attn.in_proj_qkv",p+".linear_attn.in_proj_z",p+".linear_attn.in_proj_b",p+".linear_attn.in_proj_a"});
    }
  }
  TORCH_CHECK(!options_.cuda_graph || options_.optimized,"CUDA Graph requires optimized mode");
  C10_CUDA_CHECK(cudaDeviceSynchronize());
}
void Engine::fuse_weights(const std::string& dest,const std::vector<std::string>& source) {
  std::vector<Tensor> data,scales;
  for(auto& name:source) {auto& w=weights_.at(name+".weight");data.push_back(w.data);if(w.scale.defined())scales.push_back(w.scale);}
  TORCH_CHECK(scales.empty()||scales.size()==data.size(),"Cannot fuse mixed representations");
  Weight combined;combined.data=at::cat(data,0);if(!scales.empty())combined.scale=at::cat(scales,0);
  weights_.emplace(dest+".weight",combined);for(auto& name:source)weights_.erase(name+".weight");
}
Tensor Engine::tensor(const std::string& name) {
  auto it=weights_.find(name); TORCH_CHECK(it!=weights_.end(),"Missing tensor ",name);
  TORCH_CHECK(!it->second.scale.defined(),"Expected unquantized tensor ",name); return it->second.data;
}
Tensor Engine::read_input(const std::string& dir,const json& d) { return raw(dir,d,device_); }
Tensor Engine::sum(Tensor x) {
  if(world_==1) return x;
  auto f=x.to(at::kFloat).contiguous();
  auto r=ncclAllReduce(f.data_ptr(),f.data_ptr(),f.numel(),ncclFloat,ncclSum,comm_,at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r==ncclSuccess,ncclGetErrorString(r)); return f.to(x.scalar_type());
}
Tensor Engine::linear(Tensor x,const std::string& name,bool reduce) {
  auto it=weights_.find(name+".weight"); TORCH_CHECK(it!=weights_.end(),"Missing linear ",name);
  auto& w=it->second; Tensor y;
  TORCH_CHECK(w.data.dim()>=2,"Invalid matrix ",name);
  if(w.scale.defined() && options_.optimized) {
    y=fp8_linear(x.reshape({-1,x.size(-1)}).contiguous(),w.data,w.scale);
    auto shape=x.sizes().vec();shape.back()=w.data.size(0);y=y.reshape(shape);
  } else if(w.scale.defined()) {
    TORCH_CHECK(w.data.dim()==2 && x.size(-1)==w.data.size(1),"FP8 dimensions mismatch ",name);
    auto shape=x.sizes().vec(); shape.back()=w.data.size(0); y=at::empty(shape,x.options());
    // Bound BF16 temporary to 2048 output rows; full model remains FP8 resident.
    for(int64_t begin=0;begin<w.data.size(0);begin+=2048) {
      auto count=std::min<int64_t>(2048,w.data.size(0)-begin);
      auto decoded=fp8_decode(w.data.narrow(0,begin,count),w.scale.narrow(0,begin,count));
      y.narrow(-1,begin,count).copy_(at::matmul(x,decoded.t()));
    }
  } else {
    auto matrix=w.data.reshape({w.data.size(0),-1});
    y=at::matmul(x,matrix.t());
  }
  if(reduce) y=sum(y);
  auto b=weights_.find(name+".bias"); if(b!=weights_.end()) y=y+b->second.data;
  return y;
}
Tensor Engine::norm(Tensor x,const std::string& name,bool one_center) {
  if(options_.optimized) return fused_rms(x,tensor(name+".weight"),eps_,one_center);
  auto f=x.to(at::kFloat); auto w=tensor(name+".weight").to(at::kFloat);
  auto normalized=f*at::rsqrt((f*f).mean(-1,true)+eps_);
  if(one_center) return (normalized*(w+1)).to(x.scalar_type());
  return (normalized.to(x.scalar_type())*w.to(x.scalar_type())).to(x.scalar_type());
}
Tensor Engine::layer_norm(Tensor x,const std::string& name) {
  return at::layer_norm(x,{x.size(-1)},tensor(name+".weight"),tensor(name+".bias"),1e-6);
}
static Tensor rotate(Tensor x,Tensor cos,Tensor sin,int d) {
  auto a=x.narrow(-1,0,d), pass=x.narrow(-1,d,x.size(-1)-d);
  auto swapped=at::cat({-a.narrow(-1,d/2,d/2),a.narrow(-1,0,d/2)},-1);
  return at::cat({a*cos+swapped*sin,pass},-1);
}
Tensor Engine::embed(Tensor ids) {
  auto& w=weights_.at("model.language_model.embed_tokens.weight");
  auto selected=w.data.index_select(0,ids.reshape({-1}));
  return w.scale.defined()?fp8_decode(selected.contiguous(),w.scale.index_select(0,ids.reshape({-1})).contiguous()):selected;
}
Tensor Engine::logits(Tensor x) { return linear(norm(x,"model.language_model.norm"),"lm_head").to(at::kFloat); }
Tensor Engine::full_attention(Tensor x,Tensor positions,int layer,const std::string& p,Tensor projected,bool finish) {
  auto T=x.size(0); int H=text_.at("num_attention_heads").get<int>()/world_;
  int HK=text_.at("num_key_value_heads").get<int>()/world_, D=text_.at("head_dim");
  if(options_.optimized && !projected.defined())projected=linear(x,p+".qkv_gate");
  auto qg=(options_.optimized?projected.narrow(-1,0,H*2*D):linear(x,p+".q_proj")).reshape({T,H,2*D});
  auto q=norm(qg.narrow(-1,0,D),p+".q_norm"), gate=qg.narrow(-1,D,D).reshape({T,H*D});
  auto k=norm((options_.optimized?projected.narrow(-1,H*2*D,HK*D):linear(x,p+".k_proj")).reshape({T,HK,D}),p+".k_norm");
  auto v=(options_.optimized?projected.narrow(-1,H*2*D+HK*D,HK*D):linear(x,p+".v_proj")).reshape({T,HK,D});
  auto rope=text_.at("rope_parameters"); int rd=int(D*rope.value("partial_rotary_factor",0.25));
  auto sections=rope.at("mrope_section").get<std::vector<int>>();
  double theta=rope.at("rope_theta");
  if(options_.optimized) {
    q=fused_rope(q,positions,rd,theta,sections[1],sections[2]);
    k=fused_rope(k,positions,rd,theta,sections[1],sections[2]);
  } else {
  std::vector<Tensor> freqs;
  for(int j=0;j<rd/2;j++) {
    int axis=0; if(j%3==1 && j<sections[1]*3) axis=1; if(j%3==2 && j<sections[2]*3) axis=2;
    freqs.push_back(positions.select(0,axis).to(at::kFloat)*std::pow(theta,-2.0*j/rd));
  }
  auto freq=at::stack(freqs,-1); auto angle=at::cat({freq,freq},-1);
  auto cos=angle.cos().to(x.scalar_type()).unsqueeze(1), sin=angle.sin().to(x.scalar_type()).unsqueeze(1);
  q=rotate(q,cos,sin,rd); k=rotate(k,cos,sin,rd);
  }
  auto& state=states_.at(layer);
  TORCH_CHECK(decode_mode_ || state.length+T<=session_capacity(),"KV capacity exceeded");
  if(!state.key.defined()) { state.key=at::empty({session_capacity(),HK,D},x.options()); state.value=at::empty_like(state.key); }
  if(decode_mode_) {
    auto out=gqa_decode(q,k,v,state.key,state.value,decode_offset_).reshape({T,H*D})*gate.sigmoid();
    return finish?linear(out,p+".o_proj",true):out;
  }
  int old=state.length; state.key.narrow(0,old,T).copy_(k); state.value.narrow(0,old,T).copy_(v); state.length+=T;
  auto keys=state.key.narrow(0,0,state.length).repeat_interleave(H/HK,1).transpose(0,1).unsqueeze(0);
  auto values=state.value.narrow(0,0,state.length).repeat_interleave(H/HK,1).transpose(0,1).unsqueeze(0);
  auto qp=q.transpose(0,1).unsqueeze(0);
  // Explicit offset-aware mask: causal=true alone is incorrect for cached chunks.
  auto ints=x.options().dtype(at::kLong);
  auto mask=at::arange(state.length,ints).unsqueeze(0)<=at::arange(old,state.length,ints).unsqueeze(1);
  auto out=at::scaled_dot_product_attention(qp,keys,values,mask,0.0,false).squeeze(0).transpose(0,1).reshape({T,H*D});
  out=out*gate.sigmoid();return finish?linear(out,p+".o_proj",true):out;
}
Tensor Engine::delta_attention(Tensor x,int layer,const std::string& p,Tensor projected,bool finish) {
  int K=text_.at("linear_key_head_dim"), V=text_.at("linear_value_head_dim");
  int HK=text_.at("linear_num_key_heads").get<int>()/world_, H=text_.at("linear_num_value_heads").get<int>()/world_;
  int kernel=text_.at("linear_conv_kernel_dim"); auto T=x.size(0); auto& s=states_.at(layer);
  int C=2*HK*K+H*V;
  if(options_.optimized && !projected.defined())projected=linear(x,p+".in_proj_all");
  auto mixed=(options_.optimized?projected.narrow(-1,0,C):linear(x,p+".in_proj_qkv")).t().unsqueeze(0);
  if(!s.conv.defined()) {
    s.conv=at::zeros({1,C,kernel-1},x.options());
    s.recurrent=at::zeros({H,K,V},x.options().dtype(at::kFloat));
  }
  Tensor conv;
  if(decode_mode_) conv=conv_decode(mixed.reshape({1,C}),tensor(p+".conv1d.weight"),s.conv);
  else {
    auto input=at::cat({s.conv,mixed},-1);
    s.conv.copy_(input.narrow(-1,input.size(-1)-(kernel-1),kernel-1));
    conv=at::silu(at::conv1d(input,tensor(p+".conv1d.weight"),{},at::IntArrayRef{1},at::IntArrayRef{0},at::IntArrayRef{1},C)).squeeze(0).t();
  }
  auto q=conv.narrow(-1,0,HK*K).reshape({T,HK,K});
  auto k=conv.narrow(-1,HK*K,HK*K).reshape({T,HK,K});
  auto v=conv.narrow(-1,2*HK*K,H*V).reshape({T,H,V});
  // Match reference normalization and rounding before the FP32 recurrence.
  if(options_.optimized){q=fused_l2(q).repeat_interleave(H/HK,1).contiguous();k=fused_l2(k).repeat_interleave(H/HK,1).contiguous();}
  else {
  q=(q.to(at::kFloat)*at::rsqrt(q.to(at::kFloat).square().sum(-1,true)+1e-6)).to(at::kBFloat16).repeat_interleave(H/HK,1).contiguous();
  k=(k.to(at::kFloat)*at::rsqrt(k.to(at::kFloat).square().sum(-1,true)+1e-6)).to(at::kBFloat16).repeat_interleave(H/HK,1).contiguous();
  }
  auto a=(options_.optimized?projected.narrow(-1,C+H*V+H,H):linear(x,p+".in_proj_a")).to(at::kFloat);
  auto g=-tensor(p+".A_log").to(at::kFloat).exp()*at::softplus(a+tensor(p+".dt_bias").to(at::kFloat));
  auto beta=(options_.optimized?projected.narrow(-1,C+H*V,H):linear(x,p+".in_proj_b")).sigmoid().to(at::kFloat);
  auto result=(options_.optimized&&(K==128||K==16))?delta_scan_fast(q,k,v.contiguous(),g.contiguous(),beta.contiguous(),s.recurrent):delta_scan(q,k,v.contiguous(),g.contiguous(),beta.contiguous(),s.recurrent);
  auto z=(options_.optimized?projected.narrow(-1,C,H*V):linear(x,p+".in_proj_z")).reshape({T,H,V});
  result=(norm(result,p+".norm",false).to(at::kFloat)*at::silu(z.to(at::kFloat))).to(at::kBFloat16);
  result=result.reshape({T,H*V});return finish?linear(result,p+".out_proj",true):result;
}
Tensor Engine::step(Tensor x,Tensor positions) {
  TORCH_CHECK(x.dim()==2 && positions.dim()==2 && positions.size(0)==3 && positions.size(1)==x.size(0),"Invalid text input shape");
  for(size_t i=0;i<states_.size();i++) {
    std::string p="model.language_model.layers."+std::to_string(i);
    auto n=norm(x,p+".input_layernorm");
    if(text_.at("layer_types").at(i)=="full_attention") x=x+full_attention(n,positions,i,p+".self_attn");
    else x=x+delta_attention(n,i,p+".linear_attn");
    n=norm(x,p+".post_attention_layernorm");
    auto gated=options_.optimized?fused_swiglu(linear(n,p+".mlp.gate_up")):at::silu(linear(n,p+".mlp.gate_proj"))*linear(n,p+".mlp.up_proj");
    x=x+linear(gated,p+".mlp.down_proj",true);
  }
  return x;
}
Tensor Engine::vision(const std::string& dir,const json& request) {
  if(!request.contains("images") || request.at("images").empty()) return {};
  std::vector<Tensor> outputs; int H=vision_.at("num_heads"), hidden=vision_.at("hidden_size"), D=hidden/H;
  int merge=vision_.at("spatial_merge_size");
  for(auto& image:request.at("images")) {
    auto cache_key=options_.image_cache_bytes?request_hash(dir,image):std::string();
    auto cached=images_.find(cache_key);
    if(cached!=images_.end()) {cached->second.used=++cache_clock_;++image_hits_;outputs.push_back(cached->second.tensor);continue;}

    auto patches=read_input(dir,image.at("patches")).to(at::kBFloat16); auto T=patches.size(0);
    auto x=linear(patches,"model.visual.patch_embed.proj");
    auto indices=read_input(dir,image.at("position_indices"));
    auto factors=read_input(dir,image.at("position_weights")).to(at::kBFloat16);
    Tensor pos;
    for(int i=0;i<4;i++) {
      auto term=tensor("model.visual.pos_embed.weight").index_select(0,indices.select(0,i))*factors.select(0,i).unsqueeze(-1);
      pos=i==0?term:pos+term;
    }
    x=x+pos;
    auto coords=read_input(dir,image.at("coords")).to(at::kFloat);
    std::vector<Tensor> frequencies;
    for(int axis=0;axis<2;axis++) for(int j=0;j<D/4;j++) frequencies.push_back(coords.select(1,axis)*std::pow(10000.0,-4.0*j/D));
    auto f=at::stack(frequencies,-1); auto angles=at::cat({f,f},-1).unsqueeze(1);
    auto cos=angles.cos(),sin=angles.sin();
    for(int i=0;i<vision_.at("depth").get<int>();i++) {
      auto p="model.visual.blocks."+std::to_string(i); auto n=layer_norm(x,p+".norm1");
      auto qkv=linear(n,p+".attn.qkv").reshape({T,3,H,D});
      auto q=rotate(qkv.select(1,0).to(at::kFloat),cos,sin,D).to(at::kBFloat16);
      auto k=rotate(qkv.select(1,1).to(at::kFloat),cos,sin,D).to(at::kBFloat16);
      auto v=qkv.select(1,2);
      auto out=at::scaled_dot_product_attention(q.transpose(0,1).unsqueeze(0),k.transpose(0,1).unsqueeze(0),v.transpose(0,1).unsqueeze(0),{},0.0,false);
      x=x+linear(out.squeeze(0).transpose(0,1).reshape({T,hidden}),p+".attn.proj");
      n=layer_norm(x,p+".norm2"); x=x+linear(at::gelu(linear(n,p+".mlp.linear_fc1"),"tanh"),p+".mlp.linear_fc2");
    }
    x=layer_norm(x,"model.visual.merger.norm").reshape({-1,hidden*merge*merge});
    x=linear(at::gelu(linear(x,"model.visual.merger.linear_fc1"),"none"),"model.visual.merger.linear_fc2");
    if(options_.image_cache_bytes && x.nbytes()<=options_.image_cache_bytes) {
      while(image_bytes_+x.nbytes()>options_.image_cache_bytes && !images_.empty()) {
        auto oldest=std::min_element(images_.begin(),images_.end(),[](const auto& a,const auto& b){return a.second.used<b.second.used;});
        image_bytes_-=oldest->second.bytes;images_.erase(oldest);
      }
      images_.emplace(cache_key,Image{x,x.nbytes(),++cache_clock_});image_bytes_+=x.nbytes();
    }
    outputs.push_back(x);
  }
  return at::cat(outputs,0);
}

struct DecodeGraph {std::unique_ptr<at::cuda::CUDAGraph> graph;Tensor ids,positions,output;};
Engine::~Engine()=default;
int Engine::session_capacity() const {
    auto it=session_capacities_.find(active_);return it==session_capacities_.end()?capacity_:it->second;
  }
  void Engine::reserve_session(int id,int tokens) {
    TORCH_CHECK(tokens>0&&tokens<=capacity_&&!session_capacities_.count(id),"Invalid session reservation");
    session_capacities_[id]=tokens;
  }
  void Engine::activate(int id) {
  TORCH_CHECK(id>=0,"Invalid session ID");if(id==active_)return;
  bool populated=std::any_of(states_.begin(),states_.end(),[](const State& s){return s.key.defined()||s.conv.defined();});
    if(populated)sessions_[active_]=std::move(states_);else sessions_.erase(active_);auto it=sessions_.find(id);
  if(it==sessions_.end())states_=std::vector<State>(text_.at("num_hidden_layers").get<int>());
  else {states_=std::move(it->second);sessions_.erase(it);}active_=id;
}
void Engine::drop(int id) {
  C10_CUDA_CHECK(cudaDeviceSynchronize());graphs_.erase(id);session_capacities_.erase(id);
  if(id==active_)states_=std::vector<State>(text_.at("num_hidden_layers").get<int>());else sessions_.erase(id);
}
Tensor Engine::decode(int64_t token,int64_t position,int consumed) {
  TORCH_CHECK(consumed>=0 && consumed<session_capacity(),"Decode capacity exceeded");
  auto longs=decode_offset_.options();
  if(!options_.optimized) return logits(step(embed(at::full({1},token,longs)),at::full({3,1},position,longs)));
  decode_mode_=true;decode_offset_.fill_(consumed);
  Tensor result;
  if(!options_.cuda_graph)result=logits(step(embed(at::full({1},token,longs)),at::full({3,1},position,longs)));
  else {
    auto& holder=graphs_[active_];if(!holder)holder=std::make_shared<DecodeGraph>();auto& graph=*holder;
    if(!graph.graph) {
      graph.ids=at::full({1},token,longs);graph.positions=at::full({3,1},position,longs);
      // Warmup mutates recurrence/conv. Snapshot and restore them before capture/replay.
      std::vector<std::pair<Tensor,Tensor>> saved;
      for(auto& st:states_)saved.emplace_back(st.conv.defined()?st.conv.clone():Tensor(),st.recurrent.defined()?st.recurrent.clone():Tensor());
      auto restore=[&](){for(size_t i=0;i<states_.size();i++)if(saved[i].first.defined()){states_[i].conv.copy_(saved[i].first);states_[i].recurrent.copy_(saved[i].second);}};
      C10_CUDA_CHECK(cudaDeviceSynchronize());auto stream=c10::cuda::getStreamFromPool(false,device_);
      {
        c10::cuda::CUDAStreamGuard guard(stream);
        for(int i=0;i<2;i++){graph.output=logits(step(embed(graph.ids),graph.positions));restore();}
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        graph.graph=std::make_unique<at::cuda::CUDAGraph>();graph.graph->capture_begin();
        graph.output=logits(step(embed(graph.ids),graph.positions));graph.graph->capture_end();
      }
    } else {graph.ids.fill_(token);graph.positions.fill_(position);}
    graph.graph->replay();result=graph.output;
  }
  decode_mode_=false;for(auto& st:states_)if(st.key.defined())st.length=consumed+1;return result;
}
Tensor Engine::decode_batch(const std::vector<int>& sessions,const std::vector<int64_t>& tokens,const std::vector<int64_t>& positions,const std::vector<int>& consumed) {
  size_t B=sessions.size();TORCH_CHECK(B>0&&tokens.size()==B&&positions.size()==B&&consumed.size()==B,"Batch metadata mismatch");
  if(B==1){activate(sessions[0]);return decode(tokens[0],positions[0],consumed[0]);}
  if(!options_.optimized) {std::vector<Tensor> out;for(size_t i=0;i<B;i++){activate(sessions[i]);out.push_back(decode(tokens[i],positions[i],consumed[i]));}return at::cat(out);}
  for(size_t i=0;i<B;i++){activate(sessions[i]);TORCH_CHECK(consumed[i]>=0&&consumed[i]<session_capacity(),"Batch exceeds capacity");}
  auto ids=at::empty({int64_t(B)},at::TensorOptions().dtype(at::kLong));std::copy(tokens.begin(),tokens.end(),ids.data_ptr<int64_t>());ids=ids.to(decode_offset_.device());
  auto x=embed(ids);decode_mode_=true;
  // Projection/MLP GEMMs and all-reduces are batched; each request retains independent KV/GDN state.
  for(int layer=0;layer<text_.at("num_hidden_layers").get<int>();layer++) {
    std::string p="model.language_model.layers."+std::to_string(layer);bool full=text_.at("layer_types").at(layer)=="full_attention";
    auto n=norm(x,p+".input_layernorm");auto projected=linear(n,p+(full?".self_attn.qkv_gate":".linear_attn.in_proj_all"));
    std::vector<Tensor> local;
    for(size_t i=0;i<B;i++) {
      activate(sessions[i]);decode_offset_.fill_(consumed[i]);auto row=n.narrow(0,i,1),proj=projected.narrow(0,i,1);
      local.push_back(full?full_attention(row,at::full({3,1},positions[i],decode_offset_.options()),layer,p+".self_attn",proj,false):delta_attention(row,layer,p+".linear_attn",proj,false));
    }
    x=x+linear(at::cat(local,0),p+(full?".self_attn.o_proj":".linear_attn.out_proj"),true);
    n=norm(x,p+".post_attention_layernorm");x=x+linear(fused_swiglu(linear(n,p+".mlp.gate_up")),p+".mlp.down_proj",true);
  }
  decode_mode_=false;for(size_t i=0;i<B;i++){activate(sessions[i]);for(auto& st:states_)if(st.key.defined())st.length=consumed[i]+1;}
  return logits(x);
}
void Engine::demote_prefix(const std::string& key) {
    auto it=prefixes_.find(key);TORCH_CHECK(it!=prefixes_.end(),"Missing checkpoint");
    auto& entry=it->second;
    if(options_.host_prefix_cache_bytes && entry.bytes<=options_.host_prefix_cache_bytes) {
      while(host_prefix_bytes_+entry.bytes>options_.host_prefix_cache_bytes&&!host_prefixes_.empty()) {
        auto old=std::min_element(host_prefixes_.begin(),host_prefixes_.end(),[](const auto& a,const auto& b){return a.second.used<b.second.used;});
        host_prefix_bytes_-=old->second.bytes;host_prefixes_.erase(old);
      }
      auto host=[](const Tensor& x){if(!x.defined())return Tensor();auto y=at::empty(x.sizes(),x.options().device(at::kCPU).pinned_memory(true));y.copy_(x,false);return y;};
      Prefix copy;copy.bytes=entry.bytes;copy.used=entry.used;copy.logits=host(entry.logits);
      for(auto& st:entry.states){State dst;dst.length=st.length;dst.key=host(st.key);dst.value=host(st.value);dst.conv=host(st.conv);dst.recurrent=host(st.recurrent);copy.states.push_back(std::move(dst));}
      host_prefix_bytes_+=copy.bytes;host_prefixes_[key]=std::move(copy);
    }
    prefix_bytes_-=entry.bytes;prefixes_.erase(it);
  }
  void Engine::save_prefix(const std::string& key,Tensor output) {
  if(!options_.prefix_cache_bytes || (prefixes_.count(key)||host_prefixes_.count(key)))return;
  size_t bytes=output.nbytes();for(auto& st:states_){if(st.key.defined())bytes+=2*st.length*st.key.size(1)*st.key.size(2)*st.key.element_size();if(st.conv.defined())bytes+=st.conv.nbytes()+st.recurrent.nbytes();}
  if(bytes>options_.prefix_cache_bytes)return;
  while(prefix_bytes_+bytes>options_.prefix_cache_bytes&&!prefixes_.empty()) {
    auto oldest=std::min_element(prefixes_.begin(),prefixes_.end(),[](const auto& a,const auto& b){return a.second.used<b.second.used;});auto victim=oldest->first;demote_prefix(victim);
  }
  Prefix entry;entry.bytes=bytes;entry.used=++cache_clock_;entry.logits=output.clone();
  for(auto& st:states_) {State copy;copy.length=st.length;if(st.key.defined()){copy.key=st.key.narrow(0,0,st.length).clone();copy.value=st.value.narrow(0,0,st.length).clone();}if(st.conv.defined()){copy.conv=st.conv.clone();copy.recurrent=st.recurrent.clone();}entry.states.push_back(copy);}
  prefixes_.emplace(key,std::move(entry));prefix_bytes_+=bytes;
}
Tensor Engine::restore_prefix(const std::string& key) {
  auto it=prefixes_.find(key);auto host=host_prefixes_.find(key);if(it==prefixes_.end()&&host==host_prefixes_.end())return {};bool from_host=it==prefixes_.end();auto& entry=from_host?host->second:it->second;entry.used=++cache_clock_;++prefix_hits_;if(from_host)++host_prefix_hits_;
  int reserved=session_capacity();drop(active_);session_capacities_[active_]=reserved;for(size_t i=0;i<states_.size();i++){auto& src=entry.states[i];auto& dst=states_[i];dst.length=src.length;
    if(src.key.defined()){dst.key=at::empty({session_capacity(),src.key.size(1),src.key.size(2)},src.key.options().device(at::Device(at::kCUDA,device_)).pinned_memory(false));dst.value=at::empty_like(dst.key);dst.key.narrow(0,0,src.length).copy_(src.key);dst.value.narrow(0,0,src.length).copy_(src.value);}
    if(src.conv.defined()){dst.conv=src.conv.to(at::Device(at::kCUDA,device_),src.conv.scalar_type(),false,true);dst.recurrent=src.recurrent.to(at::Device(at::kCUDA,device_),src.recurrent.scalar_type(),false,true);}}
  return entry.logits.to(at::Device(at::kCUDA,device_),entry.logits.scalar_type(),false,true);
}
json Engine::cache_stats() const {return {{"image_bytes",image_bytes_},{"prefix_bytes",prefix_bytes_},{"image_hits",image_hits_},{"prefix_hits",prefix_hits_},{"host_prefix_bytes",host_prefix_bytes_},{"host_prefix_hits",host_prefix_hits_}};}
std::string request_hash(const std::string& dir,const json& request) {
  std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),EVP_MD_CTX_free);
  TORCH_CHECK(context&&EVP_DigestInit_ex(context.get(),EVP_sha256(),nullptr)==1,"SHA256 initialization failed");
  auto metadata=request.dump();EVP_DigestUpdate(context.get(),metadata.data(),metadata.size());
  auto root=std::filesystem::weakly_canonical(dir);
  std::function<void(const json&)> walk=[&](const json& j){
    if(j.is_object()&&j.contains("file")) {
      auto file=std::filesystem::weakly_canonical(root/j.at("file").get<std::string>());
      auto relative=file.lexically_relative(root);TORCH_CHECK(!relative.empty()&&!relative.is_absolute(),"Invalid cache input path");for(auto& part:relative)TORCH_CHECK(part!="..","Cache path escape");
      std::ifstream f(file,std::ios::binary);TORCH_CHECK(f.good(),"Missing cache input ",file.string());char buffer[65536];
      while(f){f.read(buffer,sizeof(buffer));if(f.gcount())EVP_DigestUpdate(context.get(),buffer,f.gcount());}TORCH_CHECK(f.eof(),"Cache input read failed");
    }
    if(j.is_structured())for(auto& value:j)walk(value);
  };walk(request);unsigned char digest[EVP_MAX_MD_SIZE];unsigned int length;TORCH_CHECK(EVP_DigestFinal_ex(context.get(),digest,&length)==1,"SHA256 finalization failed");
  std::ostringstream output;for(unsigned int i=0;i<length;i++)output<<std::hex<<std::setw(2)<<std::setfill('0')<<int(digest[i]);return output.str();
}
}
