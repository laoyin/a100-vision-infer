#include "avi/tilelang.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <dlfcn.h>
#include <openssl/evp.h>
#include <array>
#include <iomanip>
#include <sstream>
#include <memory>
namespace avi {
namespace {
using GdnCall=int(*)(void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,int,cudaStream_t);
using LinearCall=int(*)(void*,void*,void*,void*,int,int,cudaStream_t);
struct Library {
 void* handle=nullptr;void* call=nullptr;const char*(*error)()=nullptr;
 // Keep modules alive until process exit: captured graphs retain CUDA code.
};
std::map<std::string,Library> modules;std::string configured;std::mutex setup;
std::string gdn_key(int c,int hq,int h){return "gdn:"+std::to_string(c)+":"+std::to_string(hq)+":"+std::to_string(h);}
std::string linear_key(int m,int split){return "fp8:"+std::to_string(m)+":"+std::to_string(split);}
void* symbol(void* library,const char* name){
 dlerror();auto result=dlsym(library,name);auto err=dlerror();
 TORCH_CHECK(!err&&result,"TileLang export missing ",name,": ",err?err:"null symbol");return result;
}
std::string digest(const std::filesystem::path& path){
 std::ifstream input(path,std::ios::binary);TORCH_CHECK(input.good(),"Cannot read TileLang library ",path.string());
 std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),EVP_MD_CTX_free);
 TORCH_CHECK(ctx&&EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)==1,"Cannot initialize SHA256");
 std::array<char,65536> block;
 while(input){input.read(block.data(),block.size());auto n=input.gcount();
  if(n)TORCH_CHECK(EVP_DigestUpdate(ctx.get(),block.data(),n)==1,"SHA256 update failed");}
 TORCH_CHECK(input.eof(),"Failed to read TileLang library");
 unsigned char hash[EVP_MAX_MD_SIZE];unsigned n=0;
 TORCH_CHECK(EVP_DigestFinal_ex(ctx.get(),hash,&n)==1&&n==32,"SHA256 final failed");
 std::ostringstream out;for(unsigned i=0;i<n;i++)out<<std::hex<<std::setw(2)<<std::setfill('0')<<int(hash[i]);
 return out.str();
}
Library& require(const std::string& key){
 auto it=modules.find(key);TORCH_CHECK(it!=modules.end(),"Missing validated TileLang kernel ",key,
  ". Run tools/export_tilelang.py; unsupported shapes are not silently substituted.");
 return it->second;
}
}
bool tilelang_configured(){return !configured.empty();}
void configure_tilelang(const std::string& directory){
 TORCH_CHECK(!directory.empty(),"--tilelang-dir must name an exported kernel directory");
 std::lock_guard<std::mutex> lock(setup);auto dir=std::filesystem::canonical(directory);
 if(!configured.empty()){TORCH_CHECK(configured==dir.string(),"Only one TileLang export directory per worker");return;}
 cudaDeviceProp prop;int device;C10_CUDA_CHECK(cudaGetDevice(&device));C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
 TORCH_CHECK(prop.major==8&&prop.minor==0,"TileLang export requires SM80");
 TORCH_CHECK(!std::filesystem::exists(dir/"INCOMPLETE"),"Incomplete TileLang export: ",directory);
 std::ifstream input(dir/"manifest.json");TORCH_CHECK(input.good(),"Missing TileLang manifest: ",directory);
 nlohmann::json manifest;input>>manifest;
 TORCH_CHECK(manifest.at("format")=="avi-tilelang-v1"&&manifest.at("arch")=="sm_80","Invalid TileLang export format/architecture");
 TORCH_CHECK(!manifest.at("kernels").empty(),"Empty TileLang export");
 std::map<std::string,Library> loaded;
 for(const auto& row:manifest.at("kernels")){
  auto path=std::filesystem::path(row.at("file").get<std::string>());
  TORCH_CHECK(!path.is_absolute()&&path.filename()==path&&path.extension()==".so","Invalid TileLang library path");
  auto kind=row.at("kind").get<std::string>();std::string key;
  if(kind=="gdn"){
   TORCH_CHECK(row.at("abi")=="Q,K,V,G,B,A,W,U,SQ,WK,last,blocks:i32,stream","Unexpected GDN export ABI");
   int c=row.at("chunk"),hq=row.at("key_heads"),h=row.at("heads");
   TORCH_CHECK((c==32||c==64)&&hq>0&&h>=hq&&h%hq==0,"Invalid GDN export geometry");key=gdn_key(c,hq,h);
  }else{
   TORCH_CHECK(kind=="fp8"&&row.at("abi")=="X,Codes,Scales,Partial,K:i32,N:i32,stream","Unexpected FP8 export ABI");
   int m=row.at("rows"),split=row.at("split");
   TORCH_CHECK(m>=2&&m<=8&&(split==1||split==4),"Invalid FP8 export geometry");key=linear_key(m,split);
  }
  TORCH_CHECK(row.at("validated").get<bool>()&&!loaded.count(key),"Unvalidated or duplicate TileLang kernel ",key);
  TORCH_CHECK(digest(dir/path)==row.at("sha256").get<std::string>(),"TileLang library checksum mismatch: ",path.string());
  Library lib;lib.handle=dlopen((dir/path).c_str(),RTLD_NOW|RTLD_LOCAL);
  if(!lib.handle){auto err=dlerror();TORCH_CHECK(false,"TileLang dlopen failed: ",path.string(),": ",err?err:"unknown");}
  lib.call=symbol(lib.handle,"call");lib.error=reinterpret_cast<const char*(*)()>(symbol(lib.handle,"get_last_error"));
  auto init=reinterpret_cast<int(*)()>(symbol(lib.handle,"init"));
  TORCH_CHECK(init()==0,"TileLang initialization failed: ",lib.error());loaded.emplace(key,lib);
 }
 modules=std::move(loaded);configured=dir.string();
}
void tilelang_gdn_prepare(const at::Tensor& Q,const at::Tensor& K,const at::Tensor& V,
 const at::Tensor& G,const at::Tensor& B,at::Tensor& A,at::Tensor& W,at::Tensor& U,
 at::Tensor& SQ,at::Tensor& WK,at::Tensor& last,int chunk,int blocks,int H,int HQ,cudaStream_t stream){
 auto& module=require(gdn_key(chunk,HQ,H));auto call=reinterpret_cast<GdnCall>(module.call);
 int result=call(Q.data_ptr(),K.data_ptr(),V.data_ptr(),G.data_ptr(),B.data_ptr(),A.data_ptr(),
 W.data_ptr(),U.data_ptr(),SQ.data_ptr(),WK.data_ptr(),last.data_ptr(),blocks,stream);
 TORCH_CHECK(result==0,"TileLang GDN launch failed: ",module.error());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
at::Tensor tilelang_fp8_small(at::Tensor x,at::Tensor codes,at::Tensor scales,int split){
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&x.dim()==2&&x.is_contiguous()&&
  codes.device()==x.device()&&codes.scalar_type()==at::kByte&&codes.dim()==2&&codes.is_contiguous()&&
  scales.device()==x.device()&&scales.scalar_type()==at::kFloat&&scales.dim()==2&&scales.is_contiguous()&&
  x.size(0)>=2&&x.size(0)<=8&&x.size(1)>0&&x.size(1)==codes.size(1)&&x.size(1)%128==0&&
  codes.size(0)>0&&codes.size(0)%16==0&&scales.size(0)==codes.size(0)&&scales.size(1)==x.size(1)/128&&
  (split==1||split==4),"Invalid TileLang FP8 W8A16 inputs");
 c10::cuda::CUDAGuard guard(x.device());int M=x.size(0),K=x.size(1),N=codes.size(0);
 auto partial=at::empty({split,M,N},x.options().dtype(at::kFloat));
 auto& module=require(linear_key(M,split));auto call=reinterpret_cast<LinearCall>(module.call);
 int result=call(x.data_ptr(),codes.data_ptr(),scales.data_ptr(),partial.data_ptr(),K,N,at::cuda::getCurrentCUDAStream());
 TORCH_CHECK(result==0,"TileLang FP8 launch failed: ",module.error());C10_CUDA_KERNEL_LAUNCH_CHECK();
 return (split==1?partial.select(0,0):partial.sum(0)).to(at::kBFloat16);
}
}
