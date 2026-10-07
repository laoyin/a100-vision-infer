#pragma once
#include <cstdlib>
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define AVI_HAS_NVTX 1
#else
#define AVI_HAS_NVTX 0
#endif
namespace avi {
inline bool nvtx_enabled(){
 static bool enabled=[](){auto v=std::getenv("AVI_NVTX");return v&&v[0]=='1';}();
 return AVI_HAS_NVTX&&enabled;
}
class TraceRange {
 bool active_;
 public:
 explicit TraceRange(const char* name):active_(nvtx_enabled()){
#if AVI_HAS_NVTX
  if(active_)nvtxRangePushA(name);
#endif
 }
 ~TraceRange(){
#if AVI_HAS_NVTX
  if(active_)nvtxRangePop();
#endif
 }
 TraceRange(const TraceRange&)=delete;
 TraceRange& operator=(const TraceRange&)=delete;
};
}
