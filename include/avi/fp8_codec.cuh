#pragma once
#include <cuda_runtime.h>
namespace avi {
// Exact E4M3FN -> FP32 conversion on SM80; no exponentiation instructions.
__device__ __forceinline__ float fp8_e4m3_value(unsigned char code) {
 unsigned e=(code>>3)&15,m=code&7,sign=(unsigned(code)&128)<<24;
 if(e==15&&m==7)return __uint_as_float(sign|0x7fc00000u);
 if(e)return __uint_as_float(sign|((e+120)<<23)|(m<<20));
 return __uint_as_float(sign|__float_as_uint(float(m)*0.001953125f));
}
}
