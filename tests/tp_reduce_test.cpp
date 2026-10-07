#include <ATen/ATen.h>
#include "avi/ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/core/InferenceMode.h>
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>

int main(int argc,char** argv){
  MPI_Init(&argc,&argv);int rank=0,world=0;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&world);
  try{
    TORCH_CHECK(world==2,"Run TP reduction validation with exactly two ranks");
    MPI_Comm local;MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&local);
    int device=0;MPI_Comm_rank(local,&device);MPI_Comm_free(&local);
    C10_CUDA_CHECK(cudaSetDevice(device));c10::InferenceMode guard;
    auto check=[](ncclResult_t r){TORCH_CHECK(r==ncclSuccess,ncclGetErrorString(r));};
    ncclUniqueId id;if(!rank)check(ncclGetUniqueId(&id));
    MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD);
    ncclComm_t comm;check(ncclCommInitRank(&comm,world,id,rank));
    auto opt=at::TensorOptions().device(at::Device(at::kCUDA,device)).dtype(at::kFloat);
    at::manual_seed(42+rank);
    for(int n:{1,5120,32768,1048576})for(double scale:{.001,1.,1000.}){
      auto input=(at::randn({n},opt)*scale).to(at::kBFloat16);
      auto expected=input.to(at::kFloat),actual=input.clone();
      auto stream=at::cuda::getCurrentCUDAStream();
      check(ncclAllReduce(expected.data_ptr(),expected.data_ptr(),n,ncclFloat,ncclSum,comm,stream));
      check(ncclAllReduce(actual.data_ptr(),actual.data_ptr(),n,ncclBfloat16,ncclSum,comm,stream));
      TORCH_CHECK(at::equal(actual,expected.to(at::kBFloat16)),"TP2 BF16 sum differs from FP32 sum rounded to BF16");
    }
    // Both ranks must make the same GPU-only candidate decision, including ties.
    for(bool tie:{false,true}){
      auto logits=at::zeros({4,4097},opt).to(at::kBFloat16);
      logits.select(1,4096).fill_(tie?8:8+rank);
      auto local=avi::vocabulary_candidates(logits,rank*4097),gathered=at::empty({2,4,2},local.options());
      check(ncclAllGather(local.data_ptr(),gathered.data_ptr(),local.numel(),ncclDouble,comm,at::cuda::getCurrentCUDAStream()));
      auto ids=avi::merge_candidates(gathered);
      TORCH_CHECK(at::equal(ids,at::full({4},tie?4096:8193,ids.options())),"Distributed candidate decision mismatch");
    }
    C10_CUDA_CHECK(cudaDeviceSynchronize());check(ncclCommDestroy(comm));
    if(!rank)std::cout<<"TP2 BF16 reduction matches FP32 reference\n";
    MPI_Finalize();return 0;
  }catch(const std::exception& error){
    std::cerr<<"Rank "<<rank<<": "<<error.what()<<"\n";MPI_Abort(MPI_COMM_WORLD,1);return 1;
  }
}
