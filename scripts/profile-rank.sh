#!/usr/bin/env bash
set -euo pipefail
: "${AVI_PROFILE_DIR:?Missing profiler output directory}"
: "${AVI_NSYS:?Missing nsys executable}"
rank="${OMPI_COMM_WORLD_RANK:?Run through OpenMPI}"
[[ "$rank" =~ ^[0-9]+$ ]] || exit 2
# One profiler per rank: CUDA profiler start/stop is process-local. Do not wrap
# only mpirun and assume a child profiler API controls all rank captures.
exec "$AVI_NSYS" profile --trace=cuda,nvtx,osrt,mpi --mpi-impl=openmpi \
 --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
 --capture-range-end=stop --cuda-graph-trace=node \
 --output="$AVI_PROFILE_DIR/rank-$rank" "$@"
