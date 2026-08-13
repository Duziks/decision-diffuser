#!/bin/bash
rm -rf /tmp/torchinductor_root/*
rm -rf ~/.triton/dump
rm -rf ~/.triton/cache
export ENABLE_ACLGRAPH=1
export TRITON_DISABLE_FFTS=1
# cann
source /data/Ascend-b080/ascend-toolkit/set_env.sh
# ir 
ir_path="/data/xxx/env/bisheng/8.5.0/compiler/bishengir"
export PATH=${ir_path}/bin:$PATH
export LD_LIBRARY_PATH=${ir_path}/lib:$LD_LIBRARY_PATH
# 图优化
export PRE_GRAPH_OPTIMIZER=1
export POST_GRAD_GRAPH_OPTIMIZER=1
# inductor
export INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE=1
export INDUCTOR_TINY_KERNEL=1
export TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING=1
export TORCHNPU_PRECOMPILE_THREADS=100
# gather
export TRITON_EMBEDDING_FUSION=1
export TRITON_INDEX_FUSION=1
# catlass
export CATLASS_EVG_FUSION=1
export CATLASS_EPILOGUE_FUSION=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_root
export TORCHINDUCTOR_COMPILE_THREADS=1
export TRITON_BENCH_METHOD=npu
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=CATLASS,ATen
export TORCHINDUCTOR_NPU_CATLASS_DIR=/data/xxx/catlass-1218
# export INDUCTOR_NPU_CATLASS_BENCH_USE_PROFILING=1