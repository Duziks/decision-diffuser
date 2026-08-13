#!/bin/bash
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export PATH=$PATH:/usr/bin/gcc
export PYTHONPATH=${PWD}/..:$PYTHONPATH

DEVICE_ID=0
DEVICE= # cpu/npu/cuda
PATTERN=eval # train/eval
WEIGHTS_PATH=${PWD}/../weights   #权重保存和加载的路径

n_diffusion_steps=10

if [ $DEVICE == "npu" ]
then
    source npu_env.sh # 激活npu环境
    export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_ID}
    echo "set npu:$DEVICE_ID"
    export ASCEND_DEVICE_ID=0
    export RANK_ID=0
    export DEVICE_ID=0
elif [ $DEVICE == "cuda" ]
then
    export CUDA_VISIBLE_DEVICES=${DEVICE_ID}
    echo "set cuda:$DEVICE_ID"
fi

python main.py  --device=${DEVICE} \
                --device_id=0 \
                --pattern=${PATTERN} \
                --n_diffusion_steps=${n_diffusion_steps} \
                --test_batch_size=10 \
                --seed=100 \
                --hf32=true \
                --graph=false \
                --compile=false \
                --check_results=false \
                --profiling_level=0 \
                --bucket=${WEIGHTS_PATH} # 权重路径

echo "finished!"