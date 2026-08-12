#!/bin/bash
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export PATH=$PATH:/usr/bin/gcc
export PYTHONPATH=${PWD}/..:$PYTHONPATH

DEVICE_ID=0
DEVICE="cuda" # cpu/npu/cuda
PATTERN="eval" # train/eval
WEIGHTS_PATH=${PWD}/../weights   #权重保存和加载的路径

n_diffusion_steps=10

# ===== 动态shape改造：动态模式开关 =====
# DYNAMIC=true  开启动态编译模式（dynamic=True，跳过图捕获，走纯前向benchmark）
# DYNAMIC=false 走原有静态/分桶模式（保持向后兼容）
DYNAMIC=true

# 动态batch_size分档列表，逗号分隔。动态模式下按此列表轮转测试不同batch
# 为空则单档使用 test_batch_size
DYNAMIC_BS_LIST="1,2,4,8,16"

# 每个batch档位重复测试轮数
DYNAMIC_ROUNDS=5

# 当 DYNAMIC=true 时，COMPILE 必须为 true（动态编译依赖torch.compile）
# 当 DYNAMIC=true 时，GRAPH 自动忽略（动态模式不走图捕获）
# 当 DYNAMIC=true 时，SHAPE_HANDLE 自动忽略
COMPILE=false
GRAPH=false
SHAPE_HANDLE=false

# 动态模式自动校正参数
if [ "$DYNAMIC" == "true" ]; then
    COMPILE=true
    GRAPH=false
    SHAPE_HANDLE=false
    echo "[launch] 动态shape模式启动: COMPILE=true, GRAPH=false, SHAPE_HANDLE=false"
    echo "[launch] bs_list=${DYNAMIC_BS_LIST}, rounds=${DYNAMIC_ROUNDS}"
fi
# =======================================

if [ "$DEVICE" == "npu" ]
then
    source npu_env.sh # 激活npu环境
    export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_ID}
    echo "set npu:$DEVICE_ID"
    export ASCEND_DEVICE_ID=0
    export RANK_ID=0
    export DEVICE_ID=0
elif [ "$DEVICE" == "cuda" ]
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
                --graph=${GRAPH} \
                --compile=${COMPILE} \
                --shape_handle=${SHAPE_HANDLE} \
                --check_results=false \
                --profiling_level=0 \
                --bucket=${WEIGHTS_PATH} \
                --dynamic=${DYNAMIC} \
                --dynamic_bs_list=${DYNAMIC_BS_LIST} \
                --dynamic_rounds=${DYNAMIC_ROUNDS}

echo "finished!"