# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse

def get_argument():
    parser = argparse.ArgumentParser()
    parser = get_model_argument(parser)
    parser = get_opt_argument(parser)

    args = parser.parse_args()
    return args

def get_model_argument(parser):
    parser.add_argument('--n_diffusion_steps', type=int, help='去噪次数(模型推理次数)', default=200)
    parser.add_argument('--pattern', type=str, help='启动模式（train or eval）', default="eval",choices=["train", "eval"])
    parser.add_argument('--bucket', type=str, help='权重路径', default="")
    parser.add_argument("--seed", type=int, default=100, help="随机数种子")
    parser.add_argument(
        "--test_batch_size", type=int, default=10, help="test_qps模式下batch_size大小"
    )
    parser.add_argument(
        "--enable_dynamic_compile",
        type=lambda v: {"true": True, "false": False, "none": None}[v.lower()],
        default=False,
        help="Dynamic compile mode: False (static), True (full dynamic), None (auto detect)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "npu"],
        help="指定训练设置，支持cpu/cuda/npu",
    )
    parser.add_argument(
        "--check_results",
        type=str,
        default="false",
        choices=["true", "false"],
        help="检查输入输出精度，是否保存输入输出",
    )
    parser.add_argument("--device_id", type=int, default=0, help="指定设备id")
    
    return parser

def get_opt_argument(parser):
    parser.add_argument(
        "--hf32",
        type=str,
        default="true",
        choices=["true", "false"],
        help="是否启用hf32(npu)/tf32(cuda)",
    )
    parser.add_argument(
        "--compile",
        type=str,
        default="true",
        choices=["true", "false"],
        help="是否启用inductor模式(torch.compile)",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="false",
        choices=["true", "false"],
        help="是否启用图下沉模式",
    )
    parser.add_argument(
        "--shape_handle",
        type=str,
        default="false",
        choices=["true", "false"],
        help="是否启用分档(torch.compile)，支持npu",
    )
    parser.add_argument(
        "--profiling_level",
        type=int, help='profiling的level等级，默认为0', default=0
    )
    return parser
