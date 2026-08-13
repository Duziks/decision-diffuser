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

import os
import shutil
import json
from contextlib import nullcontext
from ml_logger import logger
from easydict import EasyDict as edict
import torch

def output_report(times_range, batch_size):
    times_range.sort()
    report = { "model_name": "Decision-Diffuser"}
    tail_latency = round(times_range[int(len(times_range) * 0.99)] * 1000 /10, 6)
    p90_latency = round(times_range[int(len(times_range) * 0.90)] * 1000 / 10, 6)
    avg_latency = round(sum(times_range) / len(times_range) * 1000 / 10, 6)
    qps = calculate_qps(times_range, batch_size)

    report["QPS"] = qps
    report["AVG Latency"] = avg_latency
    report["P99 Latency"] = tail_latency
    report["P90 Latency"] = p90_latency
    logger.print(f"[ scripts/common ] {report}")
    saved_path = os.path.join("report/")
    save_json(report, saved_path, f"report.json")
    logger.print(f"[ scripts/common ] Report json file saved in {saved_path}")

def calculate_qps(times_range, batches_list):
    return int(sum(batches_list) / sum(times_range))


def save_json(dic: dict, path: str, file_name: str, mode="w"):
    if path:
        if not os.path.exists(path):
            os.makedirs(path)
        js = json.dumps(dic)
        with open(os.path.join(path, file_name), "w") as file:
            file.write(js)


def remove_directory_if_exists(path):
    if os.path.exists(path):
        try:
            # 使用shutil.rmtree删除文件夹及其所有内容
            shutil.rmtree(path)
            logger.print(f"[ scripts/common ] 删除文件夹: {path}")
        except Exception as e:
            logger.print(f"[ scripts/common ] 删除文件夹时出错: {e}")
    else:
        logger.print(f"[ scripts/common ] 路径不存在: {path}")


def get_loop_element(loop_list: list, idx):
    return loop_list[idx % len(loop_list)]


class Profiler:
    def __init__(self, params):
        
        self.params = edict({
            "device":params.get('device'),
            "profiling_path":"profiling/",
            "profiling_mode":True,
            "level":params.get("profiling_level")
        })
        

        if "npu" in self.params.device:
            import torch_npu
            self.level_map = {
                        0: torch_npu.profiler.ProfilerLevel.Level0,
                        1: torch_npu.profiler.ProfilerLevel.Level1,
                        2: torch_npu.profiler.ProfilerLevel.Level2,
                    }
            self.npu_experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[torch_npu.profiler.ExportType.Text],
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=self.level_map[self.params['level']], 
                l2_cache=False,
            )
        self.warmup = 1
        self.activate = 2
        self.skip_first = 10
        self.start_iter = self.skip_first + self.warmup
        self.end_iter = self.start_iter + self.activate

    def trace_handler(self, p):
        profiling_output_dir = os.path.join(self.params.profiling_path)
        if not os.path.exists(profiling_output_dir):
            os.makedirs(profiling_output_dir, mode=0o750)
        p.export_chrome_trace(
            os.path.join(profiling_output_dir, f"trace_{str(p.step_num)}.json")
        )
    

    def get_npu_profiler(self, profiling_output_dir):
        import torch_npu
        return torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                schedule=torch_npu.profiler.schedule(
                    wait=0, warmup=self.warmup, active=self.activate, repeat=1, skip_first=self.skip_first
                ),  # 与prof.step()配套使用
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    profiling_output_dir
                ),
                record_shapes=True,
                with_stack=True,
                profile_memory=False,
                with_modules=False,
                with_flops=False,
                experimental_config=self.npu_experimental_config,
            )

    def get_gpu_profiler(self, profiling_output_dir):
        return torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=0, warmup=self.warmup, active=self.activate, repeat=1, skip_first=self.skip_first
                ),  # 与prof.step()配套使用
                on_trace_ready=self.trace_handler,
                # 形状记录
                record_shapes=True,
                with_stack=True,
                profile_memory=False,
                with_modules=False,
                with_flops=False,
            )

    def get_profiler(self):
        profiler = None
        profiling_output_dir = os.path.join(self.params.profiling_path)
        remove_directory_if_exists(profiling_output_dir)
        if "npu" in self.params.device and self.params.profiling_mode:
            profiler = self.get_npu_profiler(profiling_output_dir)
        elif "cuda" in self.params.device and self.params.profiling_mode:
            profiler = self.get_gpu_profiler(profiling_output_dir)
        else:
            profiler = nullcontext()
        return profiler