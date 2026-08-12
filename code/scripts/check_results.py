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
import sys
import torch
import numpy as np
from ml_logger import logger

class ioChecker:
    def __init__(self, params):
        self.path = params["bucket"]
        self.device = params["device"]
        self.check = params["check_results"] == "true"
        torch.backends.cudnn.deterministic = self.check
        torch.backends.cudnn.benchmark = not self.check

    def _to_cpu_detached(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: self._to_cpu_detached(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = [self._to_cpu_detached(v) for v in obj]
            return type(obj)(t)
        return obj

    def _to_device(self,obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: self._to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = [self._to_device(v) for v in obj]
            return type(obj)(t)
        return obj

    def load_or_save_inputs(self, inputs, t):
        if not self.check or t != 0:
            return inputs
        save_path = os.path.normpath(self.path + '/../results')
        if os.path.exists(save_path):
            data = torch.load(save_path + '/inputs.pt', map_location="cpu")
            logger.print(
                f"[ scripts/check_results ] loading input from {save_path}/inputs.pt", 
                color='cyan'
            )
            return self._to_device(data, self.device) if self.device else data

        os.makedirs(save_path , exist_ok=True)
        torch.save(self._to_cpu_detached(inputs), save_path + '/inputs.pt',)
        logger.print(
            f"[ scripts/check_results ] save input to {save_path}/inputs.pt", 
            color='cyan'
        )
        return inputs

    def save_outputs(self, outputs, t):
        if t == 0 and self.check:
            save_path = os.path.normpath(self.path + '/../results/outputs.pt')
            os.makedirs(self.path, exist_ok=True)
            torch.save(self._to_cpu_detached(outputs), save_path)
            logger.print(
                f"[ scripts/check_results ] save output to {save_path}", 
                color='cyan'
            )

def load_value(path):
    x = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).cpu().float()
    return torch.tensor(x).cpu().float()

def compare_tensor_outputs(file1_path, file2_path, atol=1e-5, rtol=1e-5):
    input1 = load_value(file1_path)
    input2 = load_value(file2_path)

    
    if input1.shape != input2.shape:
        raise ValueError(f"Shape mismatch: gpu={input1.shape}, npu={input2.shape}")

    diff = (input2 - input1).abs()

    p99 = torch.quantile(diff, 0.99).item()
    eps = 1e-12
    rel = diff / (input1.abs() + eps)

    all_ok = torch.allclose(input2, input1, atol=atol, rtol=rtol)

    pyPath = "[ scripts/compare_results ]"
    logger.print(f"{pyPath} ==== NPU vs GPU output====", color='cyan')
    logger.print(f"{pyPath} files1: {file1_path} ", color='cyan')
    logger.print(f"{pyPath} files2: {file2_path}", color='cyan')
    logger.print(f"{pyPath} shape: {tuple(input1.shape)} | numel: {input1.numel()}", color='cyan')
    logger.print(f"{pyPath} allclose(atol={atol}, rtol={rtol}): {all_ok}", color='cyan')
    logger.print(f"{pyPath} mean_abs_err  : {diff.mean().item():.6g}", color='cyan')
    logger.print(f"{pyPath} mean_rel_err  : {rel.mean().item():.6g}", color='cyan')
    logger.print(f"{pyPath} p99_abs_err   : {p99:.6g}", color='cyan')
    torch.testing.assert_close(input1, input2, rtol=rtol, atol=atol, equal_nan=True)
    if all_ok:
        logger.print("Precision check pass!")
    else:
        logger.print("Precision check failed!")


if __name__ == "__main__":

    if len(sys.argv) < 4:
        logger.print("Usage: python3 check_result.py <file1_path> <file2_path> <tolerance>")

    file1_path = sys.argv[1]  # 第一个参数：file1_result_path
    file2_path = sys.argv[2]  # 第二个参数：file2_result_path
    tolerance = float(sys.argv[3])  # 第三个参数：2e-3
    compare_tensor_outputs(
        file1_path=file1_path,
        file2_path=file2_path,
        atol=tolerance,
        rtol=tolerance,
    )
