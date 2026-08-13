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
from ml_logger import logger
from scripts.evaluate_inv_parallel import evaluate
from scripts.train import main
from analysis.parser import get_argument

if __name__ == '__main__':
    args = get_argument()

    kwargs = {
        "RUN.prefix": "diffuser/", 
        "seed": 100, 
        "returns_condition": True, 
        "predict_epsilon": True, 
        "n_diffusion_steps": 200, 
        "condition_dropout": 0.25, 
        "diffusion": "models.GaussianInvDynDiffusion", 
        "n_train_steps": 10000.0, 
        "dataset": "hopper-medium-expert-v2", 
        "returns_scale": 400.0, 
        "RUN.job_counter": 1, 
        "RUN.job_name": "test"
        }
    
    kwargs.update(vars(args))
    logger.print(f"[ analysis/main ] {kwargs}", color = "cyan")

    if kwargs.get("pattern") == 'train':
        main(**kwargs)
    else:
        evaluate(**kwargs)