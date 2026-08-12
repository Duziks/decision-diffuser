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
from copy import deepcopy
import os
import numpy as np
import gym
import d4rl
import time
import torch
import diffuser.utils as utils
from ml_logger import logger, RUN
from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.datasets.d4rl import suppress_output
from scripts.common import Profiler, output_report
from scripts.check_results import ioChecker

def load_state(Config):
    loadpath = os.path.join(Config.bucket, 'checkpoint' + str(Config.n_diffusion_steps))

    if Config.save_checkpoints:
        loadpath = os.path.join(loadpath, f'state.pt')
    else:
        loadpath = os.path.join(loadpath, 'state.pt')

    state_dict = torch.load(loadpath, map_location=Config.device)
    return state_dict

def set_dataConfig(Config):
    dataset_config = utils.Config(
        Config.loader,
        savepath='tmp/dataset_config.pkl',
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
    )
    render_config = utils.Config(
        Config.renderer,
        savepath='tmp/render_config.pkl',
        env=Config.dataset,
    )
    dataset = dataset_config()
    renderer = render_config()
    return dataset, renderer

def set_modelConfig(Config, transition_dim, observation_dim, action_dim):
    model_config = utils.Config(
        Config.model,
        savepath='tmp/model_config.pkl',
        horizon=Config.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        device=Config.device,
        returns_condition=Config.returns_condition,
        condition_dropout=Config.condition_dropout,
    )
    diffusion_config = utils.Config(
        Config.diffusion,
        savepath='tmp/diffusion_config.pkl',
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        predict_epsilon=Config.predict_epsilon,
        loss_discount=Config.loss_discount,
        loss_weights=Config.loss_weights,
        device=Config.device,
        condition_guidance_w=Config.condition_guidance_w,
        returns_condition=Config.returns_condition,
    )
    return model_config, diffusion_config

def set_trainer(Config):
    trainer_config = utils.Config(
        utils.Trainer,
        savepath='tmp/trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        label_freq=Config.label_freq,
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
    )
    return trainer_config

def dynamic_benchmark(trainer, dataset, obs, returns, observation_dim, bs_list, rounds, deps):
    """
    动态shape纯前向benchmark：按bs_list轮转不同batch_size，测量前向性能。
    不与Gym环境交互，只测模型推理性能。
    """
    profiler = Profiler(deps)
    profiling = profiler.get_profiler()
    checker = ioChecker(deps)

    times_range = []
    batch_size_record = []

    try:
        import torch._dynamo
        torch._dynamo.config.verbose = True
    except Exception:
        pass

    for cur_bs in bs_list:
        cur_obs = obs[:cur_bs]
        cur_returns = returns[:cur_bs]
        cur_obs = dataset.normalizer.normalize(cur_obs, 'observations')
        conditions = {0: to_torch(cur_obs, device=Config.device)}
        inputs = trainer.generate_inputs(conditions, cur_returns, observation_dim)

        for _ in range(3):
            with torch.no_grad():
                _ = trainer.model_infer(inputs)
        trainer.synchronize()

        for r in range(rounds):
            with profiling as prof:
                trainer.synchronize()
                start_time = time.time()
                samples, action = trainer.model_infer(inputs)
                trainer.synchronize()
                end_time = time.time()
                times_range.append(end_time - start_time)
                batch_size_record.append(cur_bs)
                prof.step()

        avg_time = sum(times_range[-rounds:]) / rounds
        logger.print(
            f"[dynamic bench] batch_size={cur_bs}, avg_time={avg_time:.4f}s",
            color="green"
        )

    try:
        counters = torch._dynamo.utils.counters
        if 'stats' in counters and 'unique_graphs' in counters['stats']:
            unique_graphs = counters['stats']['unique_graphs']
            logger.print(
                f"[dynamic bench] torch.compile unique_graphs={unique_graphs} "
                f"(1=动态编译成功泛化, >1=每个batch重编译)",
                color="green"
            )
    except Exception:
        pass

    output_report(times_range, batch_size_record)

def evaluate(**deps):
    RUN._update(deps)
    Config._update(deps)
    logger.remove('*.pkl')
    logger.remove("*.log")
    logger.remove("traceback.err")

    state_dict = load_state(Config)
    # Load configs
    utils.set_seed(deps)

    # DataConfig
    dataset, renderer = set_dataConfig(Config)

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    transition_dim = observation_dim
    if Config.diffusion != 'models.GaussianInvDynDiffusion':
        transition_dim = observation_dim + action_dim

    model_config, diffusion_config = set_modelConfig(
        Config,
        transition_dim,
        observation_dim,
        action_dim
    )
    trainer_config = set_trainer(Config)
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)
    trainer.step = state_dict['step']
    trainer.model.load_state_dict(state_dict['model'])
    trainer.ema_model.load_state_dict(state_dict['ema'])

    is_dynamic = deps.get("dynamic", "false") == "true"
    bs_list_str = deps.get("dynamic_bs_list", "")
    rounds = int(deps.get("dynamic_rounds", 3))

    if bs_list_str:
        bs_list = [int(x) for x in bs_list_str.split(",") if x.strip()]
    else:
        bs_list = [deps["test_batch_size"]]

    num_eval = max(bs_list)
    # 动态模式用合成数据，不依赖 Gym 环境
    num_eval_dyn = max(bs_list) if bs_list else deps["test_batch_size"]
    obs = np.zeros((num_eval_dyn, observation_dim), dtype=np.float32)
    returns = to_device(Config.test_ret * torch.ones(num_eval_dyn, 1), Config.device)

    # 动态模式也需要初始化 trainer 的 handle/compile
    trainer.set_handle(deps)
    trainer.set_hf32()
    trainer.set_compile_model()
    trainer.manual_graph = trainer.is_manual_graph()
    trainer.ema_model.eval()

    if is_dynamic:
        logger.print(
            f"[evaluate] 动态shape模式启动, bs_list={bs_list}, rounds={rounds}",
            color="green"
        )
        dynamic_benchmark(
            trainer, dataset, obs, returns, observation_dim,
            bs_list, rounds, deps
        )
        logger.print("[evaluate] 动态shape benchmark完成", color="green")
        return

    env_list = [gym.make(Config.dataset) for _ in range(num_eval)]
    dones = [0 for _ in range(num_eval)]
    returns = to_device(Config.test_ret * torch.ones(num_eval, 1), Config.device)
    t = 0
    obs_list = [env.reset()[None] for env in env_list]
    obs = np.concatenate(obs_list, axis=0)
    recorded_obs = [deepcopy(obs[:, None])]

    times_range = []
    batch_size = []
    profiler = Profiler(deps)
    profiling = profiler.get_profiler()
    checker = ioChecker(deps)

    trainer.set_handle(deps)
    trainer.set_hf32()
    trainer.set_compile_model()
    trainer.manual_graph = trainer.is_manual_graph()
    trainer.ema_model.eval()


    while sum(dones) < num_eval:
        obs = dataset.normalizer.normalize(obs, 'observations')
        conditions = {0: to_torch(obs, device=Config.device)}
        inputs = trainer.generate_inputs(conditions, returns, observation_dim)
        inputs = checker.load_or_save_inputs(inputs, t)
        with profiling as prof:
            trainer.synchronize()
            start_time = time.time()
            samples, action = trainer.model_infer(inputs)
            trainer.synchronize()
            end_time = time.time()
            if t < profiler.start_iter + 10 and t >= profiler.start_iter + 3:
                times_range.append(end_time - start_time)
                batch_size.append(num_eval)
            prof.step()

        samples = to_np(samples)
        action = to_np(action)
        checker.save_outputs(samples, t)

        action = dataset.normalizer.unnormalize(action, 'actions')
        if t == 0:
            normed_observations = samples[:, :, :]
            observations = dataset.normalizer.unnormalize(normed_observations, 'observations')
            savepath = os.path.join('images', f'sample-plan.png')
            renderer.composite(savepath, observations)

        obs_list = []
        for i in range(num_eval):
            this_obs, this_reward, this_done, _ = env_list[i].step(action[i])
            obs_list.append(this_obs[None])
            if dones[i] == 1:
                continue
            if this_done:
                dones[i] = 1

        obs = np.concatenate(obs_list, axis=0)
        recorded_obs.append(deepcopy(obs[:, None]))
        t += 1

    output_report(times_range, batch_size)

    recorded_obs = np.concatenate(recorded_obs, axis=1)
    savepath = os.path.join('images', f'sample-executed.png')
    renderer.composite(savepath, recorded_obs)