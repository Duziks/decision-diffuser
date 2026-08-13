import os
import copy
import numpy as np
import torch
import einops
import diffuser
from copy import deepcopy
from tqdm import tqdm

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from ml_logger import logger
from diffuser.models.helpers import apply_conditioning


def cycle(dl):
    while True:
        for data in dl:
            yield data

transform_keys = []

def transform_pre_fn(*args, **kwargs):
    transform_inputs = []
    for key, value in args[0].items():
        transform_inputs.append(value)
        transform_keys.append(key)
    return transform_inputs

def transform_post_fn(trans_outputs, **kwargs):
    arg_list = []
    for trans_output in trans_outputs:
        arg = {}
        for idx, tensor in enumerate(trans_output):
            arg[transform_keys[idx]] = tensor
        arg_list.append((arg,))
    kwargs_list = [{}] * len(arg_list)
    return arg_list, kwargs_list

recover_keys = []

def recover_pre_fn(groups):
    recover_inputs = []
    for group in groups:
        recover_input = []
        for value in group.values():
            recover_input.append(value)
        recover_inputs.append(recover_input)
    for key in groups[0].keys():
        recover_keys.append(key)

    return recover_inputs

def recover_post_fn(re_outputs):
    real_output = {}
    for idx, re_output in enumerate(re_outputs):
        real_output[recover_keys[idx]] = re_output
    return real_output

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        n_reference=8,
        bucket=None,
        train_device='cuda',
        save_checkpoints=False,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset

        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

        self.device = train_device
        

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for _ in tqdm(range(n_train_steps)):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                self.save()

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'[{key}]: {val:8.4f}' for key, val in infos.items()])
                logger.print(f'[step]: {self.step:8} | [Loss] {loss:8.4f} | {infos_str} | [time]: {timer():8.4f}')
                

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                if self.model.__class__ == diffuser.models.diffusion.GaussianInvDynDiffusion:
                    self.inv_render_samples()
                elif self.model.__class__ == diffuser.models.diffusion.ActionGaussianDiffusion:
                    pass
                else:
                    self.render_samples()

            self.step += 1

    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.bucket, f'checkpoint{self.ema_model.n_timesteps}')
        os.makedirs(savepath, exist_ok=True)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        logger.print(f'[ utils/training ] Saved model to {savepath}')

    def load(self):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.bucket, f'checkpoint/state.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join('images', f'sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        for i in range(batch_size):
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, self.device)
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            if self.ema_model.returns_condition:
                returns = to_device(torch.ones(n_samples, 1), self.device)
            else:
                returns = None

            if self.ema_model.model.calc_energy:
                samples = self.ema_model.grad_conditional_sample(conditions, returns=returns)
            else:
                samples = self.ema_model.conditional_sample(conditions, returns=returns)

            samples = to_np(samples)

            normed_observations = samples[:, :, self.dataset.action_dim:]
            normed_conditions = to_np(batch.conditions[0])[:,None]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join('images', f'sample-{i}.png')
            self.renderer.composite(savepath, observations)

    def inv_render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, self.device)
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            if self.ema_model.returns_condition:
                returns = to_device(torch.ones(n_samples, 1), self.device)
            else:
                returns = None
            
            if self.ema_model.model.calc_energy:
                samples = self.ema_model.grad_conditional_sample(conditions, returns=returns)
            else:
                samples = self.ema_model.conditional_sample(conditions, returns=returns)

            samples = to_np(samples)

            normed_observations = samples[:, :, :]
            normed_conditions = to_np(batch.conditions[0])[:,None]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')


            savepath = os.path.join('images', f'sample-{i}.png')
            self.renderer.composite(savepath, observations)

    #-----------------------------------------------------------------------------#
    #--------------------------------- handle -----------------------------------#
    #-----------------------------------------------------------------------------#
    def set_handle(self, params):
        self.shape_handle = params["shape_handle"] == "true"
        self.compile = params["compile"] == "true"
        self.graph = params["graph"] == "true"
        self.hf32 = params["hf32"] == "true"
        self.graphs = {}
        self.manual_graph = False

        if params['device'] != "cpu":
            if params['device'] == "npu":
                import torch_npu
                torch.npu.set_device(params['device_id'])
            
        if self.shape_handle :
            self.shape_options = {
                "enable_shape_handling": True,
                "shape_handling_min_size": 1,
                "shape_handling_max_size": 1024,
                "shape_handling_dict": {
                    "trans_pre_fn": transform_pre_fn,
                    "trans_post_fn": transform_post_fn,
                    "re_pre_fn": recover_pre_fn,
                    "re_post_fn": recover_post_fn,
                },
            }
        logger.print(
            f"[ utils/training ] *********************compile:{self.compile}**********************", 
            color = "cyan"
            )
        logger.print(
            f"[ utils/training ] *********************graph:{self.graph}**********************", 
            color = "cyan"
            )

    def set_hf32(self):
        if "npu" in self.device:
            import torch_npu
            torch_npu.npu.aclnn.allow_hf32 = self.hf32
            torch_npu.npu.conv.allow_hf32 = self.hf32
            torch_npu.npu.matmul.allow_hf32 = self.hf32
        elif "cuda" in self.device:
            torch.backends.cuda.matmul.allow_tf32 = self.hf32
            torch.backends.cudnn.allow_tf32 = self.hf32
        logger.print(
            f"[ utils/training ] *********************tf32: {self.hf32}**********************",
            color = "cyan"
        )

    def synchronize(self):
        if "npu" in self.device:
            torch.npu.synchronize()
        elif "cuda" in self.device:
            torch.cuda.synchronize()
        elif "mlu" in self.device:
            torch.mlu.synchronize()

    def is_manual_graph(self):
        return not self.compile and self.graph and ("npu" in self.device or "cuda" in self.device)
    
    def generate_inputs(self, conditions, returns, observation_dim):
        shape = (len(conditions[0]), self.ema_model.horizon, self.ema_model.observation_dim)
        x = 0.5 * torch.randn(shape, device=self.ema_model.betas.device)
        inputs = {
            "conditions": conditions,
            "returns": returns,
            "observation_dim": observation_dim,
            "x": apply_conditioning(x, conditions, 0),
            "noise": 0.5 * torch.randn_like(x)
        }
        return inputs

    def forward(self, inputs):
        conditions = inputs["conditions"]
        returns = inputs["returns"]
        observation_dim = inputs["observation_dim"]
        x = inputs["x"]
        noise = inputs["noise"]
        samples = self.ema_model.conditional_sample(conditions, x=x, noise=noise, returns=returns)
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2 * observation_dim)
        action = self.ema_model.inv_model(obs_comb)
        return samples, action

    def model_infer(self, inputs):
        if self.manual_graph:
            return self.model_infer_graph(inputs, self.batch_size)
        return self.forward(inputs)

    def model_infer_graph(self, inputs, batch_size):
        if batch_size not in self.graphs:
            # new batch size scenario, a warmup is required
            new_batch = {
                "graph": (
                    torch.npu.NPUGraph()
                    if "npu" in self.device
                    else torch.cuda.CUDAGraph()
                ),
                "stream": (
                    torch.npu.Stream(self.device)
                    if "npu" in self.device
                    else None
                ),
                "static_input": None,
                "static_output": None,
            }
            new_batch["static_input"] = {k: copy.deepcopy(v) for k, v in inputs.items()}
            if "npu" in self.device:
                with torch.npu.graph(new_batch["graph"], None, new_batch["stream"]):
                    new_batch["static_output"] = self.forward(new_batch["static_input"])
            else:
                for _ in range(3):
                    with torch.no_grad():
                        _ = self.forward(new_batch["static_input"])
                self.synchronize()

                with torch.cuda.graph(new_batch["graph"]):
                    new_batch["static_output"] = self.forward(new_batch["static_input"])
            self.graphs[batch_size] = new_batch
        else:
            for k in inputs.keys():
                self.graphs[batch_size]["static_input"][k] = inputs[k]
                

        self.graphs[batch_size]["graph"].replay()
        self.synchronize()

        return self.graphs[batch_size]["static_output"]

    def set_compile_model(self):
        if self.compile:
            if self.graph and self.shape_handle and "npu" in self.device:
                self.shape_options["triton.cudagraphs"] = True
                self.forward = torch.compile(
                    self.forward, backend="inductor", dynamic=False, options=self.shape_options
                )
            elif self.shape_handle and "npu" in self.device:
                self.forward = torch.compile(
                    self.forward, backend="inductor", dynamic=False, options=self.shape_options
                )
            elif self.graph:
                self.forward = torch.compile(
                    self.forward, backend="inductor", dynamic=False, mode="reduce-overhead"
                )
            else:
                self.forward = torch.compile(self.forward, backend="inductor", dynamic=False)
        else:
            self.forward = self.forward
            if self.graph and ("npu" in self.device or "cuda" in self.device):
                self.manual_graph = True
                self.graph_prepared = False
                