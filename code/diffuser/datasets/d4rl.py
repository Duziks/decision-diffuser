import os
import collections
import numpy as np
import gym
import pdb

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def get_dataset(env):
    dataset = env.get_dataset()

    if 'antmaze' in str(env).lower():
        ## the antmaze-v0 environments have a variety of bugs
        ## involving trajectory segmentation, so manually reset
        ## the terminal and timeout fields
        dataset = antmaze_fix_timeouts(dataset)
        dataset = antmaze_scale_rewards(dataset)
        get_max_delta(dataset)

    return dataset

def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env)
    dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode

import gym
from gym.spaces import Box
import numpy as np

class DummyD4RLEnv:
    def __init__(self, name):
        self.name = name
        if 'hopper' in str(name).lower():
            obs_dim, act_dim = 11, 3
        elif 'halfcheetah' in str(name).lower() or 'walker2d' in str(name).lower():
            obs_dim, act_dim = 17, 6
        elif 'ant' in str(name).lower():
            obs_dim, act_dim = 111, 8
        else:
            obs_dim, act_dim = 11, 3
        
        self.observation_space = Box(-np.inf, np.inf, shape=(obs_dim,))
        self.action_space = Box(-1.0, 1.0, shape=(act_dim,))
        self.max_episode_steps = 1000

    def get_dataset(self, **kwargs):
        N = 2000
        obs_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]
        terminals = np.zeros(N, dtype=bool)
        terminals[999] = True
        terminals[1999] = True
        timeouts = np.zeros(N, dtype=bool)
        return {
            'observations': np.random.randn(N, obs_dim).astype(np.float32),
            'actions': np.random.randn(N, act_dim).astype(np.float32),
            'rewards': np.random.randn(N).astype(np.float32),
            'terminals': terminals,
            'timeouts': timeouts,
        }

_orig_load_environment = load_environment

def load_environment(name):
    if type(name) != str:
        return name
    try:
        return _orig_load_environment(name)
    except Exception as e:
        print(f'[Fallback] 无法加载 Gym 环境 "{name}" ({e})，已切换至 DummyD4RLEnv 用于模型构建与改造。')
        return DummyD4RLEnv(name)
