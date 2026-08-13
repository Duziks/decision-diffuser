import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from .mujoco_compat import get_mujoco_data

class HopperFullObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        asset_path = os.path.join(
            os.path.dirname(__file__), 'assets/hopper.xml')
        mujoco_env.MujocoEnv.__init__(self, asset_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        data = get_mujoco_data(self)
        posbefore = data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        data = get_mujoco_data(self)
        return np.concatenate([
            # self.sim.data.qpos.flat[1:],
            data.qpos.flat,
            np.clip(data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def set(self, state):
        qpos_dim = get_mujoco_data(self).qpos.size
        qpos = state[:qpos_dim]
        qvel = state[qpos_dim:]
        self.set_state(qpos, qvel)
        return self._get_obs()
