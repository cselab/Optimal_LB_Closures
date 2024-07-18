from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import wandb

from gymnasium import spaces
from tianshou.data import Batch
from torch.utils.data import DataLoader
from scipy.interpolate import griddata
from typing import Tuple

from lib.utils import sample_from_dataloader
from lib.environments.base import BaseEnvironment
from lib.environments.velocity_generation import VelocityFieldGenerator
from lib.datasets import get_train_val_test_initial_conditions_dataset
from lib.models.wrappers import MarlModel


# a discrete version of a pendulum 
class Pendulum():
    def __init__(self, omega, dt, x_0, x_1):
        self.x_0 = x_0
        self.x_1 = x_1
        self.x_now = x_1
        self.x_prev = x_0
        self.dt = dt
        self.n = 1
        self.omega = omega

    def step(self):
        x_next = self.x_now * (2 - self.dt**2 * self.omega**2) - self.x_prev
        self.x_prev = np.copy(self.x_now)
        self.x_now = np.copy(x_next)
        self.n += 1
        return x_next

    def reset(self):
        self.x_now = self.x_1
        self.x_prev = self.x_0
        self.n = 1


class two_pendulums(BaseEnvironment, ABC):
    def __init__(self, omega, T=10, N1=100, N2=100):
        super().__init__()
        self.x_0 = -1
        self.T = T
        self.N1 = N1
        self.N2 = N2
        self.dt1 = self.T/self.N1
        self.dt2 = self.T/self.N2
        self.x_11 = -1*np.cos(omega*self.dt1)
        self.x_12 = -1*np.cos(omega*self.dt2)
        self.omega = omega
        self.cgs = Pendulum(omega, self.dt1, self.x_0, self.x_11)
        self.fgs = Pendulum(omega, self.dt2, self.x_0, self.x_12)
        self.factor = int(self.dt1/self.dt2)
        self.counter = 1
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=2, shape=(1,), dtype=np.float32)
        self.x1 = [self.x_0, self.x_11]
        self.x2 = [self.x_0, self.x_12]
        

    def reset(self, *args, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.counter = 1
        self.cgs.reset()
        self.fgs.reset()
        self.x1 = [self.x_0, self.x_11]
        self.x2 = [self.x_0, self.x_12]
        return [self.x1[-1]], {}


    def step(self, action):
        #self.cgs.omega = self.fgs.omega * np.float64(action[0])
        self.cgs.omega = np.float64(action[0])
        self.x1.append(self.cgs.step())
        for i in range(self.factor):
            x2_next = self.fgs.step()
        self.x2.append(x2_next)
        self.counter += 1

        #compute error
        err = -1 * (self.x1[-1] - self.x2[-1])**2

        terminated = bool(np.abs(err)>0.1)
        truncated = bool(self.counter==self.N1)
        if terminated:
            err -= 100
        else:
            err += 0.1

        info = {}
        return [self.x1[-1]], err, terminated, truncated, info


    def render(self, title: str = 'state'):
        t = np.linspace(0, self.T * (self.counter/self.N1), self.counter+1)
        plt.figure()
        plt.plot(t, self.x1, label="cgs")
        plt.plot(t, self.x2, label="fgs")
        plt.xlim(0,self.T)
        plt.legend()
        plt.show()



#extra stuff
    def _play_and_get_eval_metrics(self, actor: MarlModel, ep_len: int = 21) -> Batch:

        # Play episode with and without RL
        _total_reward = 0.
        _obs, _ = self.reset()

        for i in range(ep_len):
            # Simulation with MARL in loop
            act_mean = actor.get_action_mean(_obs)

            # Compute RL adjusted update
            _obs, reward, _terminated, _, _ = self.step(act_mean.detach().cpu().numpy()[0])
            _total_reward += reward

        # Storing evaluation metrics in batch
        eval_metrics = Batch()
        eval_metrics['rews'] = _total_reward

        # TODO: add many more eval metrics

        return eval_metrics


    def play_episode_and_log_to_wandb(self,
                                      actor: MarlModel,
                                      step: int,
                                      ep_len: int) -> np.array:
        _min_duration = 9
        assert ep_len > _min_duration, f"test_sim_len must be > {_min_duration}"

        _num_pics = 4
        _plotting_freq = int(ep_len / _num_pics)  # we want to get `_num_pics` pics in total
        _obs, _ = self.reset()
        _no_rl_state = self.initial_condition.copy()

        _log_rl_imgs = []
        _log_no_rl_imgs = []
        _log_dns_imgs = []
        _log_action_imgs = []
        _log_target_acts = []

        # Log initial state
        _log_rl_imgs.append(wandb.Image(self.initial_condition.clip(0, 1), caption=f'state@{self.step_count}'))
        _log_no_rl_imgs.append(wandb.Image(_no_rl_state.clip(0, 1), caption=f'state@{self.step_count}'))
        _log_dns_imgs.append(wandb.Image(self.gt_state[::self.subsample, ::self.subsample].clip(0, 1),
                                         caption=f'dns_state@{self.step_count}'))

        for i in range(ep_len):
            # Simulation with MARL in loop
            act_mean = actor.get_action_mean(_obs).clip(self.action_space.low.min(), self.action_space.high.max())

            # Compute RL adjusted update
            _obs, reward, _terminated, _, _ = self.step(act_mean.detach().cpu().numpy()[0])
            _obs['domain'] = np.expand_dims(_obs['domain'], axis=0)  # need to create fake batch dimension
            _obs['velocity_field'] = np.expand_dims(_obs['velocity_field'], axis=0)

            _no_rl_state = self.upwind_scheme_2d_step(_no_rl_state, self.dt, self.dx, self.dy, self.c_x, self.c_y)

            # Pure numerical simulation -> will lead to numerical errors
            if self.step_count % _plotting_freq == 0:
                _log_rl_imgs.append(wandb.Image(self.rl_adjusted_state.clip(0, 1),
                                                caption=f'state@{self.step_count}'))
                _log_no_rl_imgs.append(wandb.Image(_no_rl_state.clip(0, 1),
                                                   caption=f'state@{self.step_count}'))
                _log_action_imgs.append(wandb.Image(act_mean[0].detach().cpu().numpy()[0],
                                                    caption=f'action@{self.step_count}'))
                _log_dns_imgs.append(wandb.Image(self.gt_state[::self.subsample, ::self.subsample].clip(0, 1),
                                                 caption=f'dns_state@{self.step_count}'))
                _log_target_acts.append(wandb.Image(self.mde_upwind_diffusion_term(self.state, self.c_x, self.c_y, self.dx, self.dy, self.dt),
                                                    caption=f'target_action@{self.step_count}'))

        # Log images of episode to wandb
        _log_rl_imgs.append(wandb.Image(self.rl_adjusted_state.clip(0, 1),
                                        caption='target'))
        _log_no_rl_imgs.append(wandb.Image(_no_rl_state.clip(0, 1),
                                           caption='target'))
        _log_dns_imgs.append(wandb.Image(self.gt_state[::self.subsample, ::self.subsample].clip(0, 1),
                                         caption='target'))

        rl_log_dict = {f'{self.dataset_name} RL in loop': _log_rl_imgs}
        no_rl_log_dict = {f'{self.dataset_name} no RL': _log_no_rl_imgs}
        _log_action_dict = {f'{self.dataset_name} actions': _log_action_imgs}
        _log_target_acts_dict = {f'{self.dataset_name} target actions': _log_target_acts}
        _log_dns_dict = {f'{self.dataset_name} DNS': _log_dns_imgs}
        wandb.log(rl_log_dict, step=step)
        wandb.log(no_rl_log_dict, step=step)
        wandb.log(_log_action_dict, step=step)
        wandb.log(_log_target_acts_dict, step=step)
        wandb.log(_log_dns_dict, step=step)
        return None


    def _get_obs(self):
        # return with extra channel dimension for torch NN
        return {
            'domain': np.expand_dims(self.state, axis=0),
            'velocity_field': np.array([self.c_x, self.c_y], dtype=np.float32)
        }
