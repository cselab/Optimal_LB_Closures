from abc import ABC
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import scipy as scp
from tqdm import tqdm
import wandb
import sys
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import gc
import jax.numpy as jnp
import jax
from jax import jit
from functools import partial

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit, RescaleAction, TransformObservation
from stable_baselines3.common.env_checker import check_env
from tianshou.data import Batch
from lib.environments.base import BaseEnvironment
from lib.models.wrappers import MarlModel

#temporary solution for xlb imports
sys.path.append(os.path.abspath('/home/pfischer/XLB'))
from my_flows.kolmogorov_2d import Kolmogorov_flow, Kolmogorov_flow_KBC, decaying_flow
from my_flows.helpers import get_kwargs, get_vorticity, get_velocity, get_kwargs4, get_moments, get_raw_moments
from src.utils import *

def jax2torch(x):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))

def torch2jax(x):
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))


#first Kolmogorov Environment
#State: vorticity
#Action: global alpha
#Reward: Scaled Vorticity Correlation
class KolmogorovEnvironment(BaseEnvironment, ABC):
    
    def __init__(self, kwargs1, kwargs2, step_factor=1):
        super().__init__()
        #Coarse-Grid-Simulation <- kwargs1
        self.kwargs1 = kwargs1
        self.cgs = Kolmogorov_flow(**kwargs1)
        self.omg = self.cgs.omega
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
       

        #Fine-Grid-Simulation <- kwargs2
        self.kwargs2 = kwargs2
        self.fgs = Kolmogorov_flow(**kwargs2)
        self.f2 = self.fgs.assign_fields_sharded()
        self.rho2, self.u2 = get_velocity(self.f2, self.fgs)
        
        #other stuff  
        self.factor = int(self.fgs.downsamplingFactor/self.cgs.downsamplingFactor)
        self.counter = 0
        self.observation_space = spaces.Box(low=-20, high=20, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float64)
        self.action_space = spaces.Box(low=0.9, high=1.2, shape=(1,), dtype=np.float32)
        self.step_factor = step_factor

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.counter = 0
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.fgs = Kolmogorov_flow(**self.kwargs2)
        self.cgs.omega = self.omg
        self.f1 = self.cgs.assign_fields_sharded()
        self.f2 = self.fgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.rho2, self.u2 = get_velocity(self.f2, self.fgs)

        v1 = vorticity_2d(self.u1, self.cgs.dx_eff)
        #v1 = get_vorticity(self.f1, self.cgs)
        #return v1, info
        return np.array(v1), {}
    
    def step(self, action):
        if not (np.any(self.action_space.low <= action) and np.any(action <= self.action_space.high)):
            print("WARNING: Action is not in action space")
            print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)
            

        # load in action and get rid of channel dimension
        #print(action.shape, self.action_space.shape)
        assert action.shape == self.action_space.shape

        self.cgs.omega = self.omg * np.float64(action[0])
        
        for i in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            for j in range(self.factor):
                self.f2, _ = self.fgs.step(self.f2, self.factor*self.counter+i, return_fpost=self.fgs.returnFpost)
            self.counter += 1

        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.rho2, self.u2 = get_velocity(self.f2, self.fgs)

        #v1 = get_vorticity(self.f1, self.cgs)
        #v2 = get_vorticity(self.f2, self.fgs)
        v1 = vorticity_2d(self.u1, self.cgs.dx_eff)
        v2 = vorticity_2d(self.u2, self.fgs.dx_eff)

        corr = np.corrcoef(v1.flatten(), v2.flatten())[0, 1]
        terminated = bool(corr<0.97)
        reward = (corr-0.97)/0.03
        #if terminated:
        #    reward -= 100

        return np.array(v1), reward, terminated, False, {}

    def render(self):
        #v1 = get_vorticity(self.f1, self.cgs)
        #v2 = get_vorticity(self.f2, self.fgs)
        v1 = vorticity_2d(self.u1, self.cgs.dx_eff)
        v2 = vorticity_2d(self.u2, self.fgs.dx_eff)
        #print(f"{1}:", np.mean(((v1-v2)**2))/np.mean((1e-10+(v1**2))))
        print(np.corrcoef(v1.flatten(), v2.flatten())[0, 1])
        # plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
        ax1.imshow(v1, vmin=-20, vmax=20, cmap=sn.cm.icefire)
        ax2.imshow(v2, vmin=-20, vmax=20, cmap=sn.cm.icefire)
        ax3.imshow((v1-v2)**2)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        #plot a common colorbar
        #fig.colorbar(ax1.imshow(v1, cmap=sn.cm.icefire), ax=[ax1, ax2], orientation='vertical')
        plt.show()




    def _play_and_get_eval_metrics(self,
                                   actor: MarlModel,
                                   ep_len: int = 21) -> Batch:

        # Play episode with and without RL
        _total_reward = 0.
        _obs, _ = self.reset()
        _no_rl_state = self.initial_condition.copy()

        # Calculation of true physical invariants:
        _total_mass = self.compute_total_mass(self.initial_condition)
        _total_momentum = self.compute_total_momentum(self.initial_condition, self.c_x, self.c_y)
        _total_kinetic_energy = self.compute_total_kinetic_energy(self.initial_condition, self.c_x, self.c_y)

        for i in range(ep_len):
            # Simulation with MARL in loop
            act_mean = actor.get_action_mean(_obs).clip(self.action_space.low.min(), self.action_space.high.max())

            # Compute RL adjusted update
            _obs, reward, _terminated, _, _ = self.step(act_mean.detach().cpu().numpy()[0])
            _total_reward += reward
            _no_rl_state = self.upwind_scheme_2d_step(_no_rl_state, self.dt, self.dx, self.dy, self.c_x, self.c_y)

        # Storing evaluation metrics in batch
        eval_metrics = Batch()
        eval_metrics['rews'] = _total_reward

        # Evaluation metric: Loss between RL adjusted state and ground truth at end of episode
        eval_metrics['mae_error'] = self.mean_agent_abs_error(self.state, self.gt_state)
        eval_metrics['mae_error_no_rl'] = self.mean_agent_abs_error(_no_rl_state, self.gt_state)

        # Calculation of physical invariants after simulation:
        _total_mass_rl = self.compute_total_mass(self.rl_adjusted_state)
        _total_momentum_rl = self.compute_total_momentum(self.rl_adjusted_state, self.c_x, self.c_y)
        _total_kinetic_energy_rl = self.compute_total_kinetic_energy(self.rl_adjusted_state, self.c_x, self.c_y)

        # Calculation of frequency spectra after simulation
        _, rl_energy_spectrum = np.log(self.compute_energy_spectrum(self.rl_adjusted_state))
        _, no_rl_energy_spectrum = np.log(self.compute_energy_spectrum(_no_rl_state))
        _, gt_energy_spectrum = np.log(self.compute_energy_spectrum(self.gt_state[::self.subsample, ::self.subsample]))

        # with rl errors
        eval_metrics['total_mass_error'] = np.abs((_total_mass_rl - _total_mass) / _total_mass)
        eval_metrics['total_momentum_error'] = np.abs((_total_momentum_rl - _total_momentum) / _total_momentum)
        eval_metrics['total_energy_error'] = np.abs(
            (_total_kinetic_energy_rl - _total_kinetic_energy) / _total_kinetic_energy
        )
        eval_metrics['energy_spectrum_error'] = self.energ_spec_error(rl_energy_spectrum, gt_energy_spectrum)
        eval_metrics['energy_spectrum_error_no_rl'] = self.energ_spec_error(no_rl_energy_spectrum, gt_energy_spectrum)
        # no rl errors
        eval_metrics['total_mass_error_no_rl'] = np.abs(
            (self.compute_total_mass(_no_rl_state) - _total_mass) / _total_mass
        )
        eval_metrics['total_momentum_error_no_rl'] = np.abs(
            (self.compute_total_momentum(_no_rl_state, self.c_x, self.c_y) - _total_momentum) / _total_momentum
        )
        eval_metrics['total_energy_error_no_rl'] = np.abs(
            (self.compute_total_kinetic_energy(_no_rl_state, self.c_x,
                                               self.c_y) - _total_kinetic_energy) / _total_kinetic_energy
        )
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


#copy of the above environment but with simple state output
# also allows for taking multiple solver steps for one environment interaction in the RL setup,
# by specifiying the argument "step_factor"
#State: Enerty Spectrum
#Action: global alpha
#Reward: Scaled Vorticity Correlation
class KolmogorovEnvironment2(BaseEnvironment, ABC):
    
    def __init__(self, kwargs1, kwargs2, step_factor=1, max_episode_steps=100):
        super().__init__()
        #Coarse-Grid-Simulation <- kwargs1
        self.kwargs1 = kwargs1
        self.cgs = Kolmogorov_flow(**kwargs1)
        self.omg = self.cgs.omega
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
       

        #Fine-Grid-Simulation <- kwargs2
        self.kwargs2 = kwargs2
        self.fgs = Kolmogorov_flow(**kwargs2)
        self.f2 = self.fgs.assign_fields_sharded()
        self.rho2, self.u2 = get_velocity(self.f2, self.fgs)
        
        #other stuff  
        self.factor = int(self.fgs.downsamplingFactor/self.cgs.downsamplingFactor)
        self.counter = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(int(self.cgs.nx/2 - 1),), dtype=np.float64)
        self.action_space = spaces.Box(low=0.9, high=1.1, shape=(1,), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps= int(step_factor*max_episode_steps)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.counter = 0
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.fgs = Kolmogorov_flow(**self.kwargs2)
        self.cgs.omega = self.omg
        self.f1 = self.cgs.assign_fields_sharded()
        self.f2 = self.fgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.rho2, self.u2 = get_velocity(self.f2, self.fgs)

        v1 = vorticity_2d(self.u1, self.cgs.dx_eff)
        #v1 = get_vorticity(self.f1, self.cgs)
        #return v1, info
        _, energy_spectrum = energy_spectrum_2d(self.u1)

        return energy_spectrum, {}
    
    def step(self, action):
        if not (np.any(self.action_space.low <= action) and np.any(action <= self.action_space.high)):
            print("WARNING: Action is not in action space")
            print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)
            

        # load in action and get rid of channel dimension
        #print(action.shape, self.action_space.shape)
        assert action.shape == self.action_space.shape

        self.cgs.omega = self.omg * np.float64(action[0])
        
        for i in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            for j in range(self.factor):
                self.f2, _ = self.fgs.step(self.f2, self.factor*self.counter+j, return_fpost=self.fgs.returnFpost)

            self.counter += 1

        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.rho2, self.u2 = get_velocity(self.f2, self.fgs)

        #v1 = get_vorticity(self.f1, self.cgs)
        #v2 = get_vorticity(self.f2, self.fgs)
        v1 = vorticity_2d(self.u1, self.cgs.dx_eff)
        v2 = vorticity_2d(self.u2, self.fgs.dx_eff)

        corr = np.corrcoef(v1.flatten(), v2.flatten())[0, 1]
        terminated = bool(corr<0.97)
        truncated = bool(self.counter>self.max_episode_steps)
        reward = (corr-0.97)/0.03
        
        #compute energy spectrum of cgs
        _, energy_spectrum = energy_spectrum_2d(self.u1)
        
        return energy_spectrum, reward, terminated, truncated, {}

    def render(self):
        #v1 = get_vorticity(self.f1, self.cgs)
        #v2 = get_vorticity(self.f2, self.fgs)
        v1 = vorticity_2d(self.u1, self.cgs.dx_eff)
        v2 = vorticity_2d(self.u2, self.fgs.dx_eff)
        #print(f"{1}:", np.mean(((v1-v2)**2))/np.mean((1e-10+(v1**2))))
        print(np.corrcoef(v1.flatten(), v2.flatten())[0, 1])
        # plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
        ax1.imshow(v1, vmin=-20, vmax=20, cmap=sn.cm.icefire)
        ax2.imshow(v2, vmin=-20, vmax=20, cmap=sn.cm.icefire)
        ax3.imshow((v1-v2)**2)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        #plot a common colorbar
        #fig.colorbar(ax1.imshow(v1, cmap=sn.cm.icefire), ax=[ax1, ax2], orientation='vertical')
        plt.show()

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2


# 3rd version of Kolmogorov environment -> allows for local actions
#State: ?
#Action: array of local alpha of same shape as domain
#Reward: Scaled Vorticity Correlation
class KolmogorovEnvironment3(BaseEnvironment, ABC):
    
    def __init__(self, kwargs1, kwargs2, step_factor=1, max_episode_steps=100):
        super().__init__()
        #Coarse-Grid-Simulation <- kwargs1
        self.kwargs1 = kwargs1
        self.cgs = Kolmogorov_flow(**kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
       

        #Fine-Grid-Simulation <- kwargs2
        self.kwargs2 = kwargs2
        self.fgs = Kolmogorov_flow(**kwargs2)
        self.f2 = self.fgs.assign_fields_sharded()
        self.rho2, self.u2 = get_velocity(self.f2, self.fgs)
        
        #other stuff  
        self.factor = int(self.fgs.downsamplingFactor/self.cgs.downsamplingFactor)
        self.counter = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(int(self.cgs.nx/2 - 1),), dtype=np.float64)
        self.action_space = spaces.Box(low=0.9, high=1.1, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps= int(step_factor*max_episode_steps)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.counter = 0
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.fgs = Kolmogorov_flow(**self.kwargs2)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.f2 = self.fgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.rho2, self.u2 = get_velocity(self.f2, self.fgs)

        v1 = vorticity_2d(self.u1, self.cgs.dx_eff)
        #v1 = get_vorticity(self.f1, self.cgs)
        #return v1, info
        _, energy_spectrum = energy_spectrum_2d(self.u1)

        return energy_spectrum, {}
    
    def step(self, action):
        # load in action and get rid of channel dimension
        #action = action[0]
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        #print(f"desired = {self.action_space.shape}, actual={action.shape}")
        #assert action.shape == self.action_space.shape

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            #print("WARNING: Action is not in action space")
            #print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)



        self.cgs.omega = np.copy(self.omg * action.reshape(self.omg.shape))
        
        for i in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            for j in range(self.factor):
                self.f2, _ = self.fgs.step(self.f2, self.factor*self.counter+j, return_fpost=self.fgs.returnFpost)

            self.counter += 1

        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.rho2, self.u2 = get_velocity(self.f2, self.fgs)

        #v1 = get_vorticity(self.f1, self.cgs)
        #v2 = get_vorticity(self.f2, self.fgs)
        v1 = vorticity_2d(self.u1, self.cgs.dx_eff)
        v2 = vorticity_2d(self.u2, self.fgs.dx_eff)

        corr = np.corrcoef(v1.flatten(), v2.flatten())[0, 1]
        terminated = bool(corr<0.97)
        truncated = bool(self.counter>self.max_episode_steps)
        reward = (corr-0.97)/0.03
        
        #compute energy spectrum of cgs
        _, energy_spectrum = energy_spectrum_2d(self.u1)
        
        return energy_spectrum, reward, terminated, truncated, {}

    def render(self):
        #v1 = get_vorticity(self.f1, self.cgs)
        #v2 = get_vorticity(self.f2, self.fgs)
        v1 = vorticity_2d(self.u1, self.cgs.dx_eff)
        v2 = vorticity_2d(self.u2, self.fgs.dx_eff)
        #print(f"{1}:", np.mean(((v1-v2)**2))/np.mean((1e-10+(v1**2))))
        print(np.corrcoef(v1.flatten(), v2.flatten())[0, 1])
        # plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
        ax1.imshow(v1, vmin=-20, vmax=20, cmap=sn.cm.icefire)
        ax2.imshow(v2, vmin=-20, vmax=20, cmap=sn.cm.icefire)
        ax3.imshow((v1-v2)**2)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        #plot a common colorbar
        #fig.colorbar(ax1.imshow(v1, cmap=sn.cm.icefire), ax=[ax1, ax2], orientation='vertical')
        plt.show()

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2


# 4rd version of Kolmogorov environment -> allows for local actions
#State: goes back to full image state -> vorticity for now
#Action: array of local alpha of same shape as domain
#Reward: Scaled Vorticity Correlation
class KolmogorovEnvironment4(BaseEnvironment, ABC):
    
    def __init__(self, kwargs1, kwargs2, step_factor=1, max_episode_steps=100):
        super().__init__()
        #Coarse-Grid-Simulation <- kwargs1
        self.kwargs1 = kwargs1
        self.cgs = Kolmogorov_flow(**kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
       

        #Fine-Grid-Simulation <- kwargs2
        self.kwargs2 = kwargs2
        self.fgs = Kolmogorov_flow(**kwargs2)
        self.f2 = self.fgs.assign_fields_sharded()
        self.rho2, self.u2 = get_velocity(self.f2, self.fgs)
        
        #other stuff  
        self.factor = int(self.fgs.downsamplingFactor/self.cgs.downsamplingFactor)
        self.counter = 0
        self.observation_space = spaces.Box(low=-20, high=20, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float64)
        self.action_space = spaces.Box(low=0.9, high=1.1, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps= int(step_factor*max_episode_steps)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.counter = 0
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.fgs = Kolmogorov_flow(**self.kwargs2)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.f2 = self.fgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.rho2, self.u2 = get_velocity(self.f2, self.fgs)

        v1 = vorticity_2d(self.u1, self.cgs.dx_eff)
        #_, energy_spectrum = energy_spectrum_2d(self.u1)
        return v1, {}
    
    def step(self, action):
        # load in action and get rid of channel dimension
        #action = action[0]
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        #print(f"desired = {self.action_space.shape}, actual={action.shape}")
        #assert action.shape == self.action_space.shape

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            #print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)



        self.cgs.omega = np.copy(self.omg * action.reshape(self.omg.shape))
        
        for i in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            for j in range(self.factor):
                self.f2, _ = self.fgs.step(self.f2, self.factor*self.counter+j, return_fpost=self.fgs.returnFpost)

            self.counter += 1

        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.rho2, self.u2 = get_velocity(self.f2, self.fgs)

        #v1 = get_vorticity(self.f1, self.cgs)
        #v2 = get_vorticity(self.f2, self.fgs)
        v1 = vorticity_2d(self.u1, self.cgs.dx_eff)
        v2 = vorticity_2d(self.u2, self.fgs.dx_eff)

        corr = np.corrcoef(v1.flatten(), v2.flatten())[0, 1]
        terminated = bool(corr<0.97)
        truncated = bool(self.counter>self.max_episode_steps)
        reward = (corr-0.97)/0.03
        
        #compute energy spectrum of cgs
        #_, energy_spectrum = energy_spectrum_2d(self.u1)
        
        return v1, reward, terminated, truncated, {}

    def render(self):
        #v1 = get_vorticity(self.f1, self.cgs)
        #v2 = get_vorticity(self.f2, self.fgs)
        v1 = vorticity_2d(self.u1, self.cgs.dx_eff)
        v2 = vorticity_2d(self.u2, self.fgs.dx_eff)
        #print(f"{1}:", np.mean(((v1-v2)**2))/np.mean((1e-10+(v1**2))))
        print(np.corrcoef(v1.flatten(), v2.flatten())[0, 1])
        # plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
        ax1.imshow(v1, vmin=-20, vmax=20, cmap=sn.cm.icefire)
        ax2.imshow(v2, vmin=-20, vmax=20, cmap=sn.cm.icefire)
        ax3.imshow((v1-v2)**2)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        #plot a common colorbar
        #fig.colorbar(ax1.imshow(v1, cmap=sn.cm.icefire), ax=[ax1, ax2], orientation='vertical')
        plt.show()

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2



# 5th version of Kolmogorov environment -> new state: velocity field
#State: velocity field
#Action: array of local alpha of same shape as domain
#Reward: mean squared error of velocity fields
class KolmogorovEnvironment5(BaseEnvironment, ABC):
    
    def __init__(self, kwargs1, kwargs2, step_factor=1, max_episode_steps=100):
        super().__init__()
        #Coarse-Grid-Simulation <- kwargs1
        self.kwargs1 = kwargs1
        self.cgs = Kolmogorov_flow(**kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
       

        #Fine-Grid-Simulation <- kwargs2
        self.kwargs2 = kwargs2
        self.fgs = Kolmogorov_flow(**kwargs2)
        self.f2 = self.fgs.assign_fields_sharded()
        self.rho2, self.u2 = get_velocity(self.f2, self.fgs)
        
        #other stuff  
        self.factor = int(self.fgs.downsamplingFactor/self.cgs.downsamplingFactor)
        self.counter = 0
        self.observation_space = spaces.Box(low=-20, high=20, shape=(self.cgs.nx, self.cgs.ny,2), dtype=np.float64)
        self.action_space = spaces.Box(low=0.9, high=1.1, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps= int(step_factor*max_episode_steps)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.counter = 0
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.fgs = Kolmogorov_flow(**self.kwargs2)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.f2 = self.fgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.rho2, self.u2 = get_velocity(self.f2, self.fgs)

        #v1 = vorticity_2d(self.u1, self.cgs.dx_eff)
        #_, energy_spectrum = energy_spectrum_2d(self.u1)
        return self.u1, {}
    
    def step(self, action):
        # load in action and get rid of channel dimension
        #action = action[0]
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        #print(f"desired = {self.action_space.shape}, actual={action.shape}")
        #assert action.shape == self.action_space.shape

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            #print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * action.reshape(self.omg.shape))
        for i in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            for j in range(self.factor):
                self.f2, _ = self.fgs.step(self.f2, self.factor*self.counter+j, return_fpost=self.fgs.returnFpost)

            self.counter += 1

        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.rho2, self.u2 = get_velocity(self.f2, self.fgs)

        reward = - ((self.u1 - self.u2)**2).mean() #mean squared error between velocity field of cgs and fgs
        #terminated = bool(corr<0.97)
        terminated = bool(reward < -0.01)
        truncated = bool(self.counter>self.max_episode_steps)
        
        #compute energy spectrum of cgs
        #_, energy_spectrum = energy_spectrum_2d(self.u1)
        
        return self.u1, reward, terminated, truncated, {}

    def render(self):
        #v1 = get_vorticity(self.f1, self.cgs)
        #v2 = get_vorticity(self.f2, self.fgs)
        v1 = vorticity_2d(self.u1, self.cgs.dx_eff)
        v2 = vorticity_2d(self.u2, self.fgs.dx_eff)
        #print(f"{1}:", np.mean(((v1-v2)**2))/np.mean((1e-10+(v1**2))))
        print("Correlation:", np.corrcoef(v1.flatten(), v2.flatten())[0, 1])
        print("MSE:", ((self.u1 - self.u2)**2).mean())
        print("NMSE:", ((self.u1 - self.u2)**2).sum()/((self.u2)**2).sum())
        # plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
        ax1.imshow(v1, vmin=-20, vmax=20, cmap=sn.cm.icefire)
        ax2.imshow(v2, vmin=-20, vmax=20, cmap=sn.cm.icefire)
        ax3.imshow((v1-v2)**2)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        #plot a common colorbar
        #fig.colorbar(ax1.imshow(v1, cmap=sn.cm.icefire), ax=[ax1, ax2], orientation='vertical')
        plt.show()

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2
    



# 6th environemnt. Change usage a bit to load velocity field of fgs from file
# also allow for multiple initializations from different burn ins depending on a seed argument 

# path to the initialization files
INIT_PATH = "/home/pfischer/XLB/vel_init/"
FGS_DATA_PATH = "/home/pfischer/XLB/fgs_data/"

class KolmogorovEnvironment6(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=100, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102])):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.fgs_dump_path = FGS_DATA_PATH + f"re1000_T18_N2048_S{self.sampled_seed}_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs(u0_path=u0_path, rho0_path=rho0_path, desired_time=1.06, lamb=cgs_lamb) #cgs is 128x128
        self.kwargs2, _, _, _ = get_kwargs(u0_path=u0_path, rho0_path=rho0_path, desired_time=1.06, lamb=fgs_lamb)

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-20, high=20, shape=(self.cgs.nx, self.cgs.ny,2), dtype=np.float64)
        self.action_space = spaces.Box(low=0.9, high=1.1, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps,endTime1])

        #FGS
        self.u2 = self._load_u2()

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.fgs_dump_path = FGS_DATA_PATH + f"re1000_T18_N2048_S{self.sampled_seed}_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.u2 = self._load_u2()
        return self.u1, {}
    
    def step(self, action):
        # load in action and get rid of channel dimension
        #action = action[0]
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            #print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * action.reshape(self.omg.shape))

        for i in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1

        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.u2 = self._load_u2()

        reward = - ((self.u1 - self.u2)**2).mean() #mean squared error between velocity field of cgs and fgs
        #terminated = bool(corr<0.97)
        terminated = bool(reward < -0.01)
        truncated = bool(self.counter>self.max_episode_steps)
        
        return self.u1, 1., terminated, truncated, {}

    def render(self):
        v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        print("Correlation:", np.corrcoef(v1.flatten(), v2.flatten())[0, 1])
        print("MSE:", ((self.u1 - self.u2)**2).mean())
        print("NMSE:", ((self.u1 - self.u2)**2).sum()/((self.u2)**2).sum())
        print("pointwise relative mse:", (((self.u1 - self.u2)**2)/((self.u2)**2)+1e-6).mean())
        # Plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        im1 = ax1.imshow(v1, vmin=-20, vmax=20, cmap=sn.cm.icefire)
        im2 = ax2.imshow(v2, vmin=-20, vmax=20, cmap=sn.cm.icefire)
        im3 = ax3.imshow((v1 - v2)**2, vmin=0, vmax=1)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        # Create a colorbar for the third plot
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax)
        plt.show()

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2
    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])



# change to a more informative advantage reward by checking if the action taken is better 
# than a trivial action. Compare to yans reward
class KolmogorovEnvironment7(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=1535, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102])):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.fgs_dump_path = FGS_DATA_PATH + f"re1000_T18_N2048_S{self.sampled_seed}_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs(u0_path=u0_path, rho0_path=rho0_path, desired_time=1.06, lamb=cgs_lamb) #cgs is 128x128
        self.kwargs2, _, _, _ = get_kwargs(u0_path=u0_path, rho0_path=rho0_path, desired_time=1.06, lamb=fgs_lamb)

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])

        #reference cgs
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)
        self.f1_ref = self.cgs_ref.assign_fields_sharded()
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-20, high=20, shape=(self.cgs.nx, self.cgs.ny,2), dtype=np.float64)
        self.action_space = spaces.Box(low=0.9, high=1.1, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps,endTime1])
        self.trivial_action = np.ones((self.cgs.nx, self.cgs.ny))

        #FGS
        self.u2 = self._load_u2()
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.fgs_dump_path = FGS_DATA_PATH + f"re1000_T18_N2048_S{self.sampled_seed}_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        self.u2 = self._load_u2()
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])

        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)

        return self.u1, {}
    
    def step(self, action):
        # load in action and get rid of channel dimension
        #action = action[0]
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            #print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * action.reshape(self.omg.shape))
        self.f1_ref = np.copy(self.f1)
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.f1_ref,_ = self.cgs_ref.step(self.f1_ref, self.counter, return_fpost=self.cgs_ref.returnFpost)
            self.counter += 1

        self.u2 = self._load_u2()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        _, self.u1_ref = get_velocity(self.f1_ref, self.cgs_ref)

        err1 = ((self.u2-self.u1_ref)**2).mean()
        err2 = ((self.u2-self.u1)**2).mean()
        #print(f"err1: {err1}, err2: {err2}, diff: {err1-err2}")
        #reward = 1e8*((self.u2-self.u1_ref)**2 - (self.u2-self.u1)**2).sum()
        reward = np.exp(err1-err2)
        
        #also compute the correlation of voriticity as it is a good stopping criteria
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        corr = np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1]
        terminated = bool(corr<0.97)
        truncated = bool(self.counter>=self.max_episode_steps)
        
        return self.u1, reward, terminated, truncated, {}

    def render(self):
        print("Correlation:", np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1])
        print("MSE:", ((self.u1 - self.u2)**2).mean())
        print("NMSE:", ((self.u1 - self.u2)**2).sum()/((self.u2)**2).sum())
        print("pointwise relative mse:", (((self.u1 - self.u2)**2)/((self.u2)**2)+1e-6).mean())
        # Plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        im1 = ax1.imshow(self.v1, vmin=-20, vmax=20, cmap=sn.cm.icefire)
        im2 = ax2.imshow(self.v2, vmin=-20, vmax=20, cmap=sn.cm.icefire)
        im3 = ax3.imshow((self.v1 - self.v2)**2, vmin=0, vmax=1)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        # Create a colorbar for the third plot
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax)
        plt.show()

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2
    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])
    

#    def close(self):
#        # Clear GPU tensors if needed
#        if torch.cuda.is_available():
#            torch.cuda.empty_cache()
#        
#        # Explicitly delete large objects
#        del self.cgs
#        del self.cgs_ref
#        del self.f1
#        del self.u1
#        del self.u2
#        del self.v1
#        del self.v2
#        del self.omg
#        
#        # Call garbage collector to free up memory
#        gc.collect()
#
#    # Optionally, you can also implement __del__ to ensure cleanup
#    def __del__(self):
#        self.close()



#change to local rewards 
class KolmogorovEnvironment8(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=1535, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102])):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.fgs_dump_path = FGS_DATA_PATH + f"re1000_T18_N2048_S{self.sampled_seed}_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs(u0_path=u0_path, rho0_path=rho0_path, desired_time=1.06, lamb=cgs_lamb) #cgs is 128x128
        self.kwargs2, _, _, _ = get_kwargs(u0_path=u0_path, rho0_path=rho0_path, desired_time=1.06, lamb=fgs_lamb)

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])

        #reference cgs
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)
        self.f1_ref = self.cgs_ref.assign_fields_sharded()
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-2, high=2, shape=(self.cgs.nx, self.cgs.ny,2), dtype=np.float64)
        self.action_space = spaces.Box(low=0.995, high=1.005, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])
        self.trivial_action = np.ones((self.cgs.nx, self.cgs.ny))

        #FGS
        self.u2 = self._load_u2()
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.fgs_dump_path = FGS_DATA_PATH + f"re1000_T18_N2048_S{self.sampled_seed}_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        self.u2 = self._load_u2()
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)
        return self.u1, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            #print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * action.reshape(self.omg.shape))
        self.f1_ref = np.copy(self.f1)
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.f1_ref,_ = self.cgs_ref.step(self.f1_ref, self.counter, return_fpost=self.cgs_ref.returnFpost)
            self.counter += 1

        self.u2 = self._load_u2()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        _, self.u1_ref = get_velocity(self.f1_ref, self.cgs_ref)

        err1 = ((self.u2-self.u1_ref)**2).mean(axis=2)
        err2 = ((self.u2-self.u1)**2).mean(axis=2)
        #reward = np.exp(1e5*(err1-err2))
        reward = 1e4*(err1-err2)

        #also compute the correlation of voriticity as it is a good stopping criteria
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        corr = np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1]
        terminated = bool(corr<0.97)
        truncated = bool(self.counter>=self.max_episode_steps)

        return self.u1, reward, terminated, truncated, {}

    def render(self):
        print("Correlation:", np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1])
        print("MSE:", ((self.u1 - self.u2)**2).mean())
        print("NMSE:", ((self.u1 - self.u2)**2).sum()/((self.u2)**2).sum())
        print("pointwise relative mse:", (((self.u1 - self.u2)**2)/((self.u2)**2)+1e-6).mean())
        # Plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        im1 = ax1.imshow(self.v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = ax2.imshow(self.v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im3 = ax3.imshow((self.v1 - self.v2)**2, vmin=0, vmax=1)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        # Create a colorbar for the third plot
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax)
        plt.show()

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2
    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])



FGS_DATA_PATH_2 = "/home/pfischer/XLB/fgs2_data/"
#change to Re 10000
class KolmogorovEnvironment9(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=1535, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.fgs_dump_path = FGS_DATA_PATH_2 + f"re{self.Re}_T22_N2048_S{self.sampled_seed}_U1_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs3(u0_path=u0_path, rho0_path=rho0_path, k=33, lamb=cgs_lamb, Re=self.Re) #cgs is 128x128
        self.kwargs2, _, _, _ = get_kwargs3(u0_path=u0_path, rho0_path=rho0_path, k=33, lamb=fgs_lamb, Re=self.Re)

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])

        #reference cgs
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)
        self.f1_ref = self.cgs_ref.assign_fields_sharded()
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-2, high=2, shape=(self.cgs.nx, self.cgs.ny,2), dtype=np.float64)
        self.action_space = spaces.Box(low=0.995, high=1.005, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])
        self.trivial_action = np.ones((self.cgs.nx, self.cgs.ny))

        #FGS
        self.u2 = self._load_u2()
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.fgs_dump_path = FGS_DATA_PATH_2 + f"re{self.Re}_T22_N2048_S{self.sampled_seed}_U1_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        self.u2 = self._load_u2()
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)
        return self.u1, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            #print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * action.reshape(self.omg.shape))
        self.f1_ref = np.copy(self.f1)
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.f1_ref,_ = self.cgs_ref.step(self.f1_ref, self.counter, return_fpost=self.cgs_ref.returnFpost)
            self.counter += 1

        self.u2 = self._load_u2()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        _, self.u1_ref = get_velocity(self.f1_ref, self.cgs_ref)

        err1 = ((self.u2-self.u1_ref)**2).mean(axis=2)
        err2 = ((self.u2-self.u1)**2).mean(axis=2)
        #reward = np.exp(1e5*(err1-err2))
        reward = 1e4*(err1-err2)

        #also compute the correlation of voriticity as it is a good stopping criteria
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        corr = np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1]
        terminated = bool(corr<0.97)
        truncated = bool(self.counter>=self.max_episode_steps)

        return self.u1, reward, terminated, truncated, {}

    def render(self):
        print("Correlation:", np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1])
        print("MSE:", ((self.u1 - self.u2)**2).mean())
        print("NMSE:", ((self.u1 - self.u2)**2).sum()/((self.u2)**2).sum())
        print("pointwise relative mse:", (((self.u1 - self.u2)**2)/((self.u2)**2)+1e-6).mean())
        # Plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        im1 = ax1.imshow(self.v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = ax2.imshow(self.v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im3 = ax3.imshow((self.v1 - self.v2)**2, vmin=0, vmax=1)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        # Create a colorbar for the third plot
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax)
        plt.show()

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2
    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])



# change to populations f_i  as states
class KolmogorovEnvironment10(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=1535, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.fgs_dump_path = FGS_DATA_PATH_2 + f"re{self.Re}_T18_N2048_S{self.sampled_seed}_U1_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=18, lamb=cgs_lamb, Re=self.Re) #cgs
        self.kwargs2, _, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=18, lamb=fgs_lamb, Re=self.Re) #fgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])

        #reference cgs
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)
        self.f1_ref = self.cgs_ref.assign_fields_sharded()
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.cgs.nx, self.cgs.ny, 9), dtype=np.float64)
        self.action_space = spaces.Box(low=0.995, high=1.005, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])
        self.trivial_action = np.ones((self.cgs.nx, self.cgs.ny))

        #FGS
        self.u2 = self._load_u2()
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.fgs_dump_path = FGS_DATA_PATH_2 + f"re{self.Re}_T18_N2048_S{self.sampled_seed}_U1_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        self.u2 = self._load_u2()
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)
        return self.f1, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            #print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * action.reshape(self.omg.shape))
        self.f1_ref = np.copy(self.f1)
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.f1_ref,_ = self.cgs_ref.step(self.f1_ref, self.counter, return_fpost=self.cgs_ref.returnFpost)
            self.counter += 1

        self.u2 = self._load_u2()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        _, self.u1_ref = get_velocity(self.f1_ref, self.cgs_ref)

        err1 = ((self.u2-self.u1_ref)**2).mean(axis=2)
        err2 = ((self.u2-self.u1)**2).mean(axis=2)
        #reward = np.exp(1e5*(err1-err2))
        reward = 1e4*(err1-err2)

        #also compute the correlation of voriticity as it is a good stopping criteria
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        corr = np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1]
        terminated = bool(corr<0.97)
        truncated = bool(self.counter>=self.max_episode_steps)

        return self.f1, reward, terminated, truncated, {}

    def render(self):
        print("Correlation:", np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1])
        print("MSE:", ((self.u1 - self.u2)**2).mean())
        print("NMSE:", ((self.u1 - self.u2)**2).sum()/((self.u2)**2).sum())
        print("pointwise relative mse:", (((self.u1 - self.u2)**2)/((self.u2)**2)+1e-6).mean())
        # Plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        im1 = ax1.imshow(self.v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = ax2.imshow(self.v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im3 = ax3.imshow((self.v1 - self.v2)**2, vmin=0, vmax=1)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        # Create a colorbar for the third plot
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax)
        plt.show()

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2
    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])



# change to populations f_i_neq  as states
class KolmogorovEnvironment11(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=1535, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.fgs_dump_path = FGS_DATA_PATH_2 + f"re{self.Re}_T18_N2048_S{self.sampled_seed}_U1_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=18, lamb=cgs_lamb, Re=self.Re) #cgs
        self.kwargs2, _, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=18, lamb=fgs_lamb, Re=self.Re) #fgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])

        #reference cgs
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)
        self.f1_ref = self.cgs_ref.assign_fields_sharded()
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.cgs.nx, self.cgs.ny, 9), dtype=np.float64)
        self.action_space = spaces.Box(low=0.995, high=1.005, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])
        self.trivial_action = np.ones((self.cgs.nx, self.cgs.ny))

        #FGS
        self.u2 = self._load_u2()
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.fgs_dump_path = FGS_DATA_PATH_2 + f"re{self.Re}_T18_N2048_S{self.sampled_seed}_U1_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()

        rho, u = self.cgs.update_macroscopic(self.f1)
        feq = self.cgs.equilibrium(rho, u, cast_output=False)
        self.fneq = self.f1 - feq

        _, self.u1 = get_velocity(self.f1, self.cgs)
        self.u2 = self._load_u2()
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)

        return self.fneq, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            #print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * action.reshape(self.omg.shape))
        self.f1_ref = np.copy(self.f1)
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.f1_ref,_ = self.cgs_ref.step(self.f1_ref, self.counter, return_fpost=self.cgs_ref.returnFpost)
            self.counter += 1

        self.u2 = self._load_u2()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        _, self.u1_ref = get_velocity(self.f1_ref, self.cgs_ref)

        err1 = ((self.u2-self.u1_ref)**2).mean(axis=2)
        err2 = ((self.u2-self.u1)**2).mean(axis=2)
        reward = ((err1-err2))/1.4e-05 + 1

        #compute neq filed
        rho, u = self.cgs.update_macroscopic(self.f1)
        feq = self.cgs.equilibrium(rho, u, cast_output=False)
        self.fneq = self.f1 - feq

        #also compute the correlation of voriticity as it is a good stopping criteria
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        corr = np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1]
        terminated = bool(corr<0.98)
        truncated = bool(self.counter>=self.max_episode_steps)

        #add a negative termination reward
        if terminated:
            reward -= 100
        #add a truncation reward to account for missing future rewards
        if truncated:
            reward += 1/(1-0.97)

        return self.fneq, reward, terminated, truncated, {}

    def render(self):
        print("Correlation:", np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1])
        print("MSE:", ((self.u1 - self.u2)**2).mean())
        print("NMSE:", ((self.u1 - self.u2)**2).sum()/((self.u2)**2).sum())
        print("pointwise relative mse:", (((self.u1 - self.u2)**2)/((self.u2)**2)+1e-6).mean())
        # Plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        im1 = ax1.imshow(self.v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = ax2.imshow(self.v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im3 = ax3.imshow((self.v1 - self.v2)**2, vmin=0, vmax=1)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        # Create a colorbar for the third plot
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax)
        plt.show()

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2
    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])



# test environemnt with KBC sim as actor to check reward scale and desired reward
class KolmogorovEnvironment12(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=1535, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.fgs_dump_path = FGS_DATA_PATH_2 + f"re{self.Re}_T18_N2048_S{self.sampled_seed}_U1_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=18, lamb=cgs_lamb, Re=self.Re) #cgs
        self.kwargs2, _, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=18, lamb=fgs_lamb, Re=self.Re) #fgs

        #CGS
        self.cgs = Kolmogorov_flow_KBC(**self.kwargs1)
        #self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        #self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])

        #reference cgs
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)
        self.f1_ref = self.cgs_ref.assign_fields_sharded()
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.cgs.nx, self.cgs.ny, 9), dtype=np.float64)
        self.action_space = spaces.Box(low=0.995, high=1.005, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])
        self.trivial_action = np.ones((self.cgs.nx, self.cgs.ny))

        #FGS
        self.u2 = self._load_u2()
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.fgs_dump_path = FGS_DATA_PATH_2 + f"re{self.Re}_T18_N2048_S{self.sampled_seed}_U1_dump/"
        self.cgs = Kolmogorov_flow_KBC(**self.kwargs1)
        #self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()

        rho, u = self.cgs.update_macroscopic(self.f1)
        feq = self.cgs.equilibrium(rho, u, cast_output=False)
        self.fneq = self.f1 - feq

        _, self.u1 = get_velocity(self.f1, self.cgs)
        self.u2 = self._load_u2()
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)

        return self.fneq, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            #print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        #self.cgs.omega = np.copy(self.omg * action.reshape(self.omg.shape))
        self.f1_ref = np.copy(self.f1)
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.f1_ref,_ = self.cgs_ref.step(self.f1_ref, self.counter, return_fpost=self.cgs_ref.returnFpost)
            self.counter += 1

        self.u2 = self._load_u2()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        _, self.u1_ref = get_velocity(self.f1_ref, self.cgs_ref)

        err1 = ((self.u2-self.u1_ref)**2).mean(axis=2)
        err2 = ((self.u2-self.u1)**2).mean(axis=2)
        #reward = np.exp(1e4*(err1-err2))
        #reward = np.exp(1e4*(err1-err2))
        #reward = (err1-err2)
        reward = ((err1-err2))/1.4e-05 + 1

        #compute neq filed
        rho, u = self.cgs.update_macroscopic(self.f1)
        feq = self.cgs.equilibrium(rho, u, cast_output=False)
        self.fneq = self.f1 - feq

        #also compute the correlation of voriticity as it is a good stopping criteria
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        corr = np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1]
        terminated = bool(corr<0.97)
        truncated = bool(self.counter>=self.max_episode_steps)
        if truncated:
            #add a truncation reward to account for missing future rewards
            reward += 1/(1-0.97)

        return self.fneq, reward, terminated, truncated, {}

    def render(self):
        print("Correlation:", np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1])
        print("MSE:", ((self.u1 - self.u2)**2).mean())
        print("NMSE:", ((self.u1 - self.u2)**2).sum()/((self.u2)**2).sum())
        print("pointwise relative mse:", (((self.u1 - self.u2)**2)/((self.u2)**2)+1e-6).mean())
        # Plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        im1 = ax1.imshow(self.v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = ax2.imshow(self.v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im3 = ax3.imshow((self.v1 - self.v2)**2, vmin=0, vmax=1)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        # Create a colorbar for the third plot
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax)
        plt.show()

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2
    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])


# change to scalar reward for central learning
class KolmogorovEnvironment13(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=1535, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.fgs_dump_path = FGS_DATA_PATH_2 + f"re{self.Re}_T18_N2048_S{self.sampled_seed}_U1_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=18, lamb=cgs_lamb, Re=self.Re) #cgs
        self.kwargs2, _, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=18, lamb=fgs_lamb, Re=self.Re) #fgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])

        #reference cgs
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)
        self.f1_ref = self.cgs_ref.assign_fields_sharded()
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.cgs.nx, self.cgs.ny, 9), dtype=np.float64)
        self.action_space = spaces.Box(low=0.995, high=1.005, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])
        self.trivial_action = np.ones((self.cgs.nx, self.cgs.ny))

        #FGS
        self.u2 = self._load_u2()
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.fgs_dump_path = FGS_DATA_PATH_2 + f"re{self.Re}_T18_N2048_S{self.sampled_seed}_U1_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()

        rho, u = self.cgs.update_macroscopic(self.f1)
        feq = self.cgs.equilibrium(rho, u, cast_output=False)
        self.fneq = self.f1 - feq

        _, self.u1 = get_velocity(self.f1, self.cgs)
        self.u2 = self._load_u2()
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)

        return self.fneq, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            #print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * action.reshape(self.omg.shape))
        self.f1_ref = np.copy(self.f1)
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.f1_ref,_ = self.cgs_ref.step(self.f1_ref, self.counter, return_fpost=self.cgs_ref.returnFpost)
            self.counter += 1

        self.u2 = self._load_u2()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        _, self.u1_ref = get_velocity(self.f1_ref, self.cgs_ref)

        err1 = ((self.u2-self.u1_ref)**2).mean()
        err2 = ((self.u2-self.u1)**2).mean()
        reward = ((err1-err2))/1.4e-05 + 1

        #compute neq filed
        rho, u = self.cgs.update_macroscopic(self.f1)
        feq = self.cgs.equilibrium(rho, u, cast_output=False)
        self.fneq = self.f1 - feq

        #also compute the correlation of voriticity as it is a good stopping criteria
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        corr = np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1]
        terminated = bool(corr<0.98)
        truncated = bool(self.counter>=self.max_episode_steps)

        #add a negative termination reward
        if terminated:
            reward -= 100
        #add a truncation reward to account for missing future rewards
        if truncated:
            reward += 1/(1-0.97)

        return self.fneq, reward, terminated, truncated, {}

    def render(self):
        print("Correlation:", np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1])
        print("MSE:", ((self.u1 - self.u2)**2).mean())
        print("NMSE:", ((self.u1 - self.u2)**2).sum()/((self.u2)**2).sum())
        print("pointwise relative mse:", (((self.u1 - self.u2)**2)/((self.u2)**2)+1e-6).mean())
        # Plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        im1 = ax1.imshow(self.v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = ax2.imshow(self.v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im3 = ax3.imshow((self.v1 - self.v2)**2, vmin=0, vmax=1)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        # Create a colorbar for the third plot
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax)
        plt.show()

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2
    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])




# change back to states f_i instead of f_eq
# also scale action internally
class KolmogorovEnvironment14(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=1587, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.fgs_dump_path = FGS_DATA_PATH_2 + f"re{self.Re}_T18_N2048_S{self.sampled_seed}_U1_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=18, lamb=cgs_lamb, Re=self.Re) #cgs
        self.kwargs2, _, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=18, lamb=fgs_lamb, Re=self.Re) #fgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])

        #reference cgs
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)
        self.f1_ref = self.cgs_ref.assign_fields_sharded()
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.cgs.nx, self.cgs.ny, 9), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.0025, high=0.0025, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])
        self.trivial_action = np.ones((self.cgs.nx, self.cgs.ny))

        #FGS
        self.u2 = self._load_u2()
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.fgs_dump_path = FGS_DATA_PATH_2 + f"re{self.Re}_T18_N2048_S{self.sampled_seed}_U1_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()

        rho, u = self.cgs.update_macroscopic(self.f1)
        feq = self.cgs.equilibrium(rho, u, cast_output=False)
        self.fneq = self.f1 - feq

        _, self.u1 = get_velocity(self.f1, self.cgs)
        self.u2 = self._load_u2()
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)

        return self.fneq, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            #print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * (1+action.reshape(self.omg.shape)))
        self.f1_ref = np.copy(self.f1)
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.f1_ref,_ = self.cgs_ref.step(self.f1_ref, self.counter, return_fpost=self.cgs_ref.returnFpost)
            self.counter += 1

        self.u2 = self._load_u2()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        _, self.u1_ref = get_velocity(self.f1_ref, self.cgs_ref)

        err1 = ((self.u2-self.u1_ref)**2).mean()
        err2 = ((self.u2-self.u1)**2).mean()
        reward = ((err1-err2))/1.4e-05 + 1

        #compute neq filed
        rho, u = self.cgs.update_macroscopic(self.f1)
        feq = self.cgs.equilibrium(rho, u, cast_output=False)
        self.fneq = self.f1 - feq

        #also compute the correlation of voriticity as it is a good stopping criteria
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        corr = np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1]
        terminated = bool(corr<0.97)
        truncated = bool(self.counter>=self.max_episode_steps)

        #add a negative termination reward
        if terminated:
            reward -= 100
        #add a truncation reward to account for missing future rewards
        if truncated:
            reward += 1/(1-0.97)

        return self.fneq, reward, terminated, truncated, {}

    def render(self):
        print("Correlation:", np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1])
        print("MSE:", ((self.u1 - self.u2)**2).mean())
        print("NMSE:", ((self.u1 - self.u2)**2).sum()/((self.u2)**2).sum())
        print("pointwise relative mse:", (((self.u1 - self.u2)**2)/((self.u2)**2)+1e-6).mean())
        # Plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        im1 = ax1.imshow(self.v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = ax2.imshow(self.v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im3 = ax3.imshow((self.v1 - self.v2)**2, vmin=0, vmax=1)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        # Create a colorbar for the third plot
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax)
        plt.show()

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2
    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])



FGS_DATA_PATH_3 = "/home/pfischer/XLB/fgs3_data/"
# allow for longer episodes
class KolmogorovEnvironment15(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N2048_S{self.sampled_seed}_U1_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs
        self.kwargs2, _, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=fgs_lamb, Re=self.Re) #fgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])

        #reference cgs
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)
        self.f1_ref = self.cgs_ref.assign_fields_sharded()
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.cgs.nx, self.cgs.ny, 9), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.0025, high=0.0025, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])
        self.trivial_action = np.ones((self.cgs.nx, self.cgs.ny))

        #FGS
        self.u2 = self._load_u2()
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N2048_S{self.sampled_seed}_U1_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()

        rho, u = self.cgs.update_macroscopic(self.f1)
        feq = self.cgs.equilibrium(rho, u, cast_output=False)
        self.fneq = self.f1 - feq

        _, self.u1 = get_velocity(self.f1, self.cgs)
        self.u2 = self._load_u2()
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)

        return self.fneq, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            #print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * (1+action.reshape(self.omg.shape)))
        self.f1_ref = np.copy(self.f1)
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.f1_ref,_ = self.cgs_ref.step(self.f1_ref, self.counter, return_fpost=self.cgs_ref.returnFpost)
            self.counter += 1

        self.u2 = self._load_u2()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        _, self.u1_ref = get_velocity(self.f1_ref, self.cgs_ref)

        err1 = ((self.u2-self.u1_ref)**2).mean()
        err2 = ((self.u2-self.u1)**2).mean()
        reward = ((err1-err2))/1.4e-05 + 1

        #compute neq filed
        rho, u = self.cgs.update_macroscopic(self.f1)
        feq = self.cgs.equilibrium(rho, u, cast_output=False)
        self.fneq = self.f1 - feq

        #also compute the correlation of voriticity as it is a good stopping criteria
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        corr = np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1]
        #terminated = bool(corr<0.97)
        terminated = np.any(np.isnan(self.v1))
        truncated = bool(self.counter>=self.max_episode_steps)

        #add a negative termination reward
        if terminated:
            reward -= 100
        #add a truncation reward to account for missing future rewards
        if truncated:
            reward += 1/(1-0.97)

        return self.fneq, reward, terminated, truncated, {}

    def render(self):
        print("Correlation:", np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1])
        print("MSE:", ((self.u1 - self.u2)**2).mean())
        print("NMSE:", ((self.u1 - self.u2)**2).sum()/((self.u2)**2).sum())
        print("pointwise relative mse:", (((self.u1 - self.u2)**2)/((self.u2)**2)+1e-6).mean())
        # Plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        im1 = ax1.imshow(self.v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = ax2.imshow(self.v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im3 = ax3.imshow((self.v1 - self.v2)**2)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        # Create a colorbar for the third plot
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax)
        plt.show()

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2
    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])


# new termination argumenf f<0 or f>1
class KolmogorovEnvironment16(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N2048_S{self.sampled_seed}_U1_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs
        self.kwargs2, _, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=fgs_lamb, Re=self.Re) #fgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])

        #reference cgs
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)
        self.f1_ref = self.cgs_ref.assign_fields_sharded()
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.cgs.nx, self.cgs.ny, 9), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.0025, high=0.0025, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])
        self.trivial_action = np.ones((self.cgs.nx, self.cgs.ny))

        #FGS
        self.u2 = self._load_u2()
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N2048_S{self.sampled_seed}_U1_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()

        rho, u = self.cgs.update_macroscopic(self.f1)
        feq = self.cgs.equilibrium(rho, u, cast_output=False)
        self.fneq = self.f1 - feq

        _, self.u1 = get_velocity(self.f1, self.cgs)
        self.u2 = self._load_u2()
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        self.cgs_ref = Kolmogorov_flow(**self.kwargs1)

        return self.fneq, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            #print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * (1+action.reshape(self.omg.shape)))
        self.f1_ref = np.copy(self.f1)
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.f1_ref,_ = self.cgs_ref.step(self.f1_ref, self.counter, return_fpost=self.cgs_ref.returnFpost)
            self.counter += 1

        self.u2 = self._load_u2()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        _, self.u1_ref = get_velocity(self.f1_ref, self.cgs_ref)

        err1 = ((self.u2-self.u1_ref)**2).mean()
        err2 = ((self.u2-self.u1)**2).mean()
        reward = ((err1-err2))/1.4e-05 + 1

        #compute neq filed
        rho, u = self.cgs.update_macroscopic(self.f1)
        feq = self.cgs.equilibrium(rho, u, cast_output=False)
        self.fneq = self.f1 - feq

        #also compute the correlation of voriticity as it is a good stopping criteria
        self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        corr = np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1]
        #terminated = bool(corr<0.97)
        #terminated = np.any(np.isnan(self.v1))
        terminated = np.any(self.f1 < 0) or np.any(self.f1 > 1)
        truncated = bool(self.counter>=self.max_episode_steps)

        #add a negative termination reward
        if terminated:
            reward -= 100
        #add a truncation reward to account for missing future rewards
        if truncated:
            reward += 1/(1-0.97)

        return self.fneq, reward, terminated, truncated, {}

    def render(self):
        print("Correlation:", np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1])
        print("MSE:", ((self.u1 - self.u2)**2).mean())
        print("NMSE:", ((self.u1 - self.u2)**2).sum()/((self.u2)**2).sum())
        print("pointwise relative mse:", (((self.u1 - self.u2)**2)/((self.u2)**2)+1e-6).mean())
        # Plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        im1 = ax1.imshow(self.v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = ax2.imshow(self.v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im3 = ax3.imshow((self.v1 - self.v2)**2)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        # Create a colorbar for the third plot
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax)
        plt.show()

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2
    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])
    

# remove reference cgs and even fgs?!
class KolmogorovEnvironment17(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N2048_S{self.sampled_seed}_U1_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs
        #self.kwargs2, _, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=fgs_lamb, Re=self.Re) #fgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        #self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])

        #reference cgs
        #self.cgs_ref = Kolmogorov_flow(**self.kwargs1)
        #self.f1_ref = self.cgs_ref.assign_fields_sharded()
        
        #other stuff  
        #self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.cgs.nx, self.cgs.ny, 9), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.0025, high=0.0025, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])
        self.trivial_action = np.ones((self.cgs.nx, self.cgs.ny))

        #FGS
        #self.u2 = self._load_u2()
        #self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        #self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N2048_S{self.sampled_seed}_U1_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()

        #rho, u = self.cgs.update_macroscopic(self.f1)
        #feq = self.cgs.equilibrium(rho, u, cast_output=False)
        #self.fneq = self.f1 - feq

        _, self.u1 = get_velocity(self.f1, self.cgs)
        #self.u2 = self._load_u2()
        #self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        #self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        #self.cgs_ref = Kolmogorov_flow(**self.kwargs1)

        return self.f1, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            #print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * (1+action.reshape(self.omg.shape)))
        #self.f1_ref = np.copy(self.f1)
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            #self.f1_ref,_ = self.cgs_ref.step(self.f1_ref, self.counter, return_fpost=self.cgs_ref.returnFpost)
            self.counter += 1

        #self.u2 = self._load_u2()
        #_, self.u1 = get_velocity(self.f1, self.cgs)
        #_, self.u1_ref = get_velocity(self.f1_ref, self.cgs_ref)

        #err1 = ((self.u2-self.u1_ref)**2).mean()
        #err2 = ((self.u2-self.u1)**2).mean()
        #reward = ((err1-err2))/1.4e-05 + 1

        #compute neq filed
        #rho, u = self.cgs.update_macroscopic(self.f1)
        #feq = self.cgs.equilibrium(rho, u, cast_output=False)
        #self.fneq = self.f1 - feq

        #also compute the correlation of voriticity as it is a good stopping criteria
        #self.v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        #self.v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        #corr = np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1]
        #terminated = bool(corr<0.97)
        #terminated = np.any(np.isnan(self.v1))
        terminated = np.any(self.f1 < 0) or np.any(self.f1 > 1)
        truncated = bool(self.counter>=self.max_episode_steps)

        reward = 1.

        #add a negative termination reward
        if terminated:
            reward -= 100.
        #add a truncation reward to account for missing future rewards
        if truncated:
            reward += 1/(1-0.97)

        return self.f1, reward, terminated, truncated, {}

    def render(self):
        print("Correlation:", np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1])
        print("MSE:", ((self.u1 - self.u2)**2).mean())
        print("NMSE:", ((self.u1 - self.u2)**2).sum()/((self.u2)**2).sum())
        print("pointwise relative mse:", (((self.u1 - self.u2)**2)/((self.u2)**2)+1e-6).mean())
        # Plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        im1 = ax1.imshow(self.v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        #im2 = ax2.imshow(self.v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        #im3 = ax3.imshow((self.v1 - self.v2)**2)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        # Create a colorbar for the third plot
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        #fig.colorbar(im3, cax=cax)
        plt.show()
    
    #def _load_u2(self):
    #    u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
    #    return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])


# reintroduce fgs 
class KolmogorovEnvironment18(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=102, fgs_lamb=1, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N2048_S{self.sampled_seed}_U1_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs
        self.kwargs2, _, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=fgs_lamb, Re=self.Re) #fgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.cgs.nx, self.cgs.ny, 9), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.0025, high=0.0025, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])
        self.trivial_action = np.ones((self.cgs.nx, self.cgs.ny))

        #FGS
        self.u2 = self._load_u2()


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N2048_S{self.sampled_seed}_U1_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        self.u2 = self._load_u2()

        return self.f1, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * (1+action.reshape(self.omg.shape)))
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1

        self.u2 = self._load_u2()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        # local velocity err
        err2 = np.sum((self.u1 - self.u2)**2, axis=-1)
        reward = 1 - err2

        terminated = np.any([np.any(self.f1 < 0), np.any(self.f1 > 1), np.any(np.sqrt(np.sum(self.u1**2, axis=-1)) > 100)])
        truncated = bool(self.counter>=self.max_episode_steps)

        #add a negative termination reward
        if terminated:
            reward -= 100.
        #add a truncation reward to account for missing future rewards
        if truncated:
            reward += 1/(1-0.97)

        return self.f1, reward, terminated, truncated, {}

    def render(self):
        #print("Correlation:", np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1])
        #print("MSE:", ((self.u1 - self.u2)**2).mean())
        #print("NMSE:", ((self.u1 - self.u2)**2).sum()/((self.u2)**2).sum())
        #print("pointwise relative mse:", (((self.u1 - self.u2)**2)/((self.u2)**2)+1e-6).mean())

        v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        v2 = vorticity_2d(self.u2, self.kwargs1["dx_eff"])

        # Plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        im1 = ax1.imshow(v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = ax2.imshow(v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im3 = ax3.imshow((v1 - v2)**2)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        # Create a colorbar for the third plot
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        #fig.colorbar(im3, cax=cax)
        plt.show()
    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])



# adaption for centralized learning -> compute mean over returns
class KolmogorovEnvironment19(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=102, fgs_lamb=1, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re
        self.fgs_lamb = fgs_lamb

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N{int(self.fgs_lamb*128)}_S{self.sampled_seed}_U1_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs
        self.kwargs2, _, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=fgs_lamb, Re=self.Re) #fgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.cgs.nx, self.cgs.ny, 9), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.0025, high=0.0025, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])
        self.trivial_action = np.ones((self.cgs.nx, self.cgs.ny))

        #FGS
        self.u2 = self._load_u2()


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N{int(self.fgs_lamb*128)}_S{self.sampled_seed}_U1_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        self.u2 = self._load_u2()

        return self.f1, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * (1+action.reshape(self.omg.shape)))
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1

        self.u2 = self._load_u2()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        # local velocity err
        err2 = np.sum((self.u1 - self.u2)**2, axis=-1)
        reward = (1 - err2).mean()

        terminated = False

        if np.any([np.any(self.f1 < 0), np.any(self.f1 > 1), np.any(np.sqrt(np.sum(self.u1**2, axis=-1)) > 100)]):
            terminated = True
            reward -= 100.

        if np.any(np.sqrt(np.sum(self.u2**2, axis=-1)) > 100):
            terminated = True
        
        truncated = bool(self.counter>=self.max_episode_steps)
            
        #add a truncation reward to account for missing future rewards
        if truncated:
            reward += 1/(1-0.97)

        return self.f1, reward, terminated, truncated, {}

    def render(self):
        #print("Correlation:", np.corrcoef(self.v1.flatten(), self.v2.flatten())[0, 1])
        #print("MSE:", ((self.u1 - self.u2)**2).mean())
        #print("NMSE:", ((self.u1 - self.u2)**2).sum()/((self.u2)**2).sum())
        #print("pointwise relative mse:", (((self.u1 - self.u2)**2)/((self.u2)**2)+1e-6).mean())

        v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        v2 = vorticity_2d(self.u2, self.kwargs1["dx_eff"])

        # Plot v1 and v2 next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        im1 = ax1.imshow(v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = ax2.imshow(v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im3 = ax3.imshow((v1 - v2)**2)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title("CGS")
        ax2.set_title("FGS")
        ax3.set_title("MSE")
        # Create a colorbar for the third plot
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        #fig.colorbar(im3, cax=cax)
        plt.show()
    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])



# back to single actions
class KolmogorovEnvironment20(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re
        self.fgs_lamb = fgs_lamb

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        #self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N{int(self.fgs_lamb*128)}_S{self.sampled_seed}_U1_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs
        self.kwargs2, _, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=fgs_lamb, Re=self.Re) #fgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        #self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.omg = np.copy(self.cgs.omega)
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1 = get_velocity(self.f1, self.cgs)
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.cgs.nx, self.cgs.ny, 2), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.0025, high=0.0025, shape=(1,), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])
        self.trivial_action = np.ones((self.cgs.nx, self.cgs.ny))

        #FGS
        #self.u2 = self._load_u2()


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        #self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N{int(self.fgs_lamb*128)}_S{self.sampled_seed}_U1_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        #self.u2 = self._load_u2()

        return self.u1, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        #self.cgs.omega = np.copy(self.omg * (1+action.reshape(self.omg.shape)))
        self.cgs.omega = np.copy(self.omg * (1+action))
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1

        #self.u2 = self._load_u2()
        _, self.u1 = get_velocity(self.f1, self.cgs)
        # local velocity err
        #err2 = np.sum((self.u1 - self.u2)**2, axis=-1)
        #reward = (1 - err2).mean()
        reward = 1

        terminated = False
        if np.any([np.any(self.f1 < 0), np.any(self.f1 > 1), np.any(np.sqrt(np.sum(self.u1**2, axis=-1)) > 100)]):
            terminated = True
            reward -= 100.
        #if np.any(np.sqrt(np.sum(self.u2**2, axis=-1)) > 100):
        #    terminated = True
        truncated = bool(self.counter>=self.max_episode_steps)
        #add a truncation reward to account for missing future rewards
        if truncated:
            reward += 1/(1-0.97)

        return self.u1, reward, terminated, truncated, {}

    def render(self, savefig=False):

        v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        magnitude = lambda u : np.sqrt(np.sum(u**2, axis=-1))
        # Your plotting function
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid of subplots
        # Plot CGS, FGS, and MSE fields in the first row
        im1 = axes[0, 0].imshow(v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = axes[0, 1].imshow(v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        #plot enerty spectra
        E1, E2 = self.get_spectra()
        axes[0,2].loglog(E1, label="CGS")
        axes[0,2].loglog(E2, label="FGS")
        axes[0,2].legend()
        axes[0,2].set_title("Energy spectra")
        axes[0,2].set_xlabel("wavenumber k")
        axes[0,2].set_ylabel("Energy E(k)")   
        # Plot velocity magnitude for CGS and FGS in the second row
        im4 = axes[1, 0].imshow(magnitude(self.u1), cmap='plasma')
        im5 = axes[1, 1].imshow(magnitude(self.u2), cmap='plasma')
        #plot velocity MSE
        im6 = axes[1, 2].imshow(np.sum((self.u1 - self.u2)**2, axis=-1), cmap='viridis')
        # Hide axes for the third column of the second row (unused)
        axes[1, 2].axis('off')
        # Remove axis ticks for all subplots
        for ax in axes.flat:
            ax.axis('off')
        axes[0,2].axis('on')
        # Set titles for the subplots
        axes[0, 0].set_title("Vorticity CGS")
        axes[0, 1].set_title("Vorticity FGS")
        axes[0, 2].set_title("Energy Spectrum")
        axes[1, 0].set_title("Velocity Magnitude CGS")
        axes[1, 1].set_title("Velocity Magnitude FGS")
        axes[1, 2].set_title("Velocity MSE")
        # Create a colorbar for the third plot (MSE)
        divider = make_axes_locatable(axes[1, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im6, cax=cax)
        # Create colorbars for velocity magnitude plots
        divider_cgs = make_axes_locatable(axes[1, 0])
        cax_cgs = divider_cgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im4, cax=cax_cgs)
        divider_fgs = make_axes_locatable(axes[1, 1])
        cax_fgs = divider_fgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im5, cax=cax_fgs)
        # Show the plot
        plt.tight_layout()
        #if save_fig == True:
        #    plt.savefig(f"visuals/img{i}.png", dpi=100)
        #    plt.close()
        #else:
        #    plt.show()
        plt.show()

    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2


# use KL loss of enegy spectrum
INIT_PATH_SPEC = "/home/pfischer/XLB/dns_spectrum/"
class KolmogorovEnvironment21(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re
        self.fgs_lamb = fgs_lamb

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        #self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N{int(self.fgs_lamb*128)}_S{self.sampled_seed}_U1_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs
        #self.kwargs2, _, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=fgs_lamb, Re=self.Re) #fgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        #self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.omg = np.copy(self.cgs.omega)
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        self.state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.cgs.nx, self.cgs.ny, 6), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.0025, high=0.0025, shape=(1,), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])

        #FGS
        #self.u2 = self._load_u2()
        #load enerty spectrum
        self.means_dns = np.load(INIT_PATH_SPEC+'means_dns.npy')
        stds_dns = np.load(INIT_PATH_SPEC+'stds_dns.npy')
        self.cov = np.diag(stds_dns)
        self.cov_inverse = np.diag(1/stds_dns)
        assert np.any(np.isnan(self.cov_inverse)) is not True
        assert self.cov@self.cov_inverse is not np.identity(len(self.means_dns))


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        #self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N{int(self.fgs_lamb*128)}_S{self.sampled_seed}_U1_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        self.state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)

        return self.state, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        #self.cgs.omega = np.copy(self.omg * (1+action.reshape(self.omg.shape)))
        self.cgs.omega = np.copy(self.omg * (1+action))
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1

        #self.u2 = self._load_u2()

        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        self.state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        _, E1 = energy_spectrum_2d(self.u1)
        reward = self.E_loss(E1)

        terminated = False
        if np.any([np.any(self.f1 < 0), np.any(self.f1 > 1), np.any(np.sqrt(np.sum(self.u1**2, axis=-1)) > 100)]):
            terminated = True
            reward -= 100.
        truncated = bool(self.counter>=self.max_episode_steps)

        return self.state, reward, terminated, truncated, {}

    def render(self, savefig=False):

        v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        magnitude = lambda u : np.sqrt(np.sum(u**2, axis=-1))
        # Your plotting function
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid of subplots
        # Plot CGS, FGS, and MSE fields in the first row
        im1 = axes[0, 0].imshow(v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = axes[0, 1].imshow(v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        #plot enerty spectra
        E1, E2 = self.get_spectra()
        axes[0,2].loglog(E1, label="CGS")
        axes[0,2].loglog(E2, label="FGS")
        axes[0,2].loglog(self.means_dns, label="DNS")
        axes[0,2].legend()
        axes[0,2].set_title("Energy spectra")
        axes[0,2].set_xlabel("wavenumber k")
        axes[0,2].set_ylabel("Energy E(k)")   
        # Plot velocity magnitude for CGS and FGS in the second row
        im4 = axes[1, 0].imshow(magnitude(self.u1), cmap='plasma')
        im5 = axes[1, 1].imshow(magnitude(self.u2), cmap='plasma')
        #plot velocity MSE
        im6 = axes[1, 2].imshow(np.sum((self.u1 - self.u2)**2, axis=-1), cmap='viridis')
        # Hide axes for the third column of the second row (unused)
        axes[1, 2].axis('off')
        # Remove axis ticks for all subplots
        for ax in axes.flat:
            ax.axis('off')
        axes[0,2].axis('on')
        # Set titles for the subplots
        axes[0, 0].set_title("Vorticity CGS")
        axes[0, 1].set_title("Vorticity FGS")
        axes[0, 2].set_title("Energy Spectrum")
        axes[1, 0].set_title("Velocity Magnitude CGS")
        axes[1, 1].set_title("Velocity Magnitude FGS")
        axes[1, 2].set_title("Velocity MSE")
        # Create a colorbar for the third plot (MSE)
        divider = make_axes_locatable(axes[1, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im6, cax=cax)
        # Create colorbars for velocity magnitude plots
        divider_cgs = make_axes_locatable(axes[1, 0])
        cax_cgs = divider_cgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im4, cax=cax_cgs)
        divider_fgs = make_axes_locatable(axes[1, 1])
        cax_fgs = divider_fgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im5, cax=cax_fgs)
        # Show the plot
        plt.tight_layout()
        #if save_fig == True:
        #    plt.savefig(f"visuals/img{i}.png", dpi=100)
        #    plt.close()
        #else:
        #    plt.show()
        plt.show()

    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2

    def E_loss(self, means_cgs):
        means_diff = self.means_dns-means_cgs
        return np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff)
    

# move back to full MARL setup
class KolmogorovEnvironment22_global(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=42, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re
        self.fgs_lamb = fgs_lamb

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega)
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.cgs.nx, self.cgs.ny, 6), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.005, high=0.005, shape=(1,), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])

        #load DNS enerty spectrum
        self.means_dns = np.load(INIT_PATH_SPEC+'means_log_k5-10_dns.npy')
        stds_dns = np.load(INIT_PATH_SPEC+'stds_log_k5-10_dns.npy')
        self.cov = np.diag(stds_dns)
        self.cov_inverse = np.diag(1/stds_dns)
        assert np.any(np.isnan(self.cov_inverse)) is not True
        assert self.cov@self.cov_inverse is not np.identity(len(self.means_dns))

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)

        return state, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        #self.cgs.omega = np.copy(self.omg * (1+action.reshape(self.omg.shape)))
        self.cgs.omega = np.copy(self.omg * (1+action))
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1

        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        k, E1 = energy_spectrum_2d(self.u1)
        reward = self.E_loss(E1, k)

        terminated = False
        if np.any([np.any(self.f1 < 0), np.any(self.f1 > 1), np.any(np.sqrt(np.sum(self.u1**2, axis=-1)) > 100)]):
            terminated = True
            reward -= 100.
        truncated = bool(self.counter>=self.max_episode_steps)

        return state, reward, terminated, truncated, {}

    def render(self, savefig=False):

        v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        magnitude = lambda u : np.sqrt(np.sum(u**2, axis=-1))
        # Your plotting function
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid of subplots
        # Plot CGS, FGS, and MSE fields in the first row
        im1 = axes[0, 0].imshow(v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = axes[0, 1].imshow(v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        #plot enerty spectra
        E1, E2 = self.get_spectra()
        axes[0,2].loglog(E1, label="CGS")
        axes[0,2].loglog(E2, label="FGS")
        axes[0,2].loglog(self.means_dns, label="DNS")
        axes[0,2].legend()
        axes[0,2].set_title("Energy spectra")
        axes[0,2].set_xlabel("wavenumber k")
        axes[0,2].set_ylabel("Energy E(k)")   
        # Plot velocity magnitude for CGS and FGS in the second row
        im4 = axes[1, 0].imshow(magnitude(self.u1), cmap='plasma')
        im5 = axes[1, 1].imshow(magnitude(self.u2), cmap='plasma')
        #plot velocity MSE
        im6 = axes[1, 2].imshow(np.sum((self.u1 - self.u2)**2, axis=-1), cmap='viridis')
        # Hide axes for the third column of the second row (unused)
        axes[1, 2].axis('off')
        # Remove axis ticks for all subplots
        for ax in axes.flat:
            ax.axis('off')
        axes[0,2].axis('on')
        # Set titles for the subplots
        axes[0, 0].set_title("Vorticity CGS")
        axes[0, 1].set_title("Vorticity FGS")
        axes[0, 2].set_title("Energy Spectrum")
        axes[1, 0].set_title("Velocity Magnitude CGS")
        axes[1, 1].set_title("Velocity Magnitude FGS")
        axes[1, 2].set_title("Velocity MSE")
        # Create a colorbar for the third plot (MSE)
        divider = make_axes_locatable(axes[1, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im6, cax=cax)
        # Create colorbars for velocity magnitude plots
        divider_cgs = make_axes_locatable(axes[1, 0])
        cax_cgs = divider_cgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im4, cax=cax_cgs)
        divider_fgs = make_axes_locatable(axes[1, 1])
        cax_fgs = divider_fgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im5, cax=cax_fgs)
        # Show the plot
        plt.tight_layout()
        #if save_fig == True:
        #    plt.savefig(f"visuals/img{i}.png", dpi=100)
        #    plt.close()
        #else:
        #    plt.show()
        plt.show()
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2

    def E_loss(self, means_cgs, k):
        means_diff = np.log(means_cgs[1:]*k[1:]**5)/10 - self.means_dns
        return 3 + np.log(np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff))/64


# move back to full MARL setup
class KolmogorovEnvironment22_global_decay(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=42, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re
        self.fgs_lamb = fgs_lamb

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs

        #CGS
        self.cgs = decaying_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega)
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.cgs.nx, self.cgs.ny, 6), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.005, high=0.005, shape=(1,), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])

        #load DNS enerty spectrum
        self.means_dns = np.load(INIT_PATH_SPEC+'means_log_k5-10_dns.npy')
        stds_dns = np.load(INIT_PATH_SPEC+'stds_log_k5-10_dns.npy')
        self.cov = np.diag(stds_dns)
        self.cov_inverse = np.diag(1/stds_dns)
        assert np.any(np.isnan(self.cov_inverse)) is not True
        assert self.cov@self.cov_inverse is not np.identity(len(self.means_dns))

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.cgs = decaying_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)

        return state, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        #self.cgs.omega = np.copy(self.omg * (1+action.reshape(self.omg.shape)))
        self.cgs.omega = np.copy(self.omg * (1+action))
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1

        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        k, E1 = energy_spectrum_2d(self.u1)
        reward = self.E_loss(E1, k)

        terminated = False
        if np.any([np.any(self.f1 < 0), np.any(self.f1 > 1), np.any(np.sqrt(np.sum(self.u1**2, axis=-1)) > 100)]):
            terminated = True
            reward -= 100.
        truncated = bool(self.counter>=self.max_episode_steps)

        return state, reward, terminated, truncated, {}

    def render(self, savefig=False):

        v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        magnitude = lambda u : np.sqrt(np.sum(u**2, axis=-1))
        # Your plotting function
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid of subplots
        # Plot CGS, FGS, and MSE fields in the first row
        im1 = axes[0, 0].imshow(v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = axes[0, 1].imshow(v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        #plot enerty spectra
        E1, E2 = self.get_spectra()
        axes[0,2].loglog(E1, label="CGS")
        axes[0,2].loglog(E2, label="FGS")
        axes[0,2].loglog(self.means_dns, label="DNS")
        axes[0,2].legend()
        axes[0,2].set_title("Energy spectra")
        axes[0,2].set_xlabel("wavenumber k")
        axes[0,2].set_ylabel("Energy E(k)")   
        # Plot velocity magnitude for CGS and FGS in the second row
        im4 = axes[1, 0].imshow(magnitude(self.u1), cmap='plasma')
        im5 = axes[1, 1].imshow(magnitude(self.u2), cmap='plasma')
        #plot velocity MSE
        im6 = axes[1, 2].imshow(np.sum((self.u1 - self.u2)**2, axis=-1), cmap='viridis')
        # Hide axes for the third column of the second row (unused)
        axes[1, 2].axis('off')
        # Remove axis ticks for all subplots
        for ax in axes.flat:
            ax.axis('off')
        axes[0,2].axis('on')
        # Set titles for the subplots
        axes[0, 0].set_title("Vorticity CGS")
        axes[0, 1].set_title("Vorticity FGS")
        axes[0, 2].set_title("Energy Spectrum")
        axes[1, 0].set_title("Velocity Magnitude CGS")
        axes[1, 1].set_title("Velocity Magnitude FGS")
        axes[1, 2].set_title("Velocity MSE")
        # Create a colorbar for the third plot (MSE)
        divider = make_axes_locatable(axes[1, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im6, cax=cax)
        # Create colorbars for velocity magnitude plots
        divider_cgs = make_axes_locatable(axes[1, 0])
        cax_cgs = divider_cgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im4, cax=cax_cgs)
        divider_fgs = make_axes_locatable(axes[1, 1])
        cax_fgs = divider_fgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im5, cax=cax_fgs)
        # Show the plot
        plt.tight_layout()
        #if save_fig == True:
        #    plt.savefig(f"visuals/img{i}.png", dpi=100)
        #    plt.close()
        #else:
        #    plt.show()
        plt.show()
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2

    def E_loss(self, means_cgs, k):
        means_diff = np.log(means_cgs[1:]*k[1:]**5)/10 - self.means_dns
        return 3 + np.log(np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff))/64


# move back to full MARL setup
class KolmogorovEnvironment22_global_higher(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=42, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re
        self.fgs_lamb = fgs_lamb

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega)
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.cgs.nx, self.cgs.ny, 6), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.005, high=0.005, shape=(1,), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])

        #load DNS enerty spectrum
        self.means_dns = np.load(INIT_PATH_SPEC+'means_log_k5-10_dns.npy')
        stds_dns = np.load(INIT_PATH_SPEC+'stds_log_k5-10_dns.npy')
        self.cov = np.diag(stds_dns)
        self.cov_inverse = np.diag(1/stds_dns)
        assert np.any(np.isnan(self.cov_inverse)) is not True
        assert self.cov@self.cov_inverse is not np.identity(len(self.means_dns))

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_raw_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)

        return state, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        #self.cgs.omega = np.copy(self.omg * (1+action.reshape(self.omg.shape)))
        self.cgs.omega = np.copy(self.omg * (1+action))
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1

        self.rho1, self.u1, self.P_neq1 = get_raw_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        k, E1 = energy_spectrum_2d(downsample_field(self.u1, 2))
        reward = self.E_loss(E1, k)

        terminated = False
        if np.any([np.any(self.f1 < 0), np.any(self.f1 > 1), np.any(np.sqrt(np.sum(self.u1**2, axis=-1)) > 100)]):
            terminated = True
            reward -= 100.
        truncated = bool(self.counter>=self.max_episode_steps)

        return state, reward, terminated, truncated, {}

    def render(self, savefig=False):

        v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        magnitude = lambda u : np.sqrt(np.sum(u**2, axis=-1))
        # Your plotting function
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid of subplots
        # Plot CGS, FGS, and MSE fields in the first row
        im1 = axes[0, 0].imshow(v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = axes[0, 1].imshow(v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        #plot enerty spectra
        E1, E2 = self.get_spectra()
        axes[0,2].loglog(E1, label="CGS")
        axes[0,2].loglog(E2, label="FGS")
        axes[0,2].loglog(self.means_dns, label="DNS")
        axes[0,2].legend()
        axes[0,2].set_title("Energy spectra")
        axes[0,2].set_xlabel("wavenumber k")
        axes[0,2].set_ylabel("Energy E(k)")   
        # Plot velocity magnitude for CGS and FGS in the second row
        im4 = axes[1, 0].imshow(magnitude(self.u1), cmap='plasma')
        im5 = axes[1, 1].imshow(magnitude(self.u2), cmap='plasma')
        #plot velocity MSE
        im6 = axes[1, 2].imshow(np.sum((self.u1 - self.u2)**2, axis=-1), cmap='viridis')
        # Hide axes for the third column of the second row (unused)
        axes[1, 2].axis('off')
        # Remove axis ticks for all subplots
        for ax in axes.flat:
            ax.axis('off')
        axes[0,2].axis('on')
        # Set titles for the subplots
        axes[0, 0].set_title("Vorticity CGS")
        axes[0, 1].set_title("Vorticity FGS")
        axes[0, 2].set_title("Energy Spectrum")
        axes[1, 0].set_title("Velocity Magnitude CGS")
        axes[1, 1].set_title("Velocity Magnitude FGS")
        axes[1, 2].set_title("Velocity MSE")
        # Create a colorbar for the third plot (MSE)
        divider = make_axes_locatable(axes[1, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im6, cax=cax)
        # Create colorbars for velocity magnitude plots
        divider_cgs = make_axes_locatable(axes[1, 0])
        cax_cgs = divider_cgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im4, cax=cax_cgs)
        divider_fgs = make_axes_locatable(axes[1, 1])
        cax_fgs = divider_fgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im5, cax=cax_fgs)
        # Show the plot
        plt.tight_layout()
        #if save_fig == True:
        #    plt.savefig(f"visuals/img{i}.png", dpi=100)
        #    plt.close()
        #else:
        #    plt.show()
        plt.show()
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2

    def E_loss(self, means_cgs, k):
        means_diff = np.log(means_cgs[1:]*k[1:]**5)/10 - self.means_dns
        return 3 + np.log(np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff))/64


# move back to full MARL setup
class KolmogorovEnvironment22_old(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re
        self.fgs_lamb = fgs_lamb

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        #self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.cgs.nx, self.cgs.ny, 6), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.005, high=0.005, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])

        #FGS
        self.means_dns = np.load(INIT_PATH_SPEC+'means_log_k5-10_dns.npy')
        stds_dns = np.load(INIT_PATH_SPEC+'stds_log_k5-10_dns.npy')
        self.cov = np.diag(stds_dns)
        self.cov_inverse = np.diag(1/stds_dns)
        assert np.any(np.isnan(self.cov_inverse)) is not True
        assert self.cov@self.cov_inverse is not np.identity(len(self.means_dns))

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)

        return state, {}
    
    def step(self, action):
        #if action.shape != self.action_space.shape:
        #    try:
        #        action = action.reshape(self.action_space.shape)
        #    except:
        #        print("action reshaping didn't work")
        #
        #if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
        #    print("WARNING: Action is not in action space")
        #    action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * (1+action.reshape(self.omg.shape)))
        #self.cgs.omega = np.copy(self.omg * (1+action))
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1

        #self.u2 = self._load_u2()

        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        k, E1 = energy_spectrum_2d(self.u1)
        reward = self.E_loss(E1, k)

        terminated = False
        if np.any([np.any(self.f1 < 0), np.any(self.f1 > 1), np.any(np.sqrt(np.sum(self.u1**2, axis=-1)) > 100)]):
            terminated = True
            reward -= 100.
        truncated = bool(self.counter>=self.max_episode_steps)

        return state, reward, terminated, truncated, {}

    def render(self, savefig=False):
        return 0

    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2

    def E_loss(self, means_cgs, k):
        means_diff = np.log(means_cgs[1:]*k[1:]**5)/10 - self.means_dns
        return 1 + np.log(np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff))/64
        #return np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff)




# new 22 with jax for speedup
class KolmogorovEnvironment22_new(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re
        self.fgs_lamb = fgs_lamb

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = jnp.array(self.cgs.omega, copy=True)*jnp.ones((self.cgs.nx, self.cgs.ny, 1))
        self.f1 = self.cgs.assign_fields_sharded()
        
        self.c = jnp.array(self.cgs.lattice.c, dtype=jnp.float64)
        self.cc = jnp.array(self.cgs.lattice.cc, dtype=jnp.float64)
        self.w = jnp.array(self.cgs.lattice.w, dtype=jnp.float64)
        self.N = self.cgs.nx
        self.C_u = self.cgs.C_u
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.cgs.nx, self.cgs.ny, 6), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.005, high=0.005, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float64)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])

        # Load data using numpy, then convert to JAX arrays
        self.means_dns = jnp.array(np.load(INIT_PATH_SPEC + 'means_log_k5-10_dns.npy'))
        stds_dns = jnp.array(np.load(INIT_PATH_SPEC + 'stds_log_k5-10_dns.npy'))
        self.cov_inverse = jnp.diag(1/stds_dns)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = jnp.array(self.cgs.omega, copy=True)*jnp.ones((self.cgs.nx, self.cgs.ny, 1))
        #self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        #state = self.get_state()
        #state = jax2torch(state).to(torch.float32)
        self.rho1, self.u1, self.P_neq1 = self.get_moments()
        return np.concatenate((self.rho1, self.u1, self.P_neq1), axis=-1, dtype=np.float32), {}

        #return state.detach().cpu().numpy(), {}
    
    def step(self, action):
        #action = jnp.array(action, dtype=np.float32)
        self.cgs.omega = jnp.array(self.omg * (1+action.reshape(self.omg.shape)), copy=True)
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1

        self.rho1, self.u1, self.P_neq1 = self.get_moments()
        state =  np.concatenate((self.rho1, self.u1, self.P_neq1), axis=-1, dtype=np.float32)
        k, E1 = self.energy_spectrum_2d(self.u1)
        reward = self.E_loss(E1, k)
        
        terminated = (
            jnp.any(self.f1 < 0) |
            jnp.any(self.f1 > 1) |
            jnp.any(jnp.sqrt(jnp.sum(state[...,1:3]**2, axis=-1)) > 100)
        )

        reward = jnp.where(terminated, reward - 100.0, reward)
        truncated = self.counter >= self.max_episode_steps

        #state = jax2torch(state).to(torch.float32)
        #return state.detach().cpu().numpy(), np.array(reward), terminated, truncated, {}
        return state, np.array(reward), terminated, truncated, {}

    def render(self, savefig=False):
        return 0

    @partial(jit, static_argnums=(0,))
    def get_moments(self):
        rho, u = self.cgs.update_macroscopic(self.f1)
        fneq = self.f1 - self.cgs.equilibrium(rho, u, cast_output=True)
        P_neq = self.cgs.momentum_flux(fneq)
        #rho = downsample_field(rho, sim.downsamplingFactor)
        #u = downsample_field(u, sim.downsamplingFactor)
        #P_neq = downsample_field(P_neq, sim.downsamplingFactor)
        #rho = process_allgather(rho)
        #u = process_allgather(u)/sim.C_u
        #P_neq = process_allgather(P_neq)
        return rho, u/self.C_u, P_neq
    
    @partial(jit, static_argnums=(0,1))
    def get_state(self, cast_output=True):
        rho1 =jnp.sum(self.f1, axis=-1, keepdims=True)
        cT = (self.c).T
        u1 = jnp.dot(self.f1, cT)/rho1
        if cast_output:
            rho1, u1 = self.cgs.precisionPolicy.cast_to_compute((rho1, u1))
        cu = 3.0 * jnp.dot(u1, self.c)
        usqr = 1.5 * jnp.sum(jnp.square(u1), axis=-1, keepdims=True)
        feq = rho1 * self.w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
        fneq = self.f1-feq
        P_neq1 = jnp.dot(fneq, self.cc)
        state = jnp.concatenate((rho1, u1/self.C_u, P_neq1), axis=-1, dtype=np.float32)
        return state


    @partial(jit, static_argnums=(0,))
    def energy_spectrum_2d(self, u):
        # Compute the inverse FFT
        uh = jnp.fft.ifft2(u[..., 0], norm='backward')
        vh = jnp.fft.ifft2(u[..., 1], norm='backward')
        uhat = jnp.stack([uh, vh], axis=-1)

        # Compute energy spectrum E_k
        E_k = jnp.sum(jnp.conj(uhat) * uhat, axis=-1).real / 2.
        freq = jnp.fft.fftfreq(self.N, d=1/self.N)
        kx, ky = jnp.meshgrid(freq, freq)
        knorm = jnp.sqrt(kx**2 + ky**2)
        ks = jnp.arange(1, int(self.N / 2))
        Ek = jnp.zeros(len(ks))

        # Loop to calculate the energy spectrum at each wavenumber
        def compute_Ek(i, Ek):
            k = ks[i]
            mask = (knorm > k - 0.5) & (knorm <= k + 0.5)
            # Create a masked version of E_k where only the elements within the mask are kept
            masked_E_k = jnp.where(mask, E_k, 0)
            count = jnp.sum(mask)  # Count the number of elements in the mask
            # Compute the mean only over the masked elements
            mean_value = jnp.sum(masked_E_k)/count
            Ek = Ek.at[i].set(mean_value)

            return Ek

        # Use a JAX loop (lax.scan) for the loop
        Ek = jax.lax.fori_loop(0, len(ks), compute_Ek, Ek)

        return ks, 2 * jnp.pi * Ek

    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2

    @partial(jit, static_argnums=(0,))
    def E_loss(self, means_cgs, k):
        means_diff = jnp.log(means_cgs[1:] * k[1:]**5)/10 - self.means_dns
        exponent = -0.5 * means_diff.T @ self.cov_inverse @ means_diff
        return 1 + jnp.log(jnp.exp(exponent))/64


# for evaluation on higher grid size namely 256
class KolmogorovEnvironment22_higher(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re
        self.fgs_lamb = fgs_lamb

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        print(f"omega shape = {self.omg.shape}")
        self.cgs.omg = np.copy(self.omg)
        print(f"cgs omega shape = {self.cgs.omg.shape}")
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        print(f"f shape = {self.f1.shape}")
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.cgs.nx, self.cgs.ny, 6), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.005, high=0.005, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])

        #FGS
        self.means_dns = np.load(INIT_PATH_SPEC+'means_log_k5-10_dns.npy')
        stds_dns = np.load(INIT_PATH_SPEC+'stds_log_k5-10_dns.npy')
        self.cov = np.diag(stds_dns)
        self.cov_inverse = np.diag(1/stds_dns)
        assert np.any(np.isnan(self.cov_inverse)) is not True
        assert self.cov@self.cov_inverse is not np.identity(len(self.means_dns))



    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_raw_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)

        return state, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * (1+action.reshape(self.omg.shape)))
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1

        self.rho1, self.u1, self.P_neq1 = get_raw_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        k, E1 = energy_spectrum_2d(downsample_field(self.u1, 2))
        reward = self.E_loss(E1, k)

        terminated = False
        if np.any([np.any(self.f1 < 0), np.any(self.f1 > 1), np.any(np.sqrt(np.sum(self.u1**2, axis=-1)) > 100)]):
            terminated = True
            reward -= 100.
        truncated = bool(self.counter>=self.max_episode_steps)

        return state, reward, terminated, truncated, {}

    def render(self, savefig=False):

        v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        magnitude = lambda u : np.sqrt(np.sum(u**2, axis=-1))
        # Your plotting function
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid of subplots
        # Plot CGS, FGS, and MSE fields in the first row
        im1 = axes[0, 0].imshow(v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = axes[0, 1].imshow(v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        #plot enerty spectra
        E1, E2 = self.get_spectra()
        axes[0,2].loglog(E1, label="CGS")
        axes[0,2].loglog(E2, label="FGS")
        axes[0,2].loglog(self.means_dns, label="DNS")
        axes[0,2].legend()
        axes[0,2].set_title("Energy spectra")
        axes[0,2].set_xlabel("wavenumber k")
        axes[0,2].set_ylabel("Energy E(k)")   
        # Plot velocity magnitude for CGS and FGS in the second row
        im4 = axes[1, 0].imshow(magnitude(self.u1), cmap='plasma')
        im5 = axes[1, 1].imshow(magnitude(self.u2), cmap='plasma')
        #plot velocity MSE
        im6 = axes[1, 2].imshow(np.sum((self.u1 - self.u2)**2, axis=-1), cmap='viridis')
        # Hide axes for the third column of the second row (unused)
        axes[1, 2].axis('off')
        # Remove axis ticks for all subplots
        for ax in axes.flat:
            ax.axis('off')
        axes[0,2].axis('on')
        # Set titles for the subplots
        axes[0, 0].set_title("Vorticity CGS")
        axes[0, 1].set_title("Vorticity FGS")
        axes[0, 2].set_title("Energy Spectrum")
        axes[1, 0].set_title("Velocity Magnitude CGS")
        axes[1, 1].set_title("Velocity Magnitude FGS")
        axes[1, 2].set_title("Velocity MSE")
        # Create a colorbar for the third plot (MSE)
        divider = make_axes_locatable(axes[1, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im6, cax=cax)
        # Create colorbars for velocity magnitude plots
        divider_cgs = make_axes_locatable(axes[1, 0])
        cax_cgs = divider_cgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im4, cax=cax_cgs)
        divider_fgs = make_axes_locatable(axes[1, 1])
        cax_fgs = divider_fgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im5, cax=cax_fgs)
        # Show the plot
        plt.tight_layout()
        #if save_fig == True:
        #    plt.savefig(f"visuals/img{i}.png", dpi=100)
        #    plt.close()
        #else:
        #    plt.show()
        plt.show()

    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2

    def E_loss(self, means_cgs, k):
        means_diff = np.log(means_cgs[1:]*k[1:]**5)/10 - self.means_dns
        return 1 + np.log(np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff))/64
        #return np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff)



# move back to full MARL setup
class KolmogorovEnvironment22_decay(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re
        self.fgs_lamb = fgs_lamb

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs
        
        #CGS
        self.cgs = decaying_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        print(f"omega shape = {self.omg.shape}")
        self.cgs.omg = np.copy(self.omg)
        print(f"cgs omega shape = {self.cgs.omg.shape}")
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        print(f"f shape = {self.f1.shape}")
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.cgs.nx, self.cgs.ny, 6), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.005, high=0.005, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])

        #FGS
        self.means_dns = np.load(INIT_PATH_SPEC+'means_log_k5-10_dns.npy')
        stds_dns = np.load(INIT_PATH_SPEC+'stds_log_k5-10_dns.npy')
        self.cov = np.diag(stds_dns)
        self.cov_inverse = np.diag(1/stds_dns)
        assert np.any(np.isnan(self.cov_inverse)) is not True
        assert self.cov@self.cov_inverse is not np.identity(len(self.means_dns))



    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        #self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N{int(self.fgs_lamb*128)}_S{self.sampled_seed}_U1_dump/"
        self.cgs = decaying_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)

        return state, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * (1+action.reshape(self.omg.shape)))
        #self.cgs.omega = np.copy(self.omg * (1+action))
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1

        #self.u2 = self._load_u2()

        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        k, E1 = energy_spectrum_2d(self.u1)
        reward = self.E_loss(E1, k)

        terminated = False
        if np.any([np.any(self.f1 < 0), np.any(self.f1 > 1), np.any(np.sqrt(np.sum(self.u1**2, axis=-1)) > 100)]):
            terminated = True
            reward -= 100.
        truncated = bool(self.counter>=self.max_episode_steps)

        return state, reward, terminated, truncated, {}

    def render(self, savefig=False):

        v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        magnitude = lambda u : np.sqrt(np.sum(u**2, axis=-1))
        # Your plotting function
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid of subplots
        # Plot CGS, FGS, and MSE fields in the first row
        im1 = axes[0, 0].imshow(v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = axes[0, 1].imshow(v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        #plot enerty spectra
        E1, E2 = self.get_spectra()
        axes[0,2].loglog(E1, label="CGS")
        axes[0,2].loglog(E2, label="FGS")
        axes[0,2].loglog(self.means_dns, label="DNS")
        axes[0,2].legend()
        axes[0,2].set_title("Energy spectra")
        axes[0,2].set_xlabel("wavenumber k")
        axes[0,2].set_ylabel("Energy E(k)")   
        # Plot velocity magnitude for CGS and FGS in the second row
        im4 = axes[1, 0].imshow(magnitude(self.u1), cmap='plasma')
        im5 = axes[1, 1].imshow(magnitude(self.u2), cmap='plasma')
        #plot velocity MSE
        im6 = axes[1, 2].imshow(np.sum((self.u1 - self.u2)**2, axis=-1), cmap='viridis')
        # Hide axes for the third column of the second row (unused)
        axes[1, 2].axis('off')
        # Remove axis ticks for all subplots
        for ax in axes.flat:
            ax.axis('off')
        axes[0,2].axis('on')
        # Set titles for the subplots
        axes[0, 0].set_title("Vorticity CGS")
        axes[0, 1].set_title("Vorticity FGS")
        axes[0, 2].set_title("Energy Spectrum")
        axes[1, 0].set_title("Velocity Magnitude CGS")
        axes[1, 1].set_title("Velocity Magnitude FGS")
        axes[1, 2].set_title("Velocity MSE")
        # Create a colorbar for the third plot (MSE)
        divider = make_axes_locatable(axes[1, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im6, cax=cax)
        # Create colorbars for velocity magnitude plots
        divider_cgs = make_axes_locatable(axes[1, 0])
        cax_cgs = divider_cgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im4, cax=cax_cgs)
        divider_fgs = make_axes_locatable(axes[1, 1])
        cax_fgs = divider_fgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im5, cax=cax_fgs)
        # Show the plot
        plt.tight_layout()
        #if save_fig == True:
        #    plt.savefig(f"visuals/img{i}.png", dpi=100)
        #    plt.close()
        #else:
        #    plt.show()
        plt.show()

    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2

    def E_loss(self, means_cgs, k):
        means_diff = np.log(means_cgs[1:]*k[1:]**5)/10 - self.means_dns
        return 1 + np.log(np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff))/64
        #return np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff)



# same as above but fgs data is loaded for visualization
class KolmogorovEnvironment22_visual(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re
        self.fgs_lamb = fgs_lamb

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N{int(self.fgs_lamb*128)}_S{self.sampled_seed}_U1_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs
        self.kwargs2, _, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=fgs_lamb, Re=self.Re) #fgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        print(f"omega shape = {self.omg.shape}")
        self.cgs.omg = np.copy(self.omg)
        print(f"cgs omega shape = {self.cgs.omg.shape}")
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        #self.state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        print(f"f shape = {self.f1.shape}")
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.cgs.nx, self.cgs.ny, 6), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.005, high=0.005, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])

        #FGS
        self.u2 = self._load_u2()
        #load enerty spectrum
        self.means_dns = np.load(INIT_PATH_SPEC+'means_dns.npy')
        stds_dns = np.load(INIT_PATH_SPEC+'stds_dns.npy')
        self.cov = np.diag(stds_dns)
        self.cov_inverse = np.diag(1/stds_dns)
        assert np.any(np.isnan(self.cov_inverse)) is not True
        assert self.cov@self.cov_inverse is not np.identity(len(self.means_dns))



    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N{int(self.fgs_lamb*128)}_S{self.sampled_seed}_U1_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)

        return state, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * (1+action.reshape(self.omg.shape)))
        #self.cgs.omega = np.copy(self.omg * (1+action))
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1

        self.u2 = self._load_u2()

        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        _, E1 = energy_spectrum_2d(self.u1)
        reward = self.E_loss(E1)

        terminated = False
        if np.any([np.any(self.f1 < 0), np.any(self.f1 > 1), np.any(np.sqrt(np.sum(self.u1**2, axis=-1)) > 100)]):
            terminated = True
            reward -= 100.
        truncated = bool(self.counter>=self.max_episode_steps)

        return state, reward, terminated, truncated, {}

    def render(self, savefig=False):

        v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        magnitude = lambda u : np.sqrt(np.sum(u**2, axis=-1))
        # Your plotting function
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid of subplots
        # Plot CGS, FGS, and MSE fields in the first row
        im1 = axes[0, 0].imshow(v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = axes[0, 1].imshow(v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        #plot enerty spectra
        E1, E2 = self.get_spectra()
        axes[0,2].loglog(E1, label="CGS")
        axes[0,2].loglog(E2, label="FGS")
        axes[0,2].loglog(self.means_dns, label="DNS")
        axes[0,2].legend()
        axes[0,2].set_title("Energy spectra")
        axes[0,2].set_xlabel("wavenumber k")
        axes[0,2].set_ylabel("Energy E(k)")   
        # Plot velocity magnitude for CGS and FGS in the second row
        im4 = axes[1, 0].imshow(magnitude(self.u1), cmap='plasma')
        im5 = axes[1, 1].imshow(magnitude(self.u2), cmap='plasma')
        #plot velocity MSE
        im6 = axes[1, 2].imshow(np.sum((self.u1 - self.u2)**2, axis=-1), cmap='viridis')
        # Hide axes for the third column of the second row (unused)
        axes[1, 2].axis('off')
        # Remove axis ticks for all subplots
        for ax in axes.flat:
            ax.axis('off')
        axes[0,2].axis('on')
        # Set titles for the subplots
        axes[0, 0].set_title("Vorticity CGS")
        axes[0, 1].set_title("Vorticity FGS")
        axes[0, 2].set_title("Energy Spectrum")
        axes[1, 0].set_title("Velocity Magnitude CGS")
        axes[1, 1].set_title("Velocity Magnitude FGS")
        axes[1, 2].set_title("Velocity MSE")
        # Create a colorbar for the third plot (MSE)
        divider = make_axes_locatable(axes[1, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im6, cax=cax)
        # Create colorbars for velocity magnitude plots
        divider_cgs = make_axes_locatable(axes[1, 0])
        cax_cgs = divider_cgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im4, cax=cax_cgs)
        divider_fgs = make_axes_locatable(axes[1, 1])
        cax_fgs = divider_fgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im5, cax=cax_fgs)
        # Show the plot
        plt.tight_layout()
        #if save_fig == True:
        #    plt.savefig(f"visuals/img{i}.png", dpi=100)
        #    plt.close()
        #else:
        #    plt.show()
        plt.show()

    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2

    def E_loss(self, means_cgs):
        means_diff = means_cgs - self.means_dns
        #return np.log(np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff))/64
        return np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff)



# back to velocity mse error
class KolmogorovEnvironment23(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re
        self.fgs_lamb = fgs_lamb

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N{int(self.fgs_lamb*128)}_S{self.sampled_seed}_U1_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs
        self.kwargs2, _, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=fgs_lamb, Re=self.Re) #fgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        print(f"omega shape = {self.omg.shape}")
        self.cgs.omg = np.copy(self.omg)
        print(f"cgs omega shape = {self.cgs.omg.shape}")
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        #self.state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        print(f"f shape = {self.f1.shape}")
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.cgs.nx, self.cgs.ny, 6), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.005, high=0.005, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])

        #FGS
        self.u2 = self._load_u2()
        #load enerty spectrum
        self.means_dns = np.load(INIT_PATH_SPEC+'means_dns.npy')
        stds_dns = np.load(INIT_PATH_SPEC+'stds_dns.npy')
        self.cov = np.diag(stds_dns)
        self.cov_inverse = np.diag(1/stds_dns)
        assert np.any(np.isnan(self.cov_inverse)) is not True
        assert self.cov@self.cov_inverse is not np.identity(len(self.means_dns))



    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N{int(self.fgs_lamb*128)}_S{self.sampled_seed}_U1_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)

        return state, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.cgs.omega = np.copy(self.omg * (1+action.reshape(self.omg.shape)))
        #self.cgs.omega = np.copy(self.omg * (1+action))
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1

        self.u2 = self._load_u2()

        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        _, E1 = energy_spectrum_2d(self.u1)
        reward1 = self.E_loss(E1) 
        err2 = np.sum((self.u1 - self.u2)**2, axis=-1)
        reward2 = (1 - err2).mean()
        reward = reward1 + reward2

        terminated = False
        if np.any([np.any(self.f1 < 0), np.any(self.f1 > 1), np.any(np.sqrt(np.sum(self.u1**2, axis=-1)) > 100)]):
            terminated = True
            reward -= 100.
        truncated = bool(self.counter>=self.max_episode_steps)

        return state, reward, terminated, truncated, {}

    def render(self, savefig=False):

        v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        magnitude = lambda u : np.sqrt(np.sum(u**2, axis=-1))
        # Your plotting function
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid of subplots
        # Plot CGS, FGS, and MSE fields in the first row
        im1 = axes[0, 0].imshow(v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = axes[0, 1].imshow(v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        #plot enerty spectra
        E1, E2 = self.get_spectra()
        axes[0,2].loglog(E1, label="CGS")
        axes[0,2].loglog(E2, label="FGS")
        axes[0,2].loglog(self.means_dns, label="DNS")
        axes[0,2].legend()
        axes[0,2].set_title("Energy spectra")
        axes[0,2].set_xlabel("wavenumber k")
        axes[0,2].set_ylabel("Energy E(k)")   
        # Plot velocity magnitude for CGS and FGS in the second row
        im4 = axes[1, 0].imshow(magnitude(self.u1), cmap='plasma')
        im5 = axes[1, 1].imshow(magnitude(self.u2), cmap='plasma')
        #plot velocity MSE
        im6 = axes[1, 2].imshow(np.sum((self.u1 - self.u2)**2, axis=-1), cmap='viridis')
        # Hide axes for the third column of the second row (unused)
        axes[1, 2].axis('off')
        # Remove axis ticks for all subplots
        for ax in axes.flat:
            ax.axis('off')
        axes[0,2].axis('on')
        # Set titles for the subplots
        axes[0, 0].set_title("Vorticity CGS")
        axes[0, 1].set_title("Vorticity FGS")
        axes[0, 2].set_title("Energy Spectrum")
        axes[1, 0].set_title("Velocity Magnitude CGS")
        axes[1, 1].set_title("Velocity Magnitude FGS")
        axes[1, 2].set_title("Velocity MSE")
        # Create a colorbar for the third plot (MSE)
        divider = make_axes_locatable(axes[1, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im6, cax=cax)
        # Create colorbars for velocity magnitude plots
        divider_cgs = make_axes_locatable(axes[1, 0])
        cax_cgs = divider_cgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im4, cax=cax_cgs)
        divider_fgs = make_axes_locatable(axes[1, 1])
        cax_fgs = divider_fgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im5, cax=cax_fgs)
        # Show the plot
        plt.tight_layout()
        #if save_fig == True:
        #    plt.savefig(f"visuals/img{i}.png", dpi=100)
        #    plt.close()
        #else:
        #    plt.show()
        plt.show()

    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2

    def E_loss(self, means_cgs):
        means_diff = means_cgs - self.means_dns
        #return np.log(np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff))/64
        return np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff)


# interpolating agents
class KolmogorovEnvironment24(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000, N_agents=8):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re
        self.fgs_lamb = fgs_lamb
        self.N_agents = N_agents

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.cgs.nx, self.cgs.ny, 6), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.005, high=0.005, shape=(self.N_agents, self.N_agents), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])

        #load energy spectrum
        self.means_dns = np.load(INIT_PATH_SPEC+'means_log_k5-10_dns.npy')
        stds_dns = np.load(INIT_PATH_SPEC+'stds_log_k5-10_dns.npy')
        self.cov = np.diag(stds_dns)
        self.cov_inverse = np.diag(1/stds_dns)
        assert np.any(np.isnan(self.cov_inverse)) is not True
        assert self.cov@self.cov_inverse is not np.identity(len(self.means_dns))


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)

        return state, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        interpolated_action = self.interpolate_actions(action)

        self.cgs.omega = np.copy(self.omg * (1+interpolated_action.reshape(self.omg.shape)))
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1

        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        k, E1 = energy_spectrum_2d(self.u1)
        reward = self.E_loss(E1, k)

        terminated = False
        if np.any([np.any(self.f1 < 0), np.any(self.f1 > 1), np.any(np.sqrt(np.sum(self.u1**2, axis=-1)) > 100)]):
            terminated = True
            reward -= 100.
        truncated = bool(self.counter>=self.max_episode_steps)

        return state, reward, terminated, truncated, {}

    def render(self, savefig=False):

        v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        magnitude = lambda u : np.sqrt(np.sum(u**2, axis=-1))
        # Your plotting function
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid of subplots
        # Plot CGS, FGS, and MSE fields in the first row
        im1 = axes[0, 0].imshow(v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = axes[0, 1].imshow(v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        #plot enerty spectra
        E1, E2 = self.get_spectra()
        axes[0,2].loglog(E1, label="CGS")
        axes[0,2].loglog(E2, label="FGS")
        axes[0,2].loglog(self.means_dns, label="DNS")
        axes[0,2].legend()
        axes[0,2].set_title("Energy spectra")
        axes[0,2].set_xlabel("wavenumber k")
        axes[0,2].set_ylabel("Energy E(k)")   
        # Plot velocity magnitude for CGS and FGS in the second row
        im4 = axes[1, 0].imshow(magnitude(self.u1), cmap='plasma')
        im5 = axes[1, 1].imshow(magnitude(self.u2), cmap='plasma')
        #plot velocity MSE
        im6 = axes[1, 2].imshow(np.sum((self.u1 - self.u2)**2, axis=-1), cmap='viridis')
        # Hide axes for the third column of the second row (unused)
        axes[1, 2].axis('off')
        # Remove axis ticks for all subplots
        for ax in axes.flat:
            ax.axis('off')
        axes[0,2].axis('on')
        # Set titles for the subplots
        axes[0, 0].set_title("Vorticity CGS")
        axes[0, 1].set_title("Vorticity FGS")
        axes[0, 2].set_title("Energy Spectrum")
        axes[1, 0].set_title("Velocity Magnitude CGS")
        axes[1, 1].set_title("Velocity Magnitude FGS")
        axes[1, 2].set_title("Velocity MSE")
        # Create a colorbar for the third plot (MSE)
        divider = make_axes_locatable(axes[1, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im6, cax=cax)
        # Create colorbars for velocity magnitude plots
        divider_cgs = make_axes_locatable(axes[1, 0])
        cax_cgs = divider_cgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im4, cax=cax_cgs)
        divider_fgs = make_axes_locatable(axes[1, 1])
        cax_fgs = divider_fgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im5, cax=cax_fgs)
        # Show the plot
        plt.tight_layout()
        #if save_fig == True:
        #    plt.savefig(f"visuals/img{i}.png", dpi=100)
        #    plt.close()
        #else:
        #    plt.show()
        plt.show()

    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2

    def E_loss(self, means_cgs, k):
        means_diff = np.log(means_cgs[1:]*k[1:]**5)/10 - self.means_dns
        return 1 + np.log(np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff))/64

    def interpolate_actions(self, actions):
        dist = int(self.cgs.nx // self.N_agents)  # distance between agents
        half_dist = int(dist//2)
        actions = np.pad(actions, pad_width=1, mode='wrap')
        actions = actions.flatten()
        coord = np.array([(i*dist, j*dist) for i in range(self.N_agents+2) for j in range(self.N_agents+2)])
        grid_x, grid_y = np.meshgrid(np.arange(self.cgs.nx+dist), np.arange(self.cgs.nx+dist))
        interpolated_actions = scp.interpolate.griddata(coord, actions, (grid_x, grid_y), method='cubic')
        actual_actions = interpolated_actions[half_dist:(self.cgs.nx+half_dist), half_dist:(self.cgs.nx+half_dist)]

        return actual_actions


# interpolating agents for decaying turbulence
class KolmogorovEnvironment24_decaying(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000, N_agents=8):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re
        self.fgs_lamb = fgs_lamb
        self.N_agents = N_agents

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs

        #CGS
        self.cgs = decaying_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.cgs.omg = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.cgs.nx, self.cgs.ny, 6), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.005, high=0.005, shape=(self.N_agents, self.N_agents), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])

        #load energy spectrum
        self.means_dns = np.load(INIT_PATH_SPEC+'means_log_k5-10_dns.npy')
        stds_dns = np.load(INIT_PATH_SPEC+'stds_log_k5-10_dns.npy')
        self.cov = np.diag(stds_dns)
        self.cov_inverse = np.diag(1/stds_dns)
        assert np.any(np.isnan(self.cov_inverse)) is not True
        assert self.cov@self.cov_inverse is not np.identity(len(self.means_dns))


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        self.cgs = decaying_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)

        return state, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        interpolated_action = self.interpolate_actions(action)

        self.cgs.omega = np.copy(self.omg * (1+interpolated_action.reshape(self.omg.shape)))
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1

        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        k, E1 = energy_spectrum_2d(self.u1)
        reward = self.E_loss(E1, k)

        terminated = False
        if np.any([np.any(self.f1 < 0), np.any(self.f1 > 1), np.any(np.sqrt(np.sum(self.u1**2, axis=-1)) > 100)]):
            terminated = True
            reward -= 100.
        truncated = bool(self.counter>=self.max_episode_steps)

        return state, reward, terminated, truncated, {}

    def render(self, savefig=False):

        v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        magnitude = lambda u : np.sqrt(np.sum(u**2, axis=-1))
        # Your plotting function
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid of subplots
        # Plot CGS, FGS, and MSE fields in the first row
        im1 = axes[0, 0].imshow(v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = axes[0, 1].imshow(v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        #plot enerty spectra
        E1, E2 = self.get_spectra()
        axes[0,2].loglog(E1, label="CGS")
        axes[0,2].loglog(E2, label="FGS")
        axes[0,2].loglog(self.means_dns, label="DNS")
        axes[0,2].legend()
        axes[0,2].set_title("Energy spectra")
        axes[0,2].set_xlabel("wavenumber k")
        axes[0,2].set_ylabel("Energy E(k)")   
        # Plot velocity magnitude for CGS and FGS in the second row
        im4 = axes[1, 0].imshow(magnitude(self.u1), cmap='plasma')
        im5 = axes[1, 1].imshow(magnitude(self.u2), cmap='plasma')
        #plot velocity MSE
        im6 = axes[1, 2].imshow(np.sum((self.u1 - self.u2)**2, axis=-1), cmap='viridis')
        # Hide axes for the third column of the second row (unused)
        axes[1, 2].axis('off')
        # Remove axis ticks for all subplots
        for ax in axes.flat:
            ax.axis('off')
        axes[0,2].axis('on')
        # Set titles for the subplots
        axes[0, 0].set_title("Vorticity CGS")
        axes[0, 1].set_title("Vorticity FGS")
        axes[0, 2].set_title("Energy Spectrum")
        axes[1, 0].set_title("Velocity Magnitude CGS")
        axes[1, 1].set_title("Velocity Magnitude FGS")
        axes[1, 2].set_title("Velocity MSE")
        # Create a colorbar for the third plot (MSE)
        divider = make_axes_locatable(axes[1, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im6, cax=cax)
        # Create colorbars for velocity magnitude plots
        divider_cgs = make_axes_locatable(axes[1, 0])
        cax_cgs = divider_cgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im4, cax=cax_cgs)
        divider_fgs = make_axes_locatable(axes[1, 1])
        cax_fgs = divider_fgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im5, cax=cax_fgs)
        # Show the plot
        plt.tight_layout()
        #if save_fig == True:
        #    plt.savefig(f"visuals/img{i}.png", dpi=100)
        #    plt.close()
        #else:
        #    plt.show()
        plt.show()

    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2

    def E_loss(self, means_cgs, k):
        means_diff = np.log(means_cgs[1:]*k[1:]**5)/10 - self.means_dns
        return 1 + np.log(np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff))/64

    def interpolate_actions(self, actions):
        dist = int(self.cgs.nx // self.N_agents)  # distance between agents
        half_dist = int(dist//2)
        actions = np.pad(actions, pad_width=1, mode='wrap')
        actions = actions.flatten()
        coord = np.array([(i*dist, j*dist) for i in range(self.N_agents+2) for j in range(self.N_agents+2)])
        grid_x, grid_y = np.meshgrid(np.arange(self.cgs.nx+dist), np.arange(self.cgs.nx+dist))
        interpolated_actions = scp.interpolate.griddata(coord, actions, (grid_x, grid_y), method='cubic')
        actual_actions = interpolated_actions[half_dist:(self.cgs.nx+half_dist), half_dist:(self.cgs.nx+half_dist)]

        return actual_actions


# velocity error
class KolmogorovEnvironment25(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1, max_episode_steps=20000, seed=102, fgs_lamb=16, cgs_lamb=1, seeds=np.array([102]), Re=10000, N_agents=8):
        super().__init__()

        self.possible_seeds = seeds #add seeds as argument
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.Re = Re
        self.fgs_lamb = fgs_lamb
        self.N_agents = N_agents

        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N{int(self.fgs_lamb*128)}_S{self.sampled_seed}_U1_dump/"
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs
        self.kwargs2, _, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=fgs_lamb, Re=self.Re) #fgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        print(f"omega shape = {self.omg.shape}")
        self.cgs.omg = np.copy(self.omg)
        print(f"cgs omega shape = {self.cgs.omg.shape}")
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        #self.state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        print(f"f shape = {self.f1.shape}")
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.cgs.nx, self.cgs.ny, 6), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.005, high=0.005, shape=(self.N_agents, self.N_agents), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])

        #FGS
        self.u2 = self._load_u2()
        #load enerty spectrum
        self.means_dns = np.load(INIT_PATH_SPEC+'means_log_k5-10_dns.npy')
        stds_dns = np.load(INIT_PATH_SPEC+'stds_log_k5-10_dns.npy')
        self.cov = np.diag(stds_dns)
        self.cov_inverse = np.diag(1/stds_dns)
        assert np.any(np.isnan(self.cov_inverse)) is not True
        assert self.cov@self.cov_inverse is not np.identity(len(self.means_dns))



    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        #print(f"********** reset at after {self.counter} steps **********")
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
        #self.fgs_dump_path = FGS_DATA_PATH_3 + f"re{self.Re}_T227_N{int(self.fgs_lamb*128)}_S{self.sampled_seed}_U1_dump/"
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.cgs.omega = np.copy(self.omg)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)

        return state, {}
    
    def step(self, action):
        if action.shape != self.action_space.shape:
            try:
                action = action.reshape(self.action_space.shape)
            except:
                print("action reshaping didn't work")

        if (np.any(self.action_space.low > action) or np.any(action > self.action_space.high)):
            print("WARNING: Action is not in action space")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        interpolated_action = self.interpolate_actions(action)

        self.cgs.omega = np.copy(self.omg * (1+interpolated_action.reshape(self.omg.shape)))
        #self.cgs.omega = np.copy(self.omg * (1+action))
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1

        self.u2 = self._load_u2()

        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        #k, E1 = energy_spectrum_2d(self.u1)
        #reward = self.E_loss(E1, k)
        err2 = np.sum((self.u1 - self.u2)**2, axis=-1)
        reward = (1 - err2).mean()

        terminated = False
        if np.any([np.any(self.f1 < 0), np.any(self.f1 > 1), np.any(np.sqrt(np.sum(self.u1**2, axis=-1)) > 100)]):
            terminated = True
            reward -= 100.
        truncated = bool(self.counter>=self.max_episode_steps)

        return state, reward, terminated, truncated, {}

    def render(self, savefig=False):

        v1 = vorticity_2d(self.u1, self.kwargs1["dx_eff"])
        v2 = vorticity_2d(self.u2, self.kwargs2["dx_eff"])
        magnitude = lambda u : np.sqrt(np.sum(u**2, axis=-1))
        # Your plotting function
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid of subplots
        # Plot CGS, FGS, and MSE fields in the first row
        im1 = axes[0, 0].imshow(v1, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        im2 = axes[0, 1].imshow(v2, vmin=-10, vmax=10, cmap=sn.cm.icefire)
        #plot enerty spectra
        E1, E2 = self.get_spectra()
        axes[0,2].loglog(E1, label="CGS")
        axes[0,2].loglog(E2, label="FGS")
        axes[0,2].loglog(self.means_dns, label="DNS")
        axes[0,2].legend()
        axes[0,2].set_title("Energy spectra")
        axes[0,2].set_xlabel("wavenumber k")
        axes[0,2].set_ylabel("Energy E(k)")   
        # Plot velocity magnitude for CGS and FGS in the second row
        im4 = axes[1, 0].imshow(magnitude(self.u1), cmap='plasma')
        im5 = axes[1, 1].imshow(magnitude(self.u2), cmap='plasma')
        #plot velocity MSE
        im6 = axes[1, 2].imshow(np.sum((self.u1 - self.u2)**2, axis=-1), cmap='viridis')
        # Hide axes for the third column of the second row (unused)
        axes[1, 2].axis('off')
        # Remove axis ticks for all subplots
        for ax in axes.flat:
            ax.axis('off')
        axes[0,2].axis('on')
        # Set titles for the subplots
        axes[0, 0].set_title("Vorticity CGS")
        axes[0, 1].set_title("Vorticity FGS")
        axes[0, 2].set_title("Energy Spectrum")
        axes[1, 0].set_title("Velocity Magnitude CGS")
        axes[1, 1].set_title("Velocity Magnitude FGS")
        axes[1, 2].set_title("Velocity MSE")
        # Create a colorbar for the third plot (MSE)
        divider = make_axes_locatable(axes[1, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im6, cax=cax)
        # Create colorbars for velocity magnitude plots
        divider_cgs = make_axes_locatable(axes[1, 0])
        cax_cgs = divider_cgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im4, cax=cax_cgs)
        divider_fgs = make_axes_locatable(axes[1, 1])
        cax_fgs = divider_fgs.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im5, cax=cax_fgs)
        # Show the plot
        plt.tight_layout()
        #if save_fig == True:
        #    plt.savefig(f"visuals/img{i}.png", dpi=100)
        #    plt.close()
        #else:
        #    plt.show()
        plt.show()

    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])

    def get_spectra(self):
        _, E1 = energy_spectrum_2d(self.u1)
        _, E2 = energy_spectrum_2d(self.u2)
        return E1, E2

    def E_loss(self, means_cgs, k):
        means_diff = np.log(means_cgs[1:]*k[1:]**5)/10 - self.means_dns
        return 1 + np.log(np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff))/64
        #return np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff)

    def interpolate_actions(self, actions):
        dist = int(self.cgs.nx // self.N_agents)  # distance between agents
        half_dist = int(dist//2)
        actions = np.pad(actions, pad_width=1, mode='wrap')
        actions = actions.flatten()
        coord = np.array([(i*dist, j*dist) for i in range(self.N_agents+2) for j in range(self.N_agents+2)])
        grid_x, grid_y = np.meshgrid(np.arange(self.cgs.nx+dist), np.arange(self.cgs.nx+dist))
        interpolated_actions = scp.interpolate.griddata(coord, actions, (grid_x, grid_y), method='cubic')
        actual_actions = interpolated_actions[half_dist:(self.cgs.nx+half_dist), half_dist:(self.cgs.nx+half_dist)]

        return actual_actions
