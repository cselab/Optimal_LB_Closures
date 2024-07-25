from abc import ABC
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from tqdm import tqdm
import wandb
import sys
import os

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit, RescaleAction, TransformObservation
from stable_baselines3.common.env_checker import check_env

from tianshou.data import Batch
from torch.utils.data import DataLoader
from scipy.interpolate import griddata
from typing import Tuple

from lib.utils import sample_from_dataloader
from lib.environments.base import BaseEnvironment
from lib.environments.velocity_generation import VelocityFieldGenerator
from lib.datasets import get_train_val_test_initial_conditions_dataset
from lib.models.wrappers import MarlModel

#temporary solution for xlb imports
sys.path.append(os.path.abspath('/home/pfischer/XLB'))
from my_flows.kolmogorov_2d import Kolmogorov_flow
from my_flows.helpers import get_kwargs, get_vorticity, get_velocity
from src.utils import *


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
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny,1)))
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
        self.action_space = spaces.Box(low=0.9, high=1.1, shape=(self.cgs.nx, self.cgs.ny, 1), dtype=np.float32)
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
        if not (np.any(self.action_space.low <= action) and np.any(action <= self.action_space.high)):
            print("WARNING: Action is not in action space")
            print(f"action={action}; omega={self.cgs.omega}")
            action = np.clip(action, self.action_space.low, self.action_space.high)
            

        # load in action and get rid of channel dimension
        #print(action.shape, self.action_space.shape)
        assert action.shape == self.action_space.shape
        self.cgs.omega = np.copy(self.omg * action)
        
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



def main():
    #here a trivial run of the environment is displayed
    #path to initial velocity and density field
    u0_path = "/home/pfischer/XLB/vel_init/velocity_burn_in_1806594.npy" #4096x4096 simulation
    rho0_path = "/home/pfischer/XLB/vel_init/density_burn_in_1806594.npy" #4096x4096 simulation

    kwargs1, _,_,_ = get_kwargs(u0_path=u0_path, rho0_path=rho0_path, lamb=1) #cgs 
    kwargs2, _,_,_ = get_kwargs(u0_path=u0_path, rho0_path=rho0_path, lamb=2) #fgs

    env = KolmogorovEnvironment(kwargs1, kwargs2)
    env = TimeLimit(env, max_episode_steps=5000)
    env = RescaleAction(env, min_action=-1., max_action=1.)
    env = TransformObservation(env, lambda obs: (obs/20))

    check_env(env, warn=True)

    obs ,_ = env.reset()
    omg = env.unwrapped.omg
    for i in tqdm(range(1000)):
        obs, _, _, _, _  = env.step([0.])

        if i%250 == 0:
            #print(act)
            env.render()



if __name__ == "__main__":
    main()