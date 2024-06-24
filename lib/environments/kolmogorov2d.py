from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import wandb
import sys
import tqdm
import seaborn as sn

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit, RescaleAction

from tianshou.data import Batch
from torch.utils.data import DataLoader
from scipy.interpolate import griddata
from typing import Tuple

from lib.utils import sample_from_dataloader
from lib.environments.base import BaseEnvironment
from lib.environments.velocity_generation import VelocityFieldGenerator
from lib.datasets import get_train_val_test_initial_conditions_dataset
from lib.models.wrappers import MarlModel

sys.path.append('/home/pfischer/XLB')
from my_flows.kolmogorov_2d import Kolmogorov_flow
from my_flows.helpers import get_kwargs, get_vorticity



class KolmogorovEnvironment(gym.Env):
    metadata = {}
    
    def __init__(self, kwargs1, kwargs2):
        super().__init__()
        self.cgs = Kolmogorov_flow(**kwargs1)
        self.fgs = Kolmogorov_flow(**kwargs2)
        self.f1 = self.cgs.assign_fields_sharded()
        self.f2 = self.fgs.assign_fields_sharded()
        self.beta = self.cgs.omega
        self.factor = int(self.fgs.downsamplingFactor/self.cgs.downsamplingFactor)
        self.counter = 0
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Box(low=0, high=2, shape=(1,), dtype=np.float32)


    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.counter = 0
        self.f1 = self.cgs.assign_fields_sharded()
        self.f2 = self.fgs.assign_fields_sharded()
        v1 = get_vorticity(self.f1, self.cgs)
        #return v1, info
        info = {}
        return 1, info
    
    def step(self, action):
        self.cgs.omega = np.float64(action[0])
        self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
        for i in range(self.factor):
            self.f2, _ = self.fgs.step(self.f2, self.counter+i, return_fpost=self.fgs.returnFpost)
        self.counter += 1

        v1 = get_vorticity(self.f1, self.cgs)
        v2 = get_vorticity(self.f2, self.fgs)
        corr = np.corrcoef(v1.flatten(), v2.flatten())[0, 1]

        terminated = bool(corr<0.95)

        reward = corr
        if terminated:
            reward -= 1000

        info = {}
        return 1, reward, terminated, False, info

    def render(self, save=True):
        v1 = get_vorticity(self.f1, self.cgs)
        v2 = get_vorticity(self.f2, self.fgs)
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
        if save:
            plt.savefig(f'results_dump/my_plot{self.counter}.png', bbox_inches='tight', dpi=100)
            plt.close()
        else:
            plt.show()


def main():
    ################################## parameters to choose #######################################
    #path to initial velocity and density field
    u0_path = "/home/pfischer/XLB/vel_init/velocity_burn_in_1806594.npy" #4096x4096 simulation
    rho0_path = "/home/pfischer/XLB/vel_init/density_burn_in_1806594.npy" #4096x4096 simulation

    Re = 1000 # Reynolds number
    lamb = 2 # resolution factor
    n = 4 # number of vorticies in kolmogorov forcing
    desired_time = 22 # when multiplied by n**2, it gives the non-dim time
    vel_ref = 0.1*(1/np.sqrt(3)) # corresponds to a Mach number of 0.1
    ###############################################################################################
    
    kwargs1, _,_,_ = get_kwargs(u0_path=u0_path, rho0_path=rho0_path, lamb=1, desired_time=desired_time, Re=Re, n=n, vel_ref=vel_ref)
    kwargs2, _,_,_ = get_kwargs(u0_path=u0_path, rho0_path=rho0_path, lamb=1, desired_time=desired_time, Re=Re, n=n, vel_ref=vel_ref)

    env = KolmogorovEnvironment(kwargs1, kwargs2)
    env = TimeLimit(env, max_episode_steps=1000)
    env = RescaleAction(env, min_action=-1., max_action=1.)

    obs ,_ = env.reset()
    for i in range(1000):
        act = env.action_space.sample()
        #print(act, act.shape, act[0].shape, env.action_space.shape)
        obs, _, _, _, _  = env.step(act[0])
        if i%100 == 0:
            print(f"action = {act[0]}")
            env.render()


if __name__ == "__main__":
    main()
