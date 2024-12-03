from abc import ABC
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import scipy as scp
from tqdm import tqdm
import wandb
import sys
import os
import torch
import gc
import jax.numpy as jnp
import jax
from jax import jit
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gymnasium import spaces
from lib.environments.base import BaseEnvironment
#from XLB.src.utils import *
from xlb_flows.utils import vorticity_2d, get_kwargs, get_moments, energy_spectrum_2d, downsample_field
from xlb_flows.kolmogorov_2d import Kolmogorov_flow, Decaying_flow


def jax2torch(x):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))

def torch2jax(x):
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))


# path to the initialization files
#TODO: change this after new burn in simulation
INIT_PATH = os.path.expanduser(
    "~/CNN-MARL_closure_model_discovery/"
    "xlb_flows/init_fields/")

#INIT_PATH_SPEC = os.path.expanduser(
#    "~/CNN-MARL_closure_model_discovery/"
#    "xlb_flows/dns_spectrum/")
INIT_PATH_SPEC = os.path.expanduser(
    "~/CNN-MARL_closure_model_discovery/"
    "results/dns_spectra/")

#FGS_DATA_PATH = os.path.expanduser("~/XLB/fgs_data/")
#FGS_DATA_PATH_3 = os.path.expanduser("~/XLB/fgs3_data/")
# path to energy spectra


# base KolmogorvEnvironment for energy spectrum loss
class KolmogorovEnvironment(BaseEnvironment, ABC):
    
    def __init__(self,
                 step_factor=1,
                 max_episode_steps=20000,
                 seed=102,
                 fgs_lamb=16,
                 cgs_lamb=1,
                 seeds=np.array([102]),
                 Re=10000,
                 N_agents=128,
                 flow="Kolmogorov"):
        super().__init__()

        #for random initialization sample a seed from possible seeds
        self.possible_seeds = seeds
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        #variables
        self.Re = Re
        self.flow = flow
        self.step_factor = step_factor
        self.counter = 0
        self.N_agents = N_agents
        self.N = (cgs_lamb*128)
        self.cgs_lamb = cgs_lamb
        #CGS parameters
        u0_path = INIT_PATH + f"velocity_kolmogorov_2d_910368_s{self.sampled_seed}.npy" 
        rho0_path = INIT_PATH + f"density_kolmogorov_2d_910368_s{self.sampled_seed}.npy" 
        self.kwargs1, endTime1, _, _ = get_kwargs(u0_path=u0_path,
                                                    rho0_path=rho0_path,
                                                    T_wish=227,
                                                    lamb=cgs_lamb,
                                                    Re=self.Re)
        #CGS
        if self.flow == "Kolmogorov":
            self.cgs = Kolmogorov_flow(**self.kwargs1)
        elif self.flow == "Decaying":
            self.cgs = Decaying_flow(**self.kwargs1)
        #state and action
        if self.N_agents == 1:
            self.omg = np.copy(self.cgs.omega)
            action_shape = (1,)
        else:
            self.omg = np.copy(self.cgs.omega*np.ones((self.N, self.N, 1)))
            action_shape = (self.N_agents, self.N_agents)
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        #reward - enerty spectrum
        #self.means_dns = np.load(INIT_PATH_SPEC+'means_log_k5-10_dns.npy')
        self.means_dns = np.load(INIT_PATH_SPEC+'dns_mean_scaled.npy')
        #self.means_dns = np.load(INIT_PATH_SPEC+'dns_mean.npy')
        stds_dns = np.load(INIT_PATH_SPEC+'dns_std_scaled_prior.npy')
        #stds_dns = np.load(INIT_PATH_SPEC+'stds_log_k5-10_dns.npy')
        #k = np.linspace(0,62, 63)
        #take values from 1: to avoid divission by zero
        #self.means_dns = np.log((self.means_dns[1:]*k[1:]**5)/10)
        #stds_dns = np.log((stds_dns[1:]*k[1:]**5)/10)
        stds_dns = np.abs(stds_dns)
        #cov = np.diag(stds_dns)
        self.cov_inverse = np.diag(1/stds_dns)
        #assert np.any(np.isnan(self.cov_inverse)) is not True
        #assert cov@self.cov_inverse is not np.identity(len(self.means_dns))
    
        #Environment specifications
        self.observation_space = spaces.Box(low=-3,
                                            high=3,
                                             shape=(self.N, self.N, 6),
                                             dtype=np.float64)
        self.action_space = spaces.Box(low=-0.005,
                                       high=0.005,
                                       shape=action_shape,
                                       dtype=np.float32)
        self.max_episode_steps = np.min([max_episode_steps, endTime1])


    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.counter = 0
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_kolmogorov_2d_910368_s{self.sampled_seed}.npy" 
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_kolmogorov_2d_910368_s{self.sampled_seed}.npy" 
        if self.flow == "Kolmogorov":
            self.cgs = Kolmogorov_flow(**self.kwargs1)
        elif self.flow == "Decaying":
            self.cgs = Decaying_flow(**self.kwargs1)
        if self.N_agents == 1:
            self.omg = np.copy(self.cgs.omega)
        else:
            self.omg = np.copy(self.cgs.omega*np.ones((self.N, self.N, 1)))
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        return state, {}
    
    def step(self, action):
        #interpolate action
        if self.N_agents != 1 and self.N_agents != self.N:
            action = self.interpolate_actions(action)
        #update relaxation rate of cgs
        self.cgs.omega = self.omg * (1+action.reshape(self.omg.shape))
        #perfrom #step_factor cgs steps for given action
        for _ in range(self.step_factor):
            self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=self.cgs.returnFpost)
            self.counter += 1
        #compute state and reward
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        state = np.concatenate((self.rho1,self.u1, self.P_neq1), axis=-1)
        k, E1 = energy_spectrum_2d(downsample_field(self.u1, self.cgs_lamb))
        #reward = self.E_loss(E1, k)
        reward = self.E_loss(E1, k)
        terminated = False
        if np.any([np.any(self.f1 < 0),
                    np.any(self.f1 > 1),
                    np.any(np.sqrt(np.sum(self.u1**2, axis=-1)) > 100)]):
            terminated = True
            reward -= 100.
        truncated = bool(self.counter>=self.max_episode_steps)

        return state, reward, terminated, truncated, {}

    def render(self, savefig=False):
        return 0
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])

    def E_loss(self, means_cgs, k):
        means_diff = (np.log(means_cgs[1:]*k[1:]**5)/10) - self.means_dns
        #expo = np.max([np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff),1e-12])
        expo = np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff)
        return 1 + np.log(expo)/64
    
    def E_loss_2(self, means_cgs, k, alpha=0.4):
        mse = (((means_cgs - self.means_dns)/self.means_dns)**2).sum()
        return 2*np.exp((-1/(alpha*64))*mse) - 1

    
    def interpolate_actions(self, actions):
        dist = int(self.N // self.N_agents) #distance between agents
        half_dist = int(dist//2)
        actions = np.pad(actions, pad_width=1, mode='wrap')
        actions = actions.flatten()
        coord = np.array([(i*dist, j*dist) for i in range(self.N_agents+2) for j in range(self.N_agents+2)])
        grid_x, grid_y = np.meshgrid(np.arange(self.cgs.nx+dist), np.arange(self.N+dist))
        interpolated_actions = scp.interpolate.griddata(coord, actions, (grid_x, grid_y), method='cubic')
        actual_actions = interpolated_actions[half_dist:(self.N+half_dist), half_dist:(self.N+half_dist)]
        return actual_actions