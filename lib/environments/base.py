import gymnasium as gym
#import wandb
import numpy as np
import scipy as scp
from abc import ABC, abstractmethod
import os
import sys
from tianshou.data import Batch
from gymnasium import spaces
import wandb
from tqdm import tqdm

#temporary solution for xlb imports
sys.path.append(os.path.abspath(os.path.expanduser('~/XLB')))
from my_flows.kolmogorov_2d import Kolmogorov_flow, Decaying_flow
from my_flows.helpers import get_vorticity, get_velocity, get_kwargs4, get_moments, get_raw_moments
from src.utils import *


class BaseEnvironment(ABC, gym.Env):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def reset(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, action):
        pass

    def _get_info(self):
        return {}


# path to the initialization files
INIT_PATH = os.path.expanduser("~/XLB/vel_init/")
FGS_DATA_PATH = os.path.expanduser("~/XLB/fgs_data/")
FGS_DATA_PATH_3 = os.path.expanduser("~/XLB/fgs3_data/")
# path to energy dns energyp spectrum
INIT_PATH_SPEC = os.path.expanduser("~/XLB/dns_spectrum/")


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
        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" 
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" 
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path,
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
        self.means_dns = np.load(INIT_PATH_SPEC+'means_log_k5-10_dns.npy')
        stds_dns = np.load(INIT_PATH_SPEC+'stds_log_k5-10_dns.npy')
        cov = np.diag(stds_dns)
        self.cov_inverse = np.diag(1/stds_dns)
        assert np.any(np.isnan(self.cov_inverse)) is not True
        assert cov@self.cov_inverse is not np.identity(len(self.means_dns))
    
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
        self.kwargs1["u0_path"] = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy"
        self.kwargs1["rho0_path"] = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy"
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
        if self.cgs_lamb > 1:
            k, E1 = energy_spectrum_2d(downsample_field(self.u1, self.cgs_lamb))
        else:
            k, E1 = energy_spectrum_2d(self.u1)
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
        print((-0.5 * means_diff.T @ self.cov_inverse @ means_diff))
        return 1 + np.log(np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff))/64
    
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