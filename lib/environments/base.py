import gymnasium as gym
#import wandb
import numpy as np
from abc import ABC, abstractmethod
import os
#from tqdm import tqdm
#from tianshou.data import Batch
from gymnasium import spaces

#temporary solution for xlb imports
sys.path.append(os.path.abspath(os.path.expanduser('~/XLB')))
from my_flows.kolmogorov_2d import Kolmogorov_flow, decaying_flow
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
# path to energy spectra
INIT_PATH_SPEC = os.path.expanduser("~/XLB/dns_spectrum/")


# base KolmogorvEnvironment for energy spectrum lass
class Base_KolmogorovEnvironment(BaseEnvironment, ABC):
    
    def __init__(self, step_factor=1,
                max_episode_steps=20000,
                seed=102,
                fgs_lamb=16,
                cgs_lamb=1,
                seeds=np.array([102]),
                Re=10000):
        super().__init__()

        #for random initialization sample a seed from possible seeds
        self.possible_seeds = seeds
        self.sampled_seed = np.random.choice(self.possible_seeds) 
        #setup
        self.Re = Re
        self.fgs_lamb = fgs_lamb
        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{self.sampled_seed}.npy" #2048x2048 simulation
        self.kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=self.Re) #cgs

        #CGS
        self.cgs = Kolmogorov_flow(**self.kwargs1)
        self.omg = np.copy(self.cgs.omega*np.ones((self.cgs.nx, self.cgs.ny, 1)))
        self.f1 = self.cgs.assign_fields_sharded()
        self.rho1, self.u1, self.P_neq1 = get_moments(self.f1, self.cgs)
        
        #other stuff  
        self.factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.cgs.nx, self.cgs.ny, 6), dtype=np.float64)
        self.action_space = spaces.Box(low=-0.005, high=0.005, shape=(self.cgs.nx, self.cgs.ny), dtype=np.float32)
        self.step_factor = step_factor
        self.max_episode_steps = np.min([max_episode_steps, endTime1])

        #enerty spectrum
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
        self.cgs.omega = self.omg * (1+action.reshape(self.omg.shape))
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
        return 0
    
    def _load_u2(self):
        u2 = np.load(self.fgs_dump_path + f"velocity_klmgrv_s{self.sampled_seed}_{str(int(self.counter*self.factor)).zfill(6)}.npy")
        return u2
    
    def get_vorticity(self):
        return vorticity_2d(self.u1, self.kwargs1["dx_eff"])

    def E_loss(self, means_cgs, k):
        means_diff = np.log(means_cgs[1:]*k[1:]**5)/10 - self.means_dns
        return 1 + np.log(np.exp(-0.5 * means_diff.T @ self.cov_inverse @ means_diff))/64


