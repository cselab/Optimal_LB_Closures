import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
#temporary solution for xlb imports
#sys.path.append(os.path.abspath('/home/pfischer/XLB'))
sys.path.append(os.path.abspath(os.path.expanduser('~/XLB')))
from my_flows.kolmogorov_2d import Kolmogorov_flow, Kolmogorov_flow_KBC, decaying_flow
from my_flows.helpers import get_kwargs, get_vorticity, get_velocity, get_kwargs4, get_moments, get_raw_moments, update_macroscopic, momentum_flux, equilibrium
from src.utils import *
from src.lattice import LatticeD2Q9
from lib.models import *
device = "cuda" if torch.cuda.is_available() else "cpu"
from collections import OrderedDict
from tianshou.data import Batch
import jax.numpy as jnp
import torch
import jax
import time
import argparse
from functools import partial
from jax import jit

jax.config.update("jax_spmd_mode", 'allow_all')

def jax2torch(x):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))

def torch2jax(x):
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))


# PPO run for local actions and scaled loss
DUMP_PATH = "dump/Kolmogorov22_ppo_cgs1_fgs16/"
ID = "20241021-110311"
#INIT_PATH = "/home/pfischer/XLB/vel_init/"
INIT_PATH = os.path.expanduser("~/XLB/vel_init/")

#policy.load_state_dict(torch.load(DUMP_PATH+'policy_'+ID+'.pth'))
with open(DUMP_PATH+'config_'+ID+'.pkl', 'rb') as f:
    args = pickle.load(f)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()  
    parser.add_argument("--lamb", type=int, default=1) #scales spacial resolution
    return parser.parse_known_args()[0]


class execute_model():
    def __init__(self, exec_args):
        
        sampled_seed=102
        train_seeds = np.array([33])
        Re = 10000
        self.N = int(128*exec_args.lamb)
        T = 227
        self.seed = 33
        upsi = 1
        cgs_lamb=exec_args.lamb
        fgs_lamb=16
        self.step_factor=1
        vel_ref= 0.1*(1/np.sqrt(3))
        n = 4
        self.C_u = vel_ref/n # velocity transformation factor
        precision = "f64/f64"
        lattice = LatticeD2Q9(precision)
        self.c = jnp.array(lattice.c, dtype=jnp.float64)
        self.cc = jnp.array(lattice.cc, dtype=jnp.float64)
        self.w = jnp.array(lattice.w, dtype=jnp.float64)
        u0_path = INIT_PATH + f"velocity_burn_in_909313_s{sampled_seed}.npy" #2048x2048 simulation
        rho0_path = INIT_PATH + f"density_burn_in_909313_s{sampled_seed}.npy" #2048x2048 simulation
        kwargs1, self.endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=Re) #cgs
        self.cgs = Kolmogorov_flow(**kwargs1)
        self.omg = jnp.array(self.cgs.omega, copy=True)
        self.f1 = self.cgs.assign_fields_sharded()
        factor = int(fgs_lamb/cgs_lamb)
        self.counter = 0

        #actor = FullyConvNet_interpolating_agents(in_channels=6, N=args.num_agents, device=device).to(device)
        self.actor = local_actor_net_fast(in_channels=6, device=device, nx=self.N).to(device)
        #load actor from state_dict
        actor_parameters = OrderedDict()
        state_dict = torch.load(DUMP_PATH+'policy_'+ID+'.pth')
        for key, value in state_dict.items():
            if key.startswith("actor."):
                new_key = key.replace("actor.", "")  # Remove "actor." prefix
            elif key.startswith("_actor_critic.actor."):
                new_key = key.replace("_actor_critic.actor.", "")  # Remove "_actor_critic.actor." prefix
            else:
                continue  # Skip keys not related to the actor
            actor_parameters[new_key] = value

        # Load the modified state dict into your actor network
        self.actor.load_state_dict(actor_parameters)

        if not os.path.exists(f"re{int(Re)}_T{int(T)}_N{int(self.N)}_S{self.seed}_U{upsi}_dump_scaled_policy"):
            os.makedirs(f"re{int(Re)}_T{int(T)}_N{int(self.N)}_S{self.seed}_U{upsi}_dump_scaled_policy")
        os.chdir(f"re{int(Re)}_T{int(T)}_N{int(self.N)}_S{self.seed}_U{upsi}_dump_scaled_policy")


    @partial(jit, static_argnums=(0,))
    def get_state(self):
        rho1 =jnp.sum(self.f1, axis=-1, keepdims=True)
        u1 = jnp.dot(self.f1, self.c.T) / rho1
        cu = 3.0 * jnp.dot(u1, self.c)
        usqr = 1.5 * jnp.sum(jnp.square(u1), axis=-1, keepdims=True)
        feq = rho1 * self.w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
        fneq = self.f1-feq
        P_neq1 = jnp.dot(fneq, self.cc)
        state = jnp.concatenate((rho1, u1/self.C_u, P_neq1), axis=-1)
        return state


    def run_simulation(self):
        # just plays one episode
        step = 0
        self.actor.eval()
        #io_rate = 32*cgs_lamb
        io_rate = self.endTime1-1
        state = self.get_state()
        state = jax2torch(state).to(torch.float32)
        start_time = time.time()
        for step in tqdm(range(self.endTime1)):
            act = self.actor(obs = state.reshape(1,-1,self.N,self.N))
            act = torch2jax(act)
            self.cgs.omega = self.omg * (1+act.reshape(self.N, self.N, 1))
            for _ in range(self.step_factor):
                self.f1, _ = self.cgs.step(self.f1, self.counter, return_fpost=False)
                self.counter += 1
            state = self.get_state()

            state = jax2torch(state).to(torch.float32)

            if step%io_rate==0:
               #save velocity field as npy
                fname = "klmgrv"
                fname = "velocity_" + fname
                fname = fname + "_" + f"s{self.seed}"
                fname = fname + "_" + str(step).zfill(6)
                #u1 = downsample_field(test_env.u1, 2)
                np.save(fname, state[1:2,...].detach().cpu().numpy())

        end_time = time.time()

        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")


if __name__ == "__main__":

    exec_args = get_args()
    sim = execute_model(exec_args)
    sim.run_simulation()