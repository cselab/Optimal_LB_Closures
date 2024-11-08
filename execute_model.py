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

sampled_seed=102
train_seeds = np.array([33])
Re = 10000
N = 128
T = 227
seed = 33
upsi = 1
cgs_lamb=1
fgs_lamb=16
step_factor=1

vel_ref= 0.1*(1/np.sqrt(3))
n = 4
C_u = vel_ref/n # velocity transformation factor

precision = "f64/f64"
lattice = LatticeD2Q9(precision)
c = jnp.array(lattice.c, dtype=jnp.float64)
cc = jnp.array(lattice.cc, dtype=jnp.float64)
w = jnp.array(lattice.w, dtype=jnp.float64)

u0_path = INIT_PATH + f"velocity_burn_in_909313_s{sampled_seed}.npy" #2048x2048 simulation
rho0_path = INIT_PATH + f"density_burn_in_909313_s{sampled_seed}.npy" #2048x2048 simulation
kwargs1, endTime1, _, _ = get_kwargs4(u0_path=u0_path, rho0_path=rho0_path, T_wish=227, lamb=cgs_lamb, Re=Re) #cgs
cgs = Kolmogorov_flow(**kwargs1)

#omg = np.copy(cgs.omega*np.ones((cgs.nx, cgs.ny, 1)))
omg = jnp.array(cgs.omega, copy=True)
#omg = cgs.omega * jnp.ones((cgs.nx, cgs.ny, 1))
#jnp.array(x, copy=True)
#cgs.omgega = omg
f1 = cgs.assign_fields_sharded()

rho1, u1 = update_macroscopic(f1, c)
fneq = f1 - equilibrium(rho1, u1, c, w)
P_neq1 = momentum_flux(fneq, cc)
state = jnp.concatenate((rho1, u1/C_u, P_neq1), axis=-1)
state = jax2torch(state).to(torch.float32)

#other stuff  
factor = int(fgs_lamb/cgs_lamb)

#actor = FullyConvNet_interpolating_agents(in_channels=6, N=args.num_agents, device=device).to(device)
actor = local_actor_net_fast(in_channels=6, device=device).to(device)

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
actor.load_state_dict(actor_parameters)

if not os.path.exists(f"re{int(Re)}_T{int(T)}_N{int(N)}_S{seed}_U{upsi}_dump_scaled_policy"):
    os.makedirs(f"re{int(Re)}_T{int(T)}_N{int(N)}_S{seed}_U{upsi}_dump_scaled_policy")
os.chdir(f"re{int(Re)}_T{int(T)}_N{int(N)}_S{seed}_U{upsi}_dump_scaled_policy")

# just plays one episode
counter=0
step = 0
actor.eval()
episode_is_over = False
m = 20025
io_rate = 32
for step in tqdm(range(m)):
    act = actor(obs = state.reshape(1,-1,128,128))
    act = torch2jax(act)
    #print(act)
    #print(act.shape)

    cgs.omega = np.copy(omg * (1+act.reshape(cgs.nx, cgs.ny, 1)))
    for _ in range(step_factor):
        f1, _ = cgs.step(f1, counter, return_fpost=cgs.returnFpost)
        counter += 1

    rho1, u1 = update_macroscopic(f1, c)
    fneq = f1 - equilibrium(rho1, u1, c, w)
    P_neq1 = momentum_flux(fneq, cc)
    state = jnp.concatenate((rho1, u1/C_u, P_neq1), axis=-1)
    state = jax2torch(state).to(torch.float32)

    #state = np.concatenate((rho1, u1, P_neq1), axis=-1)

    #if step%io_rate==0:
    #   #save velocity field as npy
    #    fname = "klmgrv"
    #    fname = "velocity_" + fname
    #    fname = fname + "_" + f"s{seed}"
    #    fname = fname + "_" + str(step).zfill(6)
    #    #u1 = downsample_field(test_env.u1, 2)
    #    np.save(fname, u1)
