#imports
import os
import pickle
import numpy as np
from tqdm import tqdm
import argparse

from lib.models import *
from tianshou.policy import PPOPolicy
from lib.environments.kolmogorov import KolmogorovEnvironment, KolmogorovEnvironment2, KolmogorovEnvironment3
from xlb_flows.utils import create_and_navigate_to
from XLB.src.utils import downsample_field

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--Re", type=int, default=1e4)
    parser.add_argument("--upsi", type=int, default=1) #scales non-dim velocity    
    parser.add_argument("--lamb", type=int, default=1) #scales spacial resolution
    parser.add_argument("--T", type=int, default=227) # non-dimensional time
    parser.add_argument("--print_rate", type=int, default=32) # take 1 for dataset creation
    parser.add_argument("--flow", type=str, default="Kolmogorov") 
    parser.add_argument("--model", type=str, default="ClosureRL")
    parser.add_argument("--measure_speedup", type=int, default=False)
    parser.add_argument("--setup", type=str, default="glob")
   
    return parser.parse_known_args()[0]

model_ids = {
    "loc": "20241122-122518", #20241119-110929 #20241118-093042 #20241115-111342
    "glob": "20241127-053713", #20241127-045152 #20241126-160618 #20241126-123326 #20241123-172146 #20241118-112114,
    "interp": "20241127-043257", #20241126-185113
}

agents = {
    "loc": 128,
    "glob": 1,
    "interp": 16,
}

INIT_DUMP = os.path.expanduser("~/CNN-MARL_closure_model_discovery/")

#old local 
#DUMP_PATH = "dump/Kolmogorov22_ppo_cgs1_fgs16/"
#ID = "20241021-110311"

#old global
#DUMP_PATH = "dump/Kolmogorov22_global_ppo_cgs1_fgs16/"
#ID = "20241030-115319"

if __name__ == "__main__":

    test_args = get_args()
    num_agents = 1
    
    N = test_args.lamb * 128

    #load environment
    test_env = KolmogorovEnvironment2(
         step_factor=1,
         Re=test_args.Re,
         max_episode_steps=100000,
         cgs_lamb=test_args.lamb,
         seeds=np.array([test_args.seed]),
         N_agents=num_agents,
         flow=test_args.flow)

   

    # just plays one episode
    reward = 0
    step = 0
    obs ,inf = test_env.reset()
    trivial_act = np.zeros(test_env.action_space.shape)
    states = []
    episode_is_over = False
    m = 2025*test_args.lamb  #20025*test_args.lamb 
    io_rate = 32*test_args.lamb
    for step in tqdm(range(m)):
        obs, rew, terminated, truncated, inf = test_env.step(trivial_act)
        states.append(obs)
        reward += rew
        if step%io_rate==0:
            #print("**********************************************************")
            #print(action.mean(), action.min(), action.max())
            #print(act.mean(), act.min(), act.max())
            #print("**********************************************************")
           #save velocity field as npy
            fname = "klmgrv"
            fname = "velocity_" + fname
            fname = fname + "_" + f"s{test_args.seed}"
            fname = fname + "_" + str(step).zfill(6)
            u1 = test_env.u1
            u1 = downsample_field(test_env.u1, test_args.lamb)
            #np.save(fname, u1)

        if terminated or truncated:
            if terminated:
                 print("terminated")
            else:
                print("truncated")
            episode_is_over = True
            break

    print(f"#steps = {step}, Total Reward = {reward.mean()}")
    print(f"states: mean {np.array(states).mean(axis=(0,1,2))}, std {np.array(states).std(axis=(0,1,2))}")
    test_env.close()