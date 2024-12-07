#imports
import os
import pickle
import numpy as np
from tqdm import tqdm
import argparse

from lib.models import *
from tianshou.policy import PPOPolicy
from lib.environments.kolmogorov import KolmogorovEnvironment
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
    "loc": "20241205-135557", 
    "glob": "20241206-121705", 
    "interp": "20241206-062003"
}

INIT_DUMP = os.path.expanduser("~/CNN-MARL_closure_model_discovery/")

if __name__ == "__main__":

    test_args = get_args()
    DUMP_PATH = "results/weights/Kolmogorov_"+test_args.setup+"_ppo/"
    ID = model_ids.get(test_args.setup)
    with open(INIT_DUMP+DUMP_PATH+'config_'+ID+'.pkl', 'rb') as f:
        train_args = pickle.load(f)
    
    N = test_args.lamb * 128
    num_agents = train_args.num_agents * test_args.lamb

    #load environment
    test_env = KolmogorovEnvironment(
         step_factor=1,
         Re=test_args.Re,
         max_episode_steps=100000,
         cgs_lamb=test_args.lamb,
         seeds=np.array([test_args.seed]),
         N_agents=num_agents,
         flow=test_args.flow)

    #get policy
    #######################################################################################################
    ####### PPO Policy ########################################################################################
    #######################################################################################################
    if train_args.setup == "loc":
            actor = local_actor_net(in_channels=6, device=device, nx=N).to(device)
            critic = central_critic_net(in_channels=6, device=device).to(device)
    elif train_args.setup == "glob":
        actor = central_actor_net(in_channels=6, device=device, nx=N).to(device)
        critic = central_critic_net(in_channels=6, device=device).to(device)
    elif train_args.setup == "interp":
        actor = FullyConvNet_interpolating_agents(in_channels=6, N=num_agents, device=device, nx=N).to(device)
        critic = central_critic_net(in_channels=6, device=device).to(device)

    actor_critic = ActorCritic(actor=actor, critic=critic)
    optim = torch.optim.AdamW(actor_critic.parameters(), lr=train_args.learning_rate, eps=train_args.adam_eps)
    dist = torch.distributions.Normal

    policy = PPOPolicy(actor=actor,
        critic=critic, 
        optim=optim,
        dist_fn=dist, 
        action_space=test_env.action_space,
        deterministic_eval=True,
        action_scaling=train_args.action_scaling,
        action_bound_method=train_args.action_bound_method,
    )

    #load policy
    policy.load_state_dict(torch.load(INIT_DUMP+DUMP_PATH+'best_policy_'+ID+'.pth'))

    # Define folder paths
    main_folder = f"../results/re{int(test_args.Re)}_T{int(test_args.T)}_S{test_args.seed}_{test_args.flow}_runs"
    run_folder = f"re{int(test_args.Re)}_T{int(test_args.T)}_N{int(N)}_S{test_args.seed}_U{test_args.upsi}_{test_args.model+'-'+train_args.setup}"
    create_and_navigate_to(main_folder)
    create_and_navigate_to(run_folder)
    # remove all .png files in the current directory
    os.system("rm -rf ./*.png")
    os.system("rm -rf ./*.npy")
    os.system("rm -rf ./*.json")



    # just plays one episode
    reward = 0
    step = 0
    policy.eval()
    obs ,inf = test_env.reset()
    acts = []
    states = []
    episode_is_over = False
    m = 20025*test_args.lamb
    io_rate = 32*test_args.lamb
    for step in tqdm(range(m)):
        batch = policy(Batch(obs=np.array([obs]), info=inf))
        action = batch.act[0].detach().cpu().numpy()
        act = policy.map_action(action)
        obs, rew, terminated, truncated, inf = test_env.step(act)
        reward += rew
        if step%io_rate==0:
           #save velocity field as npy
            fname = "klmgrv"
            fname = "velocity_" + fname
            fname = fname + "_" + f"s{test_args.seed}"
            fname = fname + "_" + str(step).zfill(6)
            u1 = test_env.u1
            u1 = downsample_field(test_env.u1, test_args.lamb)
            np.save(fname, u1)
            acts.append(act)
            states.append(obs)

        if terminated or truncated:
            if terminated:
                 print("terminated")
            else:
                print("truncated")
            break

    print(f"#steps = {step}, Total Reward = {reward.mean()}")
    test_env.close()