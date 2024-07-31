import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import json
from functools import partial
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
 
from tianshou.utils import WandbLogger
from tianshou.data import Batch, Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import BasePolicy, PPOPolicy, PGPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
#from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils.net.continuous import Actor, Critic, ActorProb
from tianshou.trainer.utils import gather_info, test_episode

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit, RescaleAction, TransformObservation
from stable_baselines3.common.env_checker import check_env

from lib.environments import get_environment
from lib.environments.kolmogorov import KolmogorovEnvironment, KolmogorovEnvironment3, KolmogorovEnvironment4
from lib.policy import get_rl_algo
from lib.distributions import ElementwiseNormal
from lib.models import get_actor_critic
from lib.utils import str2bool, Config, dict_to_wandb_table, restrict_to_num_threads
from lib.trainer import MyOnpolicyTrainer
from lib.models import FcNN, MyFCNNActorProb, MyFCNNActorProb2
from lib.custom_tianshou.my_logger import WandbLogger2

#temporary solution for xlb imports
sys.path.append(os.path.abspath('/home/pfischer/XLB'))
#from my_flows.kolmogorov_2d import Kolmogorov_flow
from my_flows.helpers import get_kwargs

#from lib.custom_tianshou.my_actors import MyActorProb

import wandb
wandb.require("core")

device = "cuda" if torch.cuda.is_available() else "cpu"



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CartPole-v1")
    parser.add_argument("--model", type=str, default="ppo")
    parser.add_argument("--reward_threshold", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer_size", type=int, default=20000)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--step_per_epoch", type=int, default=100)
    parser.add_argument("--train_num", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--gamma", type=float, default=0.90)
    parser.add_argument("--lr", help='learning rate', type=float, default=1e-4)
    parser.add_argument("--repeat_per_collect", type=int, default=1)
    parser.add_argument("--episode_per_test", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--step_per_collect", type=int, default=200)

    return parser.parse_known_args()[0]


def create_env(kwargs1, kwargs2, min_a=-1., max_a=1., step_factor=10, max_t=100):
    """
    creates the environemnt and applyes wrappers to action and
    observations space and sets time limit.
    """
    env = KolmogorovEnvironment4(kwargs1, kwargs2, step_factor=step_factor, max_episode_steps=max_t)
    env = TransformObservation(env, lambda obs: (obs/15))
    #env = TimeLimit(env, max_episode_steps=max_t)
    return env


if __name__ == '__main__':

    #######################################################################################################
    ####### setup stuff *##################################################################################
    #######################################################################################################
    args = get_args()
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #restrict_to_num_threads(1)

    #######################################################################################################
    ####### environments ##################################################################################
    #######################################################################################################
    u0_path = "/home/pfischer/XLB/vel_init/velocity_burn_in_1806594.npy" #4096x4096 simulation
    rho0_path = "/home/pfischer/XLB/vel_init/density_burn_in_1806594.npy" #4096x4096 simulation
    kwargs1, T1,_,_ = get_kwargs(u0_path=u0_path, rho0_path=rho0_path, lamb=1) #cgs 
    kwargs2, T2,_,_ = get_kwargs(u0_path=u0_path, rho0_path=rho0_path, lamb=1) #fgs
    step_factor=2
    #check if cgs time is a factor of fgs time
    assert (T2%T1 == 0)
    env = create_env(kwargs1, kwargs2, step_factor=step_factor,  max_t=100)
    train_env = DummyVectorEnv([lambda: create_env(kwargs1, kwargs2, step_factor=step_factor, max_t=100) for _ in range(args.train_num)])
    test_env = DummyVectorEnv([lambda: create_env(kwargs1, kwargs2, step_factor=step_factor, max_t=100) for _ in range(args.test_num)])
    #check_env(env)

    #Policy
    assert env.observation_space.shape is not None  # for mypy
    assert env.action_space.shape is not None

    #Policy
    #actor = MyFCNNActorProb(env.action_space.shape, device=device).to(device)
    actor = MyFCNNActorProb2(env.action_space.shape, device=device).to(device)
    optim = torch.optim.AdamW(actor.parameters(), lr=0.001)
    dist = torch.distributions.Normal
    policy = PGPolicy(model=actor,optim=optim, dist_fn=dist, action_space=env.action_space,
        discount_factor=0.97,reward_normalization=False, deterministic_eval=True,
        observation_space=env.observation_space, action_scaling=True, action_bound_method = "tanh",
    )

    #Collectors
    train_collector = Collector(policy=policy, env=train_env, buffer=VectorReplayBuffer(args.buffer_size, len(train_env)))
    test_collector = Collector(policy=policy, env=test_env)
    train_collector.reset()
    test_collector.reset()

    #wandb Logger
    log_path = os.path.join(args.logdir, args.task, "pg")
    logger = WandbLogger2(config=args)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger.load(writer)

    #Trainer
    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.max_epoch,
        step_per_epoch=100,
        repeat_per_collect=1,
        episode_per_test=1,
        batch_size=100,
        episode_per_collect=3,
        show_progress=True,
        logger=logger,
        stop_fn=lambda mean_reward: mean_reward >= args.reward_threshold,
    )

    result = trainer.run()


    #save policy
    torch.save(policy.state_dict(), "dump/GlobOmegLocAct_4.pth")
    print("run is finished")

 