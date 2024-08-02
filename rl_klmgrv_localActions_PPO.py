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
from tianshou.policy import BasePolicy, PPOPolicy, PGPolicy, A2CPolicy
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
from lib.models import *
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
    parser.add_argument("--seed", type=int, default=0)

    #ENVIRONMENT ARGUMENTS 
    parser.add_argument("--step_factor", type=int, default=2)
    parser.add_argument("--cgs_resolution", type=int, default=1)    
    parser.add_argument("--fgs_resolution", type=int, default=1)
    parser.add_argument("--max_interactions", type=int, default=100)
    parser.add_argument("--train_num", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=1)

    #POLICY ARGUMENTS 
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--adam_eps", type=float, default=1e-7)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward_normalization", type=bool, default=True) 
    parser.add_argument("--deterministic_eval", type=bool, default=True)
    parser.add_argument("--action_scaling", type=bool, default=True)
    parser.add_argument("--action_bound_method", type=str, default="tanh")
    parser.add_argument("--ent_coef", type=float, default=-0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--gae_lambda", type=float, default=0.9) 

    #COLLECTOR ARGUMENTS
    parser.add_argument("--buffer_size", type=int, default=20000)

    #LOGGER ARGUMENTS
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--task", type=str, default="local-omega-learning")
    
    #TRAINER ARGUMENTS
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--step_per_epoch", type=int, default=100)
    parser.add_argument("--repeat_per_collect", type=int, default=3)
    parser.add_argument("--episode_per_test", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--step_per_collect", type=int, default=100)
    parser.add_argument("--episode_per_collect", type=int, default=1)
    parser.add_argument("--reward_threshold", type=int, default=100.9)

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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #######################################################################################################
    ####### environments ##################################################################################
    #######################################################################################################
    u0_path = "/home/pfischer/XLB/vel_init/velocity_burn_in_1806594.npy" #4096x4096 simulation
    rho0_path = "/home/pfischer/XLB/vel_init/density_burn_in_1806594.npy" #4096x4096 simulation
    kwargs1, T1,_,_ = get_kwargs(u0_path=u0_path, rho0_path=rho0_path, lamb=args.cgs_resolution) #cgs 
    kwargs2, T2,_,_ = get_kwargs(u0_path=u0_path, rho0_path=rho0_path, lamb=args.fgs_resolution) #fgs
    assert (T2%T1 == 0) # checks if cgs time is a factor of fgs time
    env = create_env(kwargs1, kwargs2, step_factor=args.step_factor,  max_t=args.max_interactions)
    train_env = DummyVectorEnv([lambda: create_env(kwargs1, kwargs2, step_factor=args.step_factor, max_t=args.max_interactions) for _ in range(args.train_num)])
    test_env = DummyVectorEnv([lambda: create_env(kwargs1, kwargs2, step_factor=args.step_factor, max_t=args.max_interactions) for _ in range(args.test_num)])
    #check_env(env)

    #######################################################################################################
    ####### Policy ########################################################################################
    #######################################################################################################
    assert env.observation_space.shape is not None  # for mypy
    assert env.action_space.shape is not None
    #initialize PPO
    actor = MyFCNNActorProb(device=device).to(device)
    critic_backbone = Backbone(device=device).to(device)
    critic = Critic(preprocess_net=critic_backbone, preprocess_net_output_dim=64, device=device).to(device)
    optim = torch.optim.Adam(actor.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    dist = torch.distributions.Normal
    policy = PPOPolicy(actor=actor,
        critic=critic, 
        optim=optim,
        dist_fn=dist, 
        action_space=env.action_space,
        discount_factor=args.gamma,
        reward_normalization=args.reward_normalization, 
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
        ent_coef = args.ent_coef,
        max_grad_norm = args.max_grad_norm,
        gae_lambda=args.gae_lambda, 
    )

    #######################################################################################################
    ####### Collectors ####################################################################################
    #######################################################################################################
    train_collector = Collector(policy=policy, env=train_env, buffer=VectorReplayBuffer(args.buffer_size, len(train_env)))
    test_collector = Collector(policy=policy, env=test_env)
    train_collector.reset()
    test_collector.reset()

    #######################################################################################################
    ####### Logger ########################################################################################
    #######################################################################################################
    log_path = os.path.join(args.logdir, args.task, "ppo")
    logger = WandbLogger2(config=args, train_interval=1, update_interval=1,
                             test_interval=1, info_interval=1)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger.load(writer)

    #######################################################################################################
    ####### Trainer #######################################################################################
    #######################################################################################################
    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.max_epoch,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.episode_per_test,
        batch_size=args.batch_size,
        step_per_collect=args.step_per_collect,
        #episode_per_collect=1,
        show_progress=True,
        logger=logger,
        stop_fn=lambda mean_reward: mean_reward >= args.reward_threshold,
    )
    result = trainer.run()

    #save policy
    #TODO: save policy under a unique name
    torch.save(policy.state_dict(), "dump/GlobOmegLocAct_67.pth")

 