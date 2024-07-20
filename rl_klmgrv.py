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
#from tianshou.utils import WandbLogger
from tianshou.data import Batch, Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import BasePolicy, PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import Actor, Critic, ActorProb
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit, RescaleAction, TransformObservation
from stable_baselines3.common.env_checker import check_env

from lib.environments import get_environment
from lib.environments.kolmogorov import KolmogorovEnvironment
from lib.policy import get_rl_algo
from lib.distributions import ElementwiseNormal
from lib.models import get_actor_critic
from lib.utils import str2bool, Config, dict_to_wandb_table, restrict_to_num_threads
from lib.trainer import MyOnpolicyTrainer
from lib.my_logger import WandbLogger2

#temporary solution for xlb imports
sys.path.append(os.path.abspath('/home/pfischer/XLB'))
#from my_flows.kolmogorov_2d import Kolmogorov_flow
from my_flows.helpers import get_kwargs

import wandb
wandb.require("core")

device = "cuda" if torch.cuda.is_available() else "cpu"


#def checkpoint_fn(epoch, env_step, gradient_step):
#    #test_data = test_env.tests(_policy.actor, step=test_env.step, num_eps=10)
#    #mse_dict[f"epoch_{epoch}"] = test_data["mae_error"]
#    #torch.save(_policy.state_dict(), f'{policy_dump_path}/policy_ep{epoch}.pt')
#    print("checkpoint fct is called!!!")
#    return ""


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CartPole-v1")
    parser.add_argument("--model", type=str, default="ppo")
    parser.add_argument("--reward_threshold", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer_size", type=int, default=20000)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--step_per_epoch", type=int, default=100)
    parser.add_argument("--train_num", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--lr", help='learning rate', type=float, default=0.0003)
    parser.add_argument("--repeat_per_collect", type=int, default=10)
    parser.add_argument("--episode_per_test", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--step_per_collect", type=int, default=100) 
    parser.add_argument("--architecture", type=int, default=[64, 64])
    parser.add_argument("--backbone_out_dim", type=int, default=64)

    return parser.parse_known_args()[0]


def create_env(kwargs1, kwargs2, max_t=50000, min_a=-1., max_a=1.):
    """
    creates the environemnt and applyes wrappers to action and
    observations space and sets time limit.
    """
    env = KolmogorovEnvironment(kwargs1, kwargs2)
    env = TimeLimit(env, max_episode_steps=max_t)
    env = RescaleAction(env, min_action=min_a, max_action=max_a)
    env = TransformObservation(env, lambda obs: (obs/20))
    return env


class Backbone(nn.Module):

    def __init__(self, out_size=64, device="cpu"):
        super(Backbone, self).__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.Conv2d(2, 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )
        
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            #nn.Dropout(p=0.5),
            nn.Linear(128, out_size),
            nn.ReLU(True),
        )
        
    #def forward(self, x):
    #    x = x.reshape(x.shape[0],1,128,128)
    #    x = self.encoder_cnn(x)
    #    x = x.reshape(x.shape[0], -1)
    #    x = self.encoder_lin(x)
    #    return x

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=device)
        batch = obs.shape[0]

        obs = self.encoder_cnn(obs.reshape(batch, 1, 128, 128))
        obs = obs.reshape(batch, -1)
        logits = self.encoder_lin(obs)

        return logits, state



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
    #check if cgs time is a factor of fgs time
    assert (T2%T1 == 0)
    env = create_env(kwargs1, kwargs2, max_t=T1)
    train_env = DummyVectorEnv([lambda: create_env(kwargs1, kwargs2, max_t=T1) for _ in range(args.train_num)])
    test_env = DummyVectorEnv([lambda: create_env(kwargs1, kwargs2, max_t=T1) for _ in range(args.test_num)])

    #Policy
    assert env.observation_space.shape is not None  # for mypy
    assert env.action_space.shape is not None

    net = Net(state_shape=env.observation_space.shape, hidden_sizes=args.architecture, device=device).to(device)
    #net = Backbone(out_size=args.backbone_out_dim, device=device).to(device)
    #net = Net1(state_shape=env.observation_space.shape, action_shape=(64,)).to(device)
    actor = ActorProb(preprocess_net=net, action_shape=env.action_space.shape, max_action=1,
                     preprocess_net_output_dim=args.backbone_out_dim, device=device).to(device)
    critic = Critic(preprocess_net=net, preprocess_net_output_dim=args.backbone_out_dim, device=device).to(device)
    actor_critic = ActorCritic(actor=actor, critic=critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    dist = torch.distributions.Normal
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        action_space=env.action_space,
        deterministic_eval=True,
        action_scaling=False,
    )

    #Collectors
    train_collector = Collector(
        policy=policy,
        env=train_env,
        buffer=VectorReplayBuffer(args.buffer_size, len(train_env)),
    )
    test_collector = Collector(policy=policy, env=test_env)
    train_collector.reset()
    test_collector.reset()

    #wandb Logger -> doesn't really work
    log_path = os.path.join(args.logdir, args.task, "dqn")
    #logger = WandbLogger2(train_interval=1000, test_interval = 1, update_interval = 1000, save_interval = 1, config=args)
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
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,             
        episode_per_test=args.episode_per_test,
        batch_size=args.batch_size,
        step_per_collect=args.step_per_collect,
        #stop_fn=lambda mean_reward: mean_reward >= args.reward_threshold,
        #save_checkpoint_fn=checkpoint_fn,
        logger=logger,
    )

    result = trainer.run()


    #save policy
    torch.save(policy.state_dict(), "global_omega.pth")
    print("run is finished")

 