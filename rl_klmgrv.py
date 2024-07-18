import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import json
from functools import partial

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils import WandbLogger

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

#temporary solution for xlb imports
sys.path.append(os.path.abspath('/home/pfischer/XLB'))
#from my_flows.kolmogorov_2d import Kolmogorov_flow
from my_flows.helpers import get_kwargs

import wandb
wandb.require("core")


def checkpoint_fn(epoch: int,
                  env_step: int,
                  grdient_step: int,
                  _policy: nn.Module,
                  policy_dump_path: str,
                  test_env,
                  mse_dict):
    test_data = test_env.tests(_policy.actor, step=test_env.step, num_eps=10)
    mse_dict[f"epoch_{epoch}"] = test_data["mae_error"]
    torch.save(_policy.state_dict(), f'{policy_dump_path}/policy_ep{epoch}.pt')
    return f'{policy_dump_path}/policy_ep{epoch}.pt'


#argparser 
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CartPole-v1")
    parser.add_argument("--model", type=str, default="ppo")
    parser.add_argument("--reward_threshold", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--step_per_epoch", type=int, default=50000)
    parser.add_argument("--train_num", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--lr", help='learning rate', type=float, default=0.0003)
    parser.add_argument("--repeat_per_collect", type=int, default=10)
    parser.add_argument("--episode_per_test", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--step_per_collect", type=int, default=2000) 
    parser.add_argument("--architecture", type=int, default=[64, 64])

    return parser.parse_known_args()[0]


def create_env(kwargs1, kwargs2):
    """
    creates the environemnt and applyes wrappers to action and
    observations space and sets time limit.
    """
    env = KolmogorovEnvironment(kwargs1, kwargs2)
    env = TimeLimit(env, max_episode_steps=5000)
    env = RescaleAction(env, min_action=-1., max_action=1.)
    env = TransformObservation(env, lambda obs: (obs/20))
    return env




if __name__ == '__main__':

    #######################################################################################################
    ####### setup stuff *##################################################################################
    #######################################################################################################
    args = get_args()

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #######################################################################################################
    ####### environments ##################################################################################
    #######################################################################################################

    u0_path = "/home/pfischer/XLB/vel_init/velocity_burn_in_1806594.npy" #4096x4096 simulation
    rho0_path = "/home/pfischer/XLB/vel_init/density_burn_in_1806594.npy" #4096x4096 simulation

    kwargs1, _,_,_ = get_kwargs(u0_path=u0_path, rho0_path=rho0_path, lamb=1) #cgs 
    kwargs2, _,_,_ = get_kwargs(u0_path=u0_path, rho0_path=rho0_path, lamb=2) #fgs

    train_env = DummyVectorEnv([lambda: create_env(kwargs1, kwargs2) for _ in range(args.train_num)])
    test_env = DummyVectorEnv([lambda: create_env(kwargs1, kwargs2) for _ in range(args.test_num)])

    #######################################################################################################
    ####### Policy ########################################################################################
    #######################################################################################################
    actor_critic = get_actor_critic(config.architecture,
                                    device=config.DEVICE,
                                    env=config.env,
                                    action_dim=test_env.action_space.shape[0])

    # optimizer
    optim = torch.optim.AdamW(actor_critic.parameters(), lr=config.lr)

    # policy
    dist = ElementwiseNormal
    ElementwiseNormal.marl = True
    policy = get_rl_algo(config.algo,
                         actor_critic,
                         optim,
                         dist,
                         action_space=test_env.action_space,
                         config=config)

    # logger
    logging_config_dict = config.as_dict()
    logging_config_dict['total_params'] = sum(p.numel() for p in actor_critic.parameters())
    logging_config_dict['training_mode'] = 'rl'
    logging_config_dict['rl_algo'] = policy.__class__.__name__
    logger = WandbLogger(project=config.exp_name,
                         name=config.model_name(),
                         config=logging_config_dict,
                         save_interval=config.SAVE_INTERVAL)
    dict_to_wandb_table(logging_config_dict)
    logging_config_dict['wandb_url'] = wandb.run.get_url()
    writer = SummaryWriter(config.LOG_PATH)
    logger.load(writer)
    wandb.run.notes = config.note
    mse_errors = dict()  # store mse errors of saved models in this dict

    # save config dict in same directory as weights to make them identifiable
    with open(f"{POLICY_DUMP_PATH}/config.json", "w") as json_file:
        json.dump(logging_config_dict, json_file, indent=4)

    #######################################################################################################
    ####### Collector #####################################################################################
    #######################################################################################################
    trainer_kwargs = {
        'log_to_wandb_every_n_epochs': config.LOG_TO_WANDB_EVERY_N_EPOCHS,
        'policy': policy,
        'max_epoch': config.max_epochs,
        'step_per_epoch': 5 if config.DEVICE == "cpu" else config.STEP_PER_EPOCH,  # to allow local debugging
        'repeat_per_collect': config.repeat_per_collect,
        'episode_per_test': config.EPISODE_PER_TEST,
        'episode_per_collect': config.EPISODE_PER_COLLECT,
        'test_batch_size': config.batch_size,
        'batch_size': config.batch_size,
        'logger': logger,
    }
    train_collector = Collector(policy, train_env, VectorReplayBuffer(20000, len(train_env)))
    test_collector = Collector(policy, test_env)

    #######################################################################################################
    ####### Trainer #######################################################################################
    #######################################################################################################
    trainer = MyOnpolicyTrainer(
        train_collector=train_collector,
        test_collector=test_collector,
        save_checkpoint_fn=partial(checkpoint_fn, _policy=policy, policy_dump_path=POLICY_DUMP_PATH, test_env=test_env, mse_dict=mse_errors),
        **trainer_kwargs,
    )
    trainer.run()

    torch.save(policy.state_dict(), f'{POLICY_DUMP_PATH}/policy.pt')
    torch.save(policy.actor.state_dict(), f'{POLICY_DUMP_PATH}/actor.pt')
    # save config dict in same directory as weights to make them identifiable
    with open(f"{POLICY_DUMP_PATH}/mse_errors.json", "w") as json_file:
        json.dump(mse_errors, json_file, indent=4)
