import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

import os
from time import strftime
from gymnasium import wrappers
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
import gymnasium as gym
 
from tianshou.data import Batch, Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer, ReplayBuffer
from tianshou.trainer import OnpolicyTrainer, offpolicy_trainer, onpolicy_trainer
from tianshou.policy import PPOPolicy, DDPGPolicy, SACPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic, ActorProb
from tianshou.exploration import GaussianNoise, OUNoise

from lib.environments import *
from lib.policy import MarlPPOPolicy, IndpPGPolicy
from lib.utils import save_batch_to_file, model_name
from lib.models import *
from lib.custom_tianshou.my_logger import WandbLogger2
import wandb
wandb.require("core")
device = "cuda" if torch.cuda.is_available() else "cpu"



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", type=str, default="PPO")
    parser.add_argument("--environment", type=str, default="walker")

    parser.add_argument("--seed", type=int, default=0)

    #ENVIRONMENT ARGUMENTS 
    parser.add_argument("--step_factor", type=int, default=1)
    parser.add_argument("--cgs_resolution", type=int, default=1)    
    parser.add_argument("--fgs_resolution", type=int, default=16)
    parser.add_argument("--max_interactions", type=int, default=1600) #1588 - 1
    parser.add_argument("--train_num", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=1)
    parser.add_argument("--env_id", type=str, default="BipedalWalker-v3")

    #POLICY ARGUMENTS 
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--adam_eps", type=float, default=1e-7)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward_normalization", type=bool, default=True)
    parser.add_argument("--advantage_normalization", type=bool, default=False) 
    parser.add_argument("--recompute_advantage", type=bool, default=False)
    parser.add_argument("--deterministic_eval", type=bool, default=True)
    parser.add_argument("--value_clip", type=bool, default=True)
    parser.add_argument("--action_scaling", type=bool, default=True)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--action_bound_method", type=str, default="clip")
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.25)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    #COLLECTOR ARGUMENTS
    parser.add_argument("--buffer_size", type=int, default=20000)

    #LOGGER ARGUMENTS
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--task", type=str, default="local-omega-learning")
    
    #TRAINER ARGUMENTS
    parser.add_argument("--max_epoch", type=int, default=6)
    parser.add_argument("--step_per_epoch", type=int, default=5e6) #1056
    parser.add_argument("--repeat_per_collect", type=int, default=3)
    parser.add_argument("--episode_per_test", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--step_per_collect", type=int, default=512)
    #parser.add_argument("--episode_per_collect", type=int, default=1)
    parser.add_argument("--reward_threshold", type=int, default=100.)

    return parser.parse_known_args()[0]



if __name__ == '__main__':

    #######################################################################################################
    ####### setup stuff *##################################################################################
    #######################################################################################################
    args = get_args()
    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dump_dir = model_name(args)
    

    #######################################################################################################
    ####### environments ##################################################################################
    #######################################################################################################
    #train_env = DummyVectorEnv([lambda: gym.make("BipedalWalker-v3", hardcore=False) for _ in range(args.train_num)])
    #test_env = DummyVectorEnv([lambda: gym.make("BipedalWalker-v3", hardcore=False)for _ in range(args.test_num)])
    env = gym.make(args.env_id)

    train_envs = DummyVectorEnv([lambda: gym.make(args.env_id) for _ in range(args.train_num)])
    test_envs = DummyVectorEnv([lambda: gym.make(args.env_id)for _ in range(args.test_num)])
    #train_envs.seed(args.seed)
    #test_envs.seed(args.seed)
    #######################################################################################################
    ####### Policy ########################################################################################
    #######################################################################################################

   # Neural networks and policy
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    model_hyperparameters = {'hidden_sizes': [256, 256], 'learning_rate': 1e-3}

    # Actor
    net_a = Net(state_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], activation=nn.Tanh, device=device)
    actor = ActorProb(net_a, action_shape, max_action=max_action, device=device, unbounded=True).to(device)

    # Critics
    net_c1 = Net(state_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], activation=nn.Tanh,
                 device=device)
    critic1 = Critic(net_c1, device=device).to(device)

    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in list(actor.modules()) + list(critic1.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)
    
    actor_critic = ActorCritic(actor=actor, critic=critic1)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=model_hyperparameters['learning_rate'])
    dist = torch.distributions.Normal

    lr_scheduler = None
    lr_decay=True
    if lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            args.step_per_epoch / args.step_per_collect
        ) * args.max_epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )

    policy = PPOPolicy(actor=actor, critic=critic1, optim=optim, dist_fn=dist,
                       discount_factor=args.gamma,
                       gae_lambda=args.gae_lambda,
                       max_grad_norm=args.max_grad_norm,
                       vf_coef=args.vf_coef,
                       ent_coef=args.ent_coef,
                       reward_normalization=args.reward_normalization,
                       action_scaling=args.action_scaling,
                       action_bound_method=args.action_bound_method,
                       lr_scheduler=lr_scheduler,
                       action_space=env.action_space,
                       eps_clip=args.clip_range,
                       value_clip=args.value_clip,
                       dual_clip=args.dual_clip,
                       advantage_normalization=args.advantage_normalization,
                       recompute_advantage=args.recompute_advantage,
                       )


    #######################################################################################################
    ####### Collectors ####################################################################################
    #######################################################################################################
    # Collectors
    use_prioritised_replay_buffer = False
    prioritized_buffer_hyperparameters = {'total_size': 20000, 'buffer_num': 1, 'alpha': 0.4, 'beta': 0.5}
    if use_prioritised_replay_buffer:
        train_collector = Collector(policy, train_envs,
                                            PrioritizedVectorReplayBuffer(**prioritized_buffer_hyperparameters),
                                            exploration_noise=True)
    else:
        train_collector = Collector(policy, train_envs,
                                            ReplayBuffer(size=prioritized_buffer_hyperparameters['total_size']),
                                            exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    #######################################################################################################
    ####### Logger ########################################################################################
    #######################################################################################################
    log_path = os.path.join(args.logdir, args.task, args.algorithm)
    project_name = os.getenv("WANDB_PROJECT", "walker")
    logger = WandbLogger2(config=args, train_interval=1000, update_interval=10,
                             test_interval=1, info_interval=1, project=project_name)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger.load(writer)

    #######################################################################################################
    ####### Trainer #######################################################################################
    #######################################################################################################
    
    # Training
    trainer_hyperparameters = {'max_epoch': 30, 'step_per_epoch': 200_000, 'step_per_collect': 512,
                               'episode_per_test': 10,
                               'batch_size': 64}
    all_hypeparameters = dict(model_hyperparameters, **trainer_hyperparameters, **prioritized_buffer_hyperparameters)
    all_hypeparameters['seed'] = args.seed
    all_hypeparameters['use_prioritised_replay_buffer'] = use_prioritised_replay_buffer

    result = onpolicy_trainer(policy, train_collector, test_collector, **trainer_hyperparameters,
                                stop_fn=None,
                                repeat_per_collect=args.repeat_per_collect,
                                logger=logger)
    

    print(f'Finished training! Use {result["duration"]}')

    
    #######################################################################################################
    #######  run training  ################################################################################
    #######################################################################################################
    #epoch_results = []
    #for _,epoch_stats,_ in trainer:
    #    epoch_results.append(Batch(epoch_stats))
    #trainer.run()
 
    # stack totoal results
    #total_results = Batch.stack(epoch_results)

    # Generate a unique ID based on the current timestamp
    unique_id = strftime("%Y%m%d-%H%M%S")

    # Save total results
    #total_results_fname = f"{dump_dir}/training_stats_{unique_id}.pkl"
    #save_batch_to_file(total_results, total_results_fname)

    # Save config file
    #config_fname = f"{dump_dir}/config_{unique_id}.pkl"
    #save_batch_to_file(args, config_fname)

    # Save policy
    policy_fname = f"{dump_dir}/policy_{unique_id}.pth"
    torch.save(policy.state_dict(), policy_fname)

 