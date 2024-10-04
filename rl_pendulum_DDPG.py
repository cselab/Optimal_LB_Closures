import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from time import strftime
from gymnasium import wrappers
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
import gymnasium as gym
 
from tianshou.data import Batch, Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.trainer import OnpolicyTrainer, offpolicy_trainer
from tianshou.policy import PPOPolicy, DDPGPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
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

    parser.add_argument("--algorithm", type=str, default="ppo")
    parser.add_argument("--environment", type=str, default="walker")

    parser.add_argument("--seed", type=int, default=0)

    #ENVIRONMENT ARGUMENTS 
    parser.add_argument("--step_factor", type=int, default=1)
    parser.add_argument("--cgs_resolution", type=int, default=1)    
    parser.add_argument("--fgs_resolution", type=int, default=16)
    parser.add_argument("--max_interactions", type=int, default=1600) #1588 - 1
    parser.add_argument("--train_num", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=1)
    parser.add_argument("--env_id", type=str, default="Pendulum-v1")

    #POLICY ARGUMENTS 
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--adam_eps", type=float, default=1e-7)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--reward_normalization", type=bool, default=True)
    parser.add_argument("--advantage_normalization", type=bool, default=False) 
    parser.add_argument("--recompute_advantage", type=bool, default=False)
    parser.add_argument("--deterministic_eval", type=bool, default=True)
    parser.add_argument("--value_clip", type=bool, default=True)
    parser.add_argument("--action_scaling", type=bool, default=True)
    parser.add_argument("--action_bound_method", type=str, default="tanh")
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--clip_range", type=float, default=0.18)
    parser.add_argument("--max_grad_norm", type=float, default=1.)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    #COLLECTOR ARGUMENTS
    parser.add_argument("--buffer_size", type=int, default=20000)

    #LOGGER ARGUMENTS
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--task", type=str, default="local-omega-learning")
    
    #TRAINER ARGUMENTS
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--step_per_epoch", type=int, default=5e6) #1056
    parser.add_argument("--repeat_per_collect", type=int, default=1)
    parser.add_argument("--episode_per_test", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--step_per_collect", type=int, default=2048)
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
    #######################################################################################################
    ####### Policy ########################################################################################
    #######################################################################################################

    # Neural networks and policy
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    model_hyperparameters = {'hidden_sizes': [128, 128], 'learning_rate': 1e-3, 'estimation_step': 1}

    net_a = Net(state_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], device=device)
    actor = Actor(net_a, action_shape, max_action=max_action, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=model_hyperparameters['learning_rate'])
    net_c = Net(state_shape, action_shape, hidden_sizes=model_hyperparameters['hidden_sizes'], concat=True,
                device=device)
    critic = Critic(net_c, device=device).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=model_hyperparameters['learning_rate'])
    policy = DDPGPolicy(actor, actor_optim, critic, critic_optim,
                        exploration_noise=GaussianNoise(sigma=0.5 * max_action),
                        estimation_step=model_hyperparameters['estimation_step'],
                        action_space=env.action_space)


    #######################################################################################################
    ####### Collectors ####################################################################################
    #######################################################################################################
        # Collectors
    prioritized_buffer_hyperparameters = {'total_size': 20_000, 'buffer_num': 1, 'alpha': 0.7, 'beta': 0.5}
    train_collector = Collector(policy, train_envs,
                                        PrioritizedVectorReplayBuffer(**prioritized_buffer_hyperparameters,
                                                                              device=device),
                                        exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    #######################################################################################################
    ####### Logger ########################################################################################
    #######################################################################################################
    log_path = os.path.join(args.logdir, args.task, "ppo")
    project_name = os.getenv("WANDB_PROJECT", "walker")
    logger = WandbLogger2(config=args, train_interval=1000, update_interval=10,
                             test_interval=1, info_interval=1, project=project_name)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger.load(writer)

    #######################################################################################################
    ####### Trainer #######################################################################################
    #######################################################################################################
    # Sigma schedule
    def build_sigma_schedule(max_sigma=0.5, min_sigma=0.0, steps_per_epoch=50000, decay_time_steps=10000):
        def custom_sigma_schedule(epoch, env_step):
            decay_per_step = (max_sigma - min_sigma) / decay_time_steps
            step_number = (epoch - 1) * steps_per_epoch + env_step

            current_sigma = max_sigma - step_number * decay_per_step
            if current_sigma < 0.0:
                current_sigma = 0.0
            policy._noise = GaussianNoise(sigma=current_sigma * max_action)

        return custom_sigma_schedule


    
    # Training
    trainer_hyperparameters = {'max_epoch': 2, 'step_per_epoch': 75_000, 'step_per_collect': 5,
                               'episode_per_test': 10,
                               'batch_size': 64}
    all_hypeparameters = model_hyperparameters | prioritized_buffer_hyperparameters | trainer_hyperparameters
    all_hypeparameters['seed'] = args.seed
    decay_steps = int(trainer_hyperparameters['max_epoch'] * trainer_hyperparameters['step_per_epoch'] * 0.2)
    result = offpolicy_trainer(policy, train_collector, test_collector, **trainer_hyperparameters,
                                          train_fn=build_sigma_schedule(max_sigma=0.2, min_sigma=0.0,
                                                                        steps_per_epoch=trainer_hyperparameters[
                                                                            'step_per_epoch'],
                                                                        decay_time_steps=decay_steps),
                                          stop_fn=None, logger=logger)
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

 