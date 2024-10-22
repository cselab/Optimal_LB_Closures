import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from time import strftime
 
from tianshou.data import Batch, Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer, ReplayBuffer
from tianshou.trainer import OnpolicyTrainer, OffpolicyTrainer
from tianshou.policy import PPOPolicy, DDPGPolicy, SACPolicy, TD3Policy
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

    parser.add_argument("--algorithm", type=str, default="td3")
    parser.add_argument("--environment", type=str, default="Kolmogorov22")

    parser.add_argument("--seed", type=int, default=0)

    #ENVIRONMENT ARGUMENTS 
    parser.add_argument("--step_factor", type=int, default=8)
    parser.add_argument("--cgs_resolution", type=int, default=1)    
    parser.add_argument("--fgs_resolution", type=int, default=16)
    parser.add_argument("--max_interactions", type=int, default=10000) #1588 - 1
    parser.add_argument("--train_num", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=1)

    #POLICY ARGUMENTS 
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_eps", type=float, default=1e-7)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward_normalization", type=bool, default=False)
    parser.add_argument("--advantage_normalization", type=bool, default=False) 
    parser.add_argument("--recompute_advantage", type=bool, default=False)
    parser.add_argument("--deterministic_eval", type=bool, default=True)
    parser.add_argument("--value_clip", type=bool, default=True)
    parser.add_argument("--action_scaling", type=bool, default=True)
    parser.add_argument("--action_bound_method", type=str, default="clip")
    parser.add_argument("--ent_coef", type=float, default=0.) #1e-4
    parser.add_argument("--vf_coef", type=float, default=0.25)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    #COLLECTOR ARGUMENTS
    parser.add_argument("--buffer_size", type=int, default=2000)

    #LOGGER ARGUMENTS
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--task", type=str, default="local-omega-learning")
    
    #TRAINER ARGUMENTS
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--step_per_epoch", type=int, default=1500) #1056
    parser.add_argument("--repeat_per_collect", type=int, default=1)
    parser.add_argument("--episode_per_test", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--step_per_collect", type=int, default=32)
    #parser.add_argument("--episode_per_collect", type=int, default=1)
    #parser.add_argument("--reward_threshold", type=int, default=100.)

    return parser.parse_known_args()[0]



if __name__ == '__main__':
    # Generate a unique ID based on the current timestamp
    unique_id = strftime("%Y%m%d-%H%M%S")

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
    train_seeds = [102]
    val_seeds = [102]
    #test_seeds = np.array([69, 33, 420])
    
    train_env = KolmogorovEnvironment22(seeds=train_seeds, max_episode_steps=args.max_interactions, step_factor=args.step_factor)
    test_env = KolmogorovEnvironment22(seeds=val_seeds, max_episode_steps=args.max_interactions, step_factor=args.step_factor)
    #train_env = TransformObservation(train_env, lambda obs: (obs/0.00014))
    #test_env = env = TransformObservation(test_env, lambda obs: (obs/0.00014))
    #######################################################################################################
    ####### Policy ########################################################################################
    #######################################################################################################
    #initialize PPO
    #actor = local_actor_net2(device=device).to(device)
    #critic = local_critic_net2(device=device).to(device)
    #actor = MyFCNNActorProb2(in_channels=9, device=device).to(device)
    #critic = MyFCNNCriticProb2(in_channels=9, device=device).to(device)

    max_action = 0.005

    model_hyperparameters = {'estimation_step': 4}

    actor = central_actor_net(in_channels=6, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)

    critic1 = central_critic_net1(in_channels=7, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.learning_rate)
    
    critic2 = central_critic_net1(in_channels=7, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.learning_rate)


    policy = TD3Policy(actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
                       exploration_noise=GaussianNoise(sigma= 2 * max_action),
                       estimation_step=model_hyperparameters['estimation_step'],
                       action_space=test_env.action_space)

    #load trained bolicy to continue training
    #DUMP_PATH = "dump/Kolmogorov11_ppo_cgs1_fgs16/"
    #ID = "20240919-035155"
    #policy.load_state_dict(torch.load(DUMP_PATH+'policy_'+ID+'.pth'))

    #######################################################################################################
    ####### Collectors ####################################################################################
    #######################################################################################################
    # Collectors
    train_collector = Collector(policy=policy, env=train_env, buffer=VectorReplayBuffer(args.buffer_size, 1))
    test_collector = Collector(policy=policy, env=test_env)

    #######################################################################################################
    ####### Logger ########################################################################################
    #######################################################################################################
    log_path = os.path.join(args.logdir, args.task, args.algorithm)
    project_name = os.getenv("WANDB_PROJECT", "f_states")
    logger = WandbLogger2(config=args, train_interval=1, update_interval=1,
                             test_interval=1, info_interval=1, project=project_name)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(dump_dir, f"best_policy{unique_id}.pth"))

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
    
    decay_steps = int(args.max_epoch * args.step_per_epoch * 0.05)
    build_sigma_hyperparameters = {'max_sigma': 0.5, 'min_sigma': 0.0, 'decay_time_steps': decay_steps}
    
    trainer = OffpolicyTrainer(policy, train_collector, test_collector,
                               max_epoch=args.max_epoch,
                               step_per_epoch=args.step_per_epoch,
                               step_per_collect=args.step_per_collect,
                               episode_per_test=args.episode_per_test,
                               batch_size=args.batch_size,
                               train_fn=build_sigma_schedule(**build_sigma_hyperparameters,
                                                             steps_per_epoch=args.step_per_epoch),
                                stop_fn=None,
                                save_best_fn=save_best_fn,
                                logger=logger)
        
    #######################################################################################################
    #######  run training  ################################################################################
    #######################################################################################################
    epoch_results = []
    for _,epoch_stats,_ in trainer:
        epoch_results.append(Batch(epoch_stats))
 
    # stack totoal results
    total_results = Batch.stack(epoch_results)

    # Save total results
    total_results_fname = f"{dump_dir}/training_stats_{unique_id}.pkl"
    save_batch_to_file(total_results, total_results_fname)

    # Save config file
    config_fname = f"{dump_dir}/config_{unique_id}.pkl"
    save_batch_to_file(args, config_fname)

    # Save policy
    policy_fname = f"{dump_dir}/policy_{unique_id}.pth"
    torch.save(policy.state_dict(), policy_fname)

 