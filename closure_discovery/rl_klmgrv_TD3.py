import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import os
import sys
from time import strftime
 
from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.trainer import OnpolicyTrainer, OffpolicyTrainer
from tianshou.policy import PPOPolicy, TD3Policy
from tianshou.env import DummyVectorEnv
from lib.environments import *
from lib.utils import save_batch_to_file, model_name
from lib.models import *
from lib.wandb_logger.cutom_logger import WandbLogger
import wandb
wandb.require("core")
device = "cuda" if torch.cuda.is_available() else "cpu"
from tianshou.exploration import GaussianNoise


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", type=str, default="ppo")
    parser.add_argument("--environment", type=str, default="Kolmogorov")
    parser.add_argument("--setup", type=str, default="loc") #options = "loc", "glob", "interp"

    parser.add_argument("--seed", type=int, default=42)

    #ENVIRONMENT ARGUMENTS 
    parser.add_argument("--step_factor", type=int, default=8)
    parser.add_argument("--cgs_resolution", type=int, default=1)    
    parser.add_argument("--fgs_resolution", type=int, default=16)
    parser.add_argument("--max_interactions", type=int, default=10000) #1588 - 1
    parser.add_argument("--train_num", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=1)
    parser.add_argument("--num_agents", type=int, default=128)

    #POLICY ARGUMENTS 
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_eps", type=float, default=1e-7)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward_normalization", type=int, default=True)
    parser.add_argument("--advantage_normalization", type=int, default=True) 
    parser.add_argument("--recompute_advantage", type=int, default=False)
    parser.add_argument("--deterministic_eval", type=int, default=True)
    parser.add_argument("--value_clip", type=int, default=True)
    parser.add_argument("--action_scaling", type=int, default=True)
    parser.add_argument("--action_bound_method", type=str, default="clip")
    parser.add_argument("--ent_coef", type=float, default=0)
    parser.add_argument("--vf_coef", type=float, default=0.25)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--lr_decay", type=int, default=False)

    #COLLECTOR ARGUMENTS
    parser.add_argument("--buffer_size", type=int, default=2000)

    #LOGGER ARGUMENTS
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--task", type=str, default="ClosureRL")
    
    #TRAINER ARGUMENTS
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--step_per_epoch", type=int, default=1500)
    parser.add_argument("--repeat_per_collect", type=int, default=3)
    parser.add_argument("--episode_per_test", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--step_per_collect", type=int, default=128)
    #parser.add_argument("--episode_per_collect", type=int, default=1)
    #parser.add_argument("--reward_threshold", type=int, default=100.)

    #ADDITIONAL INFO
    parser.add_argument("--info", type=str, default="")

    #TD3 arguments
    parser.add_argument("--estimation_step", type=int, default=4)
    parser.add_argument("--max_action", type=float, default=0.005)

    return parser.parse_known_args()[0]


if __name__ == '__main__':
    #######################################################################################################
    ####### setup stuff *##################################################################################
    #######################################################################################################
    args = get_args()

    #seed RNGs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    #generate storage directory
    dump_dir = model_name(args)
    # Generate a unique ID based on the current timestamp
    unique_id = strftime("%Y%m%d-%H%M%S")

    # Save config file
    config_fname = f"{dump_dir}/config_{unique_id}.pkl"
    save_batch_to_file(args, config_fname)
    
    #######################################################################################################
    ####### environments ##################################################################################
    #######################################################################################################
    #seeds = np.array([102, 348, 270, 106, 71, 188, 20, 121, 214, 330, 87, 372,
    #              99, 359, 151, 130, 149, 308, 257, 343, 413, 293, 385, 191, 276,
    #              160, 313, 21, 252, 235, 344, 42])
    train_seeds = [102]
    val_seeds = [102]
    
    train_env = KolmogorovEnvironment2(step_factor=args.step_factor,
                                      max_episode_steps=args.max_interactions,
                                      seeds=train_seeds,
                                      N_agents=args.num_agents,
                                      flow=args.environment)

    test_env = KolmogorovEnvironment2(seeds=val_seeds,
                                     max_episode_steps=args.max_interactions,
                                     step_factor=args.step_factor,
                                     N_agents=args.num_agents,
                                     flow=args.environment)
    
    train_env.seed(args.seed)
    test_env.seed(args.seed)
    #######################################################################################################
    ####### Policy ########################################################################################
    #######################################################################################################
    assert train_env.observation_space.shape is not None  # for mypy
    assert train_env.action_space.shape is not None
    #initialize networks
    if args.setup == "loc":
        actor = local_actor_net_det(in_channels=5, device=device).to(device)
    elif args.setup == "glob":
        actor = central_actor_net_det(in_channels=5, device=device).to(device)
    elif args.setup == "interp":
        actor = FullyConvNet_interpolating_agents_det(in_channels=5, N=args.num_agents, device=device).to(device)

    critic1 = central_Q_net(in_channels=6, device=device).to(device)
    critic2 = central_Q_net(in_channels=6, device=device).to(device)
    
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.learning_rate)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.learning_rate)
    

    #if args.lr_decay == True:
    #    print("lr is decaying")
    #    # decay learning rate to 0 linearly
    #    max_update_num = np.ceil(args.step_per_epoch / args.step_per_collect) * args.max_epoch
    #    lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)


    policy = TD3Policy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic2_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        action_scaling=args.action_scaling,
        action_space=train_env.action_space,
        estimation_step=args.estimation_step,
    )

    #policy = PPOPolicy(actor=actor,
    #    critic=critic,
    #    optim=optim,
    #    dist_fn=dist, 
    #    action_space=train_env.action_space,
    #    discount_factor=args.gamma,
    #    reward_normalization=args.reward_normalization, 
    #    advantage_normalization = args.advantage_normalization,
    #    value_clip = args.value_clip,
    #    deterministic_eval=args.deterministic_eval,
    #    action_scaling=args.action_scaling,
    #    action_bound_method=args.action_bound_method,
    #    ent_coef = args.ent_coef,
    #    vf_coef = args.vf_coef,
    #    eps_clip=args.clip_range,
    #    max_grad_norm = args.max_grad_norm,
    #    gae_lambda=args.gae_lambda, 
    #    recompute_advantage=args.recompute_advantage,
    #    lr_scheduler=lr_scheduler if args.lr_decay else None,
    #)

    #load trained bolicy to continue training
    #DUMP_PATH = "dump/Kolmogorov11_ppo_cgs1_fgs16/"
    #ID = "20240926-220347"
    #policy.load_state_dict(torch.load(DUMP_PATH+'policy_'+ID+'.pth'))

    #######################################################################################################
    ####### Collectors ####################################################################################
    #######################################################################################################
    train_collector = Collector(policy=policy, env=train_env, buffer=VectorReplayBuffer(args.buffer_size, 1))
    test_collector = Collector(policy=policy, env=test_env)
    #train_collector.reset()
    #test_collector.reset()
    #######################################################################################################
    ####### Logger ########################################################################################
    #######################################################################################################
    log_path = os.path.join(args.logdir, args.task, "ppo")
    project_name = os.getenv("WANDB_PROJECT", "LBM_closure_discovery")
    logger = WandbLogger(config=args, train_interval=100, update_interval=10,
                             test_interval=1, info_interval=1, project=project_name)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(dump_dir, f"best_policy_{unique_id}.pth"))

    #######################################################################################################
    ####### Trainer #######################################################################################
    #######################################################################################################
    def build_sigma_schedule(max_sigma=0.5, min_sigma=0.0, steps_per_epoch=50000, decay_time_steps=10000):
        def custom_sigma_schedule(epoch, env_step):
            decay_per_step = (max_sigma - min_sigma) / decay_time_steps
            step_number = (epoch - 1) * steps_per_epoch + env_step
            current_sigma = max_sigma - step_number * decay_per_step
            if current_sigma < 0.0:
                current_sigma = 0.0
            policy._noise = GaussianNoise(sigma=current_sigma)

        return custom_sigma_schedule
    
    decay_steps = int(args.max_epoch * args.step_per_epoch)
    build_sigma_hyperparameters = {'max_sigma': 0.5, 'min_sigma': 0.0, 'decay_time_steps': decay_steps}
    
    
    trainer = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.max_epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.episode_per_test,
        batch_size=args.batch_size,
        save_best_fn=save_best_fn,
        train_fn=build_sigma_schedule(
            **build_sigma_hyperparameters,
            steps_per_epoch=args.step_per_epoch),
        show_progress=True,
        logger=logger,
    )
    
    #trainer = OnpolicyTrainer(
    #    policy=policy,
    #    train_collector=train_collector,
    #    test_collector=test_collector,
    #    max_epoch=args.max_epoch,
    #    step_per_epoch=args.step_per_epoch,
    #    repeat_per_collect=args.repeat_per_collect,
    #    episode_per_test=args.episode_per_test,
    #    batch_size=args.batch_size,
    #    step_per_collect=args.step_per_collect,
    #    #episode_per_collect=args.episode_per_collect,
    #    show_progress=True,
    #    logger=logger,
    #    save_best_fn=save_best_fn,
    #    #stop_fn=lambda mean_reward: mean_reward >= args.reward_threshold,
    #)
    
    #######################################################################################################
    #######  run training  ################################################################################
    #######################################################################################################
    epoch_results = []
    for _,epoch_stats,_ in trainer:
        epoch_results.append(Batch(epoch_stats))

    #######################################################################################################
    #######  save results  ################################################################################
    #######################################################################################################
    # stack totoal results
    total_results = Batch.stack(epoch_results)

    # Save total results
    total_results_fname = f"{dump_dir}/training_stats_{unique_id}.pkl"
    save_batch_to_file(total_results, total_results_fname)

    # Save policy
    policy_fname = f"{dump_dir}/final_policy_{unique_id}.pth"
    torch.save(policy.state_dict(), policy_fname)

 