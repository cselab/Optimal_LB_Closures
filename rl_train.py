import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import json
from functools import partial

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils import WandbLogger

from lib.environments import get_environment
from lib.policy import get_rl_algo
from lib.distributions import ElementwiseNormal
from lib.models import get_actor_critic
from lib.utils import str2bool, Config, dict_to_wandb_table, restrict_to_num_threads
from lib.trainer import MyOnpolicyTrainer
import wandb


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


def parse_args():
    # training parameters
    parser = argparse.ArgumentParser(description='RL training')

    # environment parameters
    parser.add_argument('--env',
                        help='which environment to use',
                        default='advection',  # advection, burgers
                        type=str)
    parser.add_argument('--img_size',
                        default=64,
                        type=int)
    parser.add_argument('--dataset',
                        help='which dataset to use',
                        default='mnist',  # mnist, fashion
                        type=str)
    parser.add_argument('--subsample',
                        help='subsampling factor - accurate simulation runs with grid of width img_size * subsample',
                        default=4,
                        type=int)
    parser.add_argument('--vel_type',
                        help='velocity type',
                        default='train',  # translational, vortex, vortex2, linear, solid_body_rotation, train, train2
                        type=str)

    # rl training parameters
    _default_device = "0" if torch.cuda.is_available() else "cpu"
    parser.add_argument('--gpu',
                        help='device number of GPU',
                        default=_default_device,  # cpu or int of gpu number
                        type=str)
    parser.add_argument('--max_epochs',
                        default=100,
                        type=int)
    parser.add_argument('--discount',
                        help='discount factor',
                        default=0.8,
                        type=float)
    parser.add_argument('--ent_coef',
                        help='entropy coefficient',
                        default=0.1,
                        type=float)
    parser.add_argument('--ep_len',
                        help='length of episode',
                        default=7,
                        type=int)
    parser.add_argument('--test_ep_len',
                        help='length of episode',
                        default=50,
                        type=int)
    parser.add_argument('--lr',
                        help='learning rate',
                        default=1e-5,
                        type=float)
    parser.add_argument('--repeat_per_collect',
                        default=1,
                        type=int)
    parser.add_argument('--batch_size',
                        default=10,
                        type=int)

    # logging
    parser.add_argument('--note',
                        help='what is special about this run',
                        default="",
                        type=str)
    parser.add_argument('--exp_name',
                        help='name of the wandb experiment',
                        default='trivial',
                        type=str)

    args = parser.parse_args()
    return args


class RLConfig(Config):
    def __init__(self, _args):
        self._set_attributes(_args)

        # training
        self.DEVICE = 'cpu' if self.gpu == 'cpu' else f'cuda:{int(self.gpu)}'
        self.SEED = 0
        self.LOG_PATH = "log/rl_train"
        self.NUM_THREADS = 8
        self.LOG_TO_WANDB_EVERY_N_EPOCHS = 5
        self.architecture = "IRCNN"
        self.algo = "ppo"

        # rl training parameters
        self.STEP_PER_EPOCH = 1000
        self.EPISODE_PER_TEST = 10
        self.EPISODE_PER_COLLECT = 1
        self.SAVE_INTERVAL = 50
        self.NUM_ENVS = int(self.batch_size / self.ep_len) + 1

    def model_name(self):
        """
        :return: name for model logging on wandb. Includes all hyperparameters.
        """
        if self.env == "advection":
            return f"{self.dataset}_{self.vel_type}_{self.algo}_{self.architecture}_eplen:{self.ep_len}" \
                   f"_seed:{self.SEED}_subsample:{self.subsample}_discount:{self.discount}"
        elif self.env == "burgers":
            return f"burgers_{self.vel_type}_{self.algo}_{self.architecture}_eplen:{self.ep_len}" \
                   f"_seed:{self.SEED}_subsample:{self.subsample}_discount:{self.discount}" \
                   f"_num_cgs_points:{self.img_size}_ent_coef:{self.ent_coef}"
        else:
            raise NotImplementedError(f"model_name for environment {self.env} not implemented.")


if __name__ == '__main__':
    args = parse_args()
    config = RLConfig(args)  # load args into config class
    print(f"\nSTARTING\nconfig={config.as_dict()}\n")

    # restrict program to use certain amount of threads for shared cluster environments
    restrict_to_num_threads(config.NUM_THREADS)

    # Create folder to store model checkpoints
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    POLICY_DUMP_PATH = PROJECT_ROOT + f'/weights/policy_dump/{config.model_save_id()}'
    os.makedirs(POLICY_DUMP_PATH, exist_ok=True)

    # set seeds for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # environments
    train_env_constructor = partial(get_environment,
                                    env_name=config.env,
                                    img_size=config.img_size,
                                    dataset=config.dataset,
                                    subsample=config.subsample,
                                    ep_len=config.ep_len,
                                    train=True,
                                    velocity_field_type=config.vel_type)

    # Fix test environment parameters to make reward numbers comparable
    test_env = get_environment(env_name=config.env,
                               img_size=config.img_size,
                               dataset="mnist",
                               subsample=config.subsample,
                               ep_len=config.test_ep_len,
                               train=False,
                               velocity_field_type=config.vel_type)

    train_env = DummyVectorEnv([lambda: train_env_constructor() for _ in range(config.NUM_ENVS)])

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

    # trainer
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
