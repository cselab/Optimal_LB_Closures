import gymnasium as gym

from lib.policy.a2c import MarlA2CPolicy
from lib.policy.ppo import MarlPPOPolicy

def get_rl_algo(name: str,
                actor_critic,
                optim,
                dist,
                action_space: gym.Space,
                config):
    _rl_algo_kwargs = {
        'optim': optim,
        'dist_fn': dist,
        'discount_factor': config.discount,
        'action_space': action_space,  # use test_env here because it is not vectorized
        'deterministic_eval': True
    }
    if name == 'a2c':
        return MarlA2CPolicy(**_rl_algo_kwargs,
                             actor=actor_critic.actor,
                             critic=actor_critic.critic,
                             ent_coef=config.ent_coef)
    elif name == 'ppo':
        return MarlPPOPolicy(**_rl_algo_kwargs,
                             actor=actor_critic.actor,
                             critic=actor_critic.critic,
                             ent_coef=config.ent_coef)
    else:
        raise ValueError(f'Unknown RL algorithm: {name}')
