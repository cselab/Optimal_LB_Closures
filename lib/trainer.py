import tianshou.env
from tianshou.trainer import OnpolicyTrainer
from typing import Any, Dict, Tuple
from typing import Union


class MyOnpolicyTrainer(OnpolicyTrainer):
    def __init__(self,
                 test_batch_size: int,
                 log_to_wandb_every_n_epochs: int = 1,
                 viz_ep_len: Union[int, None] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_to_wandb_every_n_epochs = log_to_wandb_every_n_epochs
        self.test_batch_size = test_batch_size
        self.viz_ep_len = self.test_collector.env.workers[0].env.ep_len if viz_ep_len is None else viz_ep_len

    def test_step(self) -> Tuple[Dict[str, Any], bool]:
        super().test_step()
        assert isinstance(self.test_collector.env, tianshou.env.DummyVectorEnv)
        assert self.policy.training is False, "Policy should be in eval mode for logging to wandb."
        # Play episodes and log evaluation metrics to wandb
        test_stat = self.test_collector.env.workers[0].env.tests(
            actor=self.policy.actor,
            step=self.env_step,
            num_eps=self.test_batch_size,
        )

        # Playing one extra episode for visualization
        if self.epoch % self.log_to_wandb_every_n_epochs == 0 or self.epoch < 5:
            # NOTE self.test_collector.env is not a IterativeDeconvEnv but a DummyVectorEnv
            self.test_collector.env.workers[0].env.play_episode_and_log_to_wandb(
                actor=self.policy.actor,
                step=self.env_step,
                ep_len=self.viz_ep_len
            )

        stop_fn_flag = False
        if self.stop_fn and self.stop_fn(self.best_reward):
            stop_fn_flag = True
        return test_stat, stop_fn_flag


