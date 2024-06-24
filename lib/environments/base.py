import gymnasium as gym
import wandb
import numpy as np
from abc import ABC, abstractmethod

from lib.models import MarlModel
from tqdm import tqdm
from tianshou.data import Batch


class BaseEnvironment(ABC, gym.Env):
    def __init__(self):
        super().__init__()
        self.ep_len = None

    @abstractmethod
    def reset(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, action):
        pass

    def _get_info(self):
        return {}


    def tests(self,
              actor: MarlModel,
              step: int,
              num_eps: int = 10) -> dict:
        # Play multiple episodes and log mean metrics to wandb
        _eval_batches = []
        with tqdm(total=num_eps, desc="Episodes", unit="ep") as pbar:
            for batch_idx in range(num_eps):
                _eval_batch = self._play_and_get_eval_metrics(actor, self.ep_len)
                _eval_batches.append(_eval_batch)
                pbar.update(1)
        eval_log_data = Batch.stack(_eval_batches)
        eval_results = {}
        eval_stds = {}
        for key in eval_log_data.keys():
            eval_stds[f"{key}_std"] = float(eval_log_data[key].std())
            eval_results[key] = float(eval_log_data[key].mean())
        if step is not None:
            eval_results['step'] = step
        wandb.log(eval_results)  # only log means to wandb
        eval_log_data = {**eval_results, **eval_stds}
        return eval_log_data
