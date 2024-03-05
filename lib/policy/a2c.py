from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from tianshou.policy import A2CPolicy, BasePolicy


class MarlA2CPolicy(A2CPolicy):
    """
    Currently only works for continuous action space
    Notes to if I want to extend to discrete action space:
        - change actor loss
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute_returns(  # from A2CPolicy
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """
            Adaption of A2CPolicy._compute_returns method from to the MARL case:
                - returns are computed for each agent and will therefore be returned in shape (batch_size, img_size, img_size)

            Changes from original A2CPolicy:
                1. squeeze action dimension explicitly instead of flattening everything apart from batch size
                2. do not flatten v_s and v_s_ - store them in shape (batch_size, img_size, img_size)
        """
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
                # CNN_MARL contributors: squeeze action dimension - we only need one value per pixel
                v_s.append(self.critic(minibatch.obs).squeeze(1))
                v_s_.append(self.critic(minibatch.obs_next).squeeze(1))
        # Do no flattening here
        batch.v_s = torch.cat(v_s, dim=0)
        v_s = batch.v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).cpu().numpy()
        batch.rew = torch.from_numpy(batch.rew).numpy()
        assert v_s.shape == v_s_.shape == batch.rew.shape  # This is important for further processing
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Emperical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        if self._rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
            gamma=self._gamma,
            gae_lambda=self._lambda
        )
        if self._rew_norm:
            batch.returns = unnormalized_returns / np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return batch

    def learn(  # from A2CPolicy
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        """
        Adaption of A2CPolicy.learn method from to the MARL case:
            - log_probs are not reshaped and transposed
            - loss computation is adapted to the MARL case (with mean over batch_size and all the actors)
        """
        losses, actor_losses, vf_losses, ent_losses = [], [], [], []
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                # calculate loss for actor
                dist = self(minibatch).dist
                log_prob = dist.log_prob(minibatch.act)  # CNN_MARL contributors: returns shape (batch_size, img_size, img_size)
                # log_prob * minibatch.adv: is computed elementwise for each agent
                actor_loss = -(log_prob * minibatch.adv).mean()  # CNN_MARL contributors: mean over batch_size and all the actors!

                # calculate loss for critic
                value = self.critic(minibatch.obs).squeeze(1)  # get rid of action dimension
                vf_loss = F.mse_loss(minibatch.returns, value)

                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = actor_loss + self._weight_vf * vf_loss - self._weight_ent * ent_loss
                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                self.optim.step()
                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        return {
            "loss": losses,
            "loss/actor": actor_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }

    @staticmethod
    def compute_episodic_return(  # from BasePolicy
            batch: Batch,
            buffer: ReplayBuffer,
            indices: np.ndarray,
            v_s_: Optional[Union[np.ndarray, torch.Tensor]] = None,
            v_s: Optional[Union[np.ndarray, torch.Tensor]] = None,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
            Adaption of BasePolicy.compute_episodic_return method from to the MARL case:
                - value mask computation is adapted to the MARL case (with transpose)

            Adapted docstring:
            Use Implementation of Generalized Advantage Estimator to MARL case.
            to calculate q/advantage value of given batch.

            :param Batch batch: a data batch which contains several episodes of data in
                sequential order. Mind that the end of each finished episode of batch
                should be marked by done flag, unfinished (or collecting) episodes will be
                recognized by buffer.unfinished_index().
            :param numpy.ndarray indices: tell batch's location in buffer, batch is equal
                to buffer[indices].
            :param np.ndarray v_s_: the value function of all next states for each agent :math:`V(s')`. (batch_size, img_size, img_size)
            :param float gamma: the discount factor, should be in [0, 1]. Default to 0.99.
            :param float gae_lambda: the parameter for Generalized Advantage Estimation,
                should be in [0, 1]. Default to 0.95.

            :return: two numpy arrays (returns, advantage) with each shape (bsz, img_size, img_size).
        """
        rew = batch.rew
        if v_s_ is None:
            assert np.isclose(gae_lambda, 1.0)
            v_s_ = np.zeros_like(rew)
        else:
            v_s_ = to_numpy(v_s_)  # There was flatten here, not sure why (should be flattened already)
            v_s_ = (v_s_.T * BasePolicy.value_mask(buffer, indices)).T
        v_s = np.roll(v_s_, 1) if v_s is None else to_numpy(v_s)

        end_flag = np.logical_or(batch.terminated, batch.truncated)
        end_flag[np.isin(indices, buffer.unfinished_index())] = True
        # For some reason jit compilation fails for MARL
        advantage = _gae_return(v_s, v_s_, rew, end_flag, gamma, gae_lambda)
        returns = advantage + v_s
        # normalization varies from each policy, so we don't do it here
        return returns, advantage


def _gae_return(
        v_s: np.ndarray,
        v_s_: np.ndarray,
        rew: np.ndarray,
        end_flag: np.ndarray,
        gamma: float,
        gae_lambda: float,
) -> np.ndarray:
    returns = np.zeros(rew.shape)
    delta = rew + v_s_ * gamma - v_s
    discount = (1.0 - end_flag) * (gamma * gae_lambda)
    gae = 0.0
    # CNN_MARL contributors: iterates over batch size
    for i in range(len(rew) - 1, -1, -1):
        gae = delta[i] + discount[i] * gae
        returns[i] = gae
    return returns


if __name__ == "__main__":
    pass
