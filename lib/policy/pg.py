from typing import Any, Dict, List, Optional, Type, Union, Tuple

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_torch, to_torch_as, to_numpy
from tianshou.policy import BasePolicy
from tianshou.utils import RunningMeanStd


class IndpPGPolicy(BasePolicy):
    """Implementation of REINFORCE algorithm.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        discount_factor: float = 0.99,
        reward_normalization: bool = False,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        deterministic_eval: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs
        )
        self.actor = model
        self.optim = optim
        self.dist_fn = dist_fn
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._rew_norm = reward_normalization
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8
        self._deterministic_eval = deterministic_eval

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        r"""Compute the discounted returns for each transition.

        .. math::
            G_t = \sum_{i=t}^T \gamma^{i-t}r_i

        where :math:`T` is the terminal time step, :math:`\gamma` is the
        discount factor, :math:`\gamma \in [0, 1]`.
        """

        batch.rew = torch.from_numpy(batch.rew).numpy()

        unnormalized_returns, _ = self.compute_episodic_return(
            batch, buffer, indices, gamma=self._gamma, gae_lambda=1.0
        )


        if self._rew_norm:
            batch.returns = (unnormalized_returns - self.ret_rms.mean) / \
                np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        logits, hidden = self.actor(batch.obs, state=state, info=batch.info)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = logits.argmax(-1)
            elif self.action_type == "continuous":
                act = logits[0]
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)


    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses = []
        entropy = [] #entropy just used for information, not used in update
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                self.optim.zero_grad()
                result = self(minibatch)
                dist = result.dist
                act = to_torch_as(minibatch.act, result.act)
                ret = to_torch(minibatch.returns, torch.float, result.act.device)
                log_prob = dist.log_prob(act)
                loss = -(log_prob * ret).mean()
                loss.backward()
                self.optim.step()
                losses.append(loss.item())
                entropy.append(dist.entropy().mean().item())

        return {"loss": losses,
                "entropy": entropy,
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