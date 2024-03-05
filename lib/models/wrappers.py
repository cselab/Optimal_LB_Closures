import torch.nn as nn
import torch


class MarlModel(nn.Module):
    """
    Wrapper class for MARL networks. (can be used for actor and critic)
    - Makes sure that forward pass is appropriate for tianshou
    - Can pass in any model that maps from state to action
    """
    def __init__(self, backbone: nn.Module, _is: str = "policy", action_dim=1):
        super().__init__()
        self._is = _is
        self.action_dim = action_dim

        if self._is == "critic":
            self.output_size = 1
        elif self._is == "actor":
            self.output_size = 2 * self.action_dim
        else:
            ValueError(f"Invalid _is argument: {_is}. Must be one of 'actor', 'critic'.")

        self.epsilon = 1e-6
        self.backbone = backbone
        self.last_layer = nn.Conv2d(in_channels=self.backbone.feature_dim,
                                    out_channels=self.output_size,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    dilation=1,
                                    bias=True,
                                    padding_mode='circular')

        self.softplus = torch.nn.Softplus()  # activation function for std

    def forward(self, _obs, *args, **kwargs):
        _features = self.backbone(_obs)
        _pred = self.last_layer(_features)

        if self._is == "critic":
            return _pred

        elif self._is == "actor":
            # split the output into mean and std
            mean, std = torch.split(_pred, split_size_or_sections=self.action_dim, dim=-3)
            if self.softplus(std).isnan().any():
                print("std is nan")

            # add a small constant to std to prevent numerical instability
            return (mean, self.softplus(std) + self.epsilon), None

    def get_action_mean(self, _obs):
        _features = self.backbone(_obs)
        _pred = self.last_layer(_features)

        if self._is == "critic":
            raise ValueError("Cannot get action mean from critic network.")
        if self._is == "actor":
            # split the output into mean and std
            mean, _ = torch.split(_pred, split_size_or_sections=self.action_dim, dim=-3)
            return mean

    def sample_action(self, _obs):
        assert not self._is == "critic", "Cannot sample action from critic network."
        act_mean_std, _ = self.forward(_obs)
        mean, std = act_mean_std
        action = torch.normal(mean, std)
        return action
