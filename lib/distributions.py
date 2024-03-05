import torch


class ElementwiseNormal(torch.distributions.normal.Normal):
    """
    Joint elementwise normal distribution with same standard deviation for each entry.
    log_prob returns joint probability for each pixel.
    """
    constant_noise = None
    marl = None

    def __init__(self, mean: torch.Tensor, std: torch.Tensor = None):
        """
        :param mean: mean of the distribution, shape (channels, height, width)
        """
        AssertionError(len(mean.shape) == 3,
                       f"mean.shape: {mean.shape} is wrong, needs to be of dim (channels, height, width)")
        if std is None:
            _sigma = torch.ones_like(mean) * ElementwiseNormal.constant_noise
        else:
            _sigma = std
        super().__init__(mean, _sigma)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if ElementwiseNormal.marl:
            return self.marl_log_prob(value)
        else:
            return self.single_agent_log_prob(value)

    def marl_log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # One probability per batch and pixel!
        log_probs = super().log_prob(value).sum(dim=(-3))  # sum over ONLY action dimension
        AssertionError(len(log_probs.shape) == len(value.shape),
                       f"dims of input: {value.shape} is wrong. Needs to be of dim (batch, channels, height, width)")
        return log_probs

    def single_agent_log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # One probability per batch
        log_probs = super().log_prob(value).sum(dim=(-3, -2, -1))  # sum over entire image and action dimension
        AssertionError(log_probs.shape == value.shape[0:1],
                       "log_prob shape does not match expected shape")
        return log_probs

    def entropy(self):
        """
        :return: per pixel entropy of the distribution, shape (channels, height, width)
        :return:
        """
        return super().entropy().sum(dim=-3)  # sum over action dimension

