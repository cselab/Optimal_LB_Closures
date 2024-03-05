import os
import torch
import torch.nn as nn
from collections import OrderedDict


# CODE ADAPTED FROM: https://github.com/cszn/KAIR/tree/master/models TO LOAD PRETRAINED WEIGHTS
def sequential(*args):
    """Advanced nn.Sequential.
    Args:
        nn.Sequential, nn.Module
    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# IRCNN denoiser
class IRCNN(nn.Module):
    def __init__(self, in_nc=3, feature_dim=64, padding_mode="circular"):
        """
        # ------------------------------------
        denoiser of IRCNN
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(IRCNN, self).__init__()
        self.feature_dim = feature_dim
        L = []
        out_nc = 3  # specify this only to allow loading of weights
        L.append(
            nn.Conv2d(in_channels=in_nc, out_channels=feature_dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
                      padding_mode=padding_mode))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=True,
                           padding_mode=padding_mode))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1, padding=3, dilation=3, bias=True,
                           padding_mode=padding_mode))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1, padding=4, dilation=4, bias=True,
                           padding_mode=padding_mode))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1, padding=3, dilation=3, bias=True,
                           padding_mode=padding_mode))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=True,
                           padding_mode=padding_mode))
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(in_channels=feature_dim, out_channels=out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
                      padding_mode=padding_mode))
        self.model = sequential(*L)

        # remove last layer, since we want to use the output of the second last layer as feature vector
        self.model.pop(-1)

    def forward(self, x):
        return self.model(x)


class AdvectionIRCNN(IRCNN):
    def __init__(self, feature_dim=64):
        # 3 input channels: (vel_x, vel_y, convolved_img)
        super().__init__(in_nc=3, feature_dim=feature_dim, padding_mode="circular")

    def forward(self, _obs):
        _device = next(self.parameters()).device
        _convolved_img = torch.from_numpy(_obs['domain']).to(_device)
        _velocity_field = torch.from_numpy(_obs['velocity_field']).to(_device)
        # create batch dimension if single sample input
        if len(_convolved_img.shape) == 3:
            _convolved_img = _convolved_img.unsqueeze(0)
            _velocity_field = _velocity_field.unsqueeze(0)
        # put input on same device as model parameters - input gets passed in as numpy array
        x = torch.cat((_convolved_img, _velocity_field), dim=1)
        return self.model(x)


class BurgersIRCNN(IRCNN):
    def __init__(self, feature_dim=64):
        # 3 input channels: (vel_x, vel_y, convolved_img)
        super().__init__(in_nc=2, feature_dim=feature_dim, padding_mode="circular")

    def forward(self, _obs):
        _device = next(self.parameters()).device
        velocity_field = torch.from_numpy(_obs['velocity_field']).to(_device)
        # create batch dimension if single sample input
        if len(velocity_field.shape) == 3:
            velocity_field = velocity_field.unsqueeze(0)
        # put input on same device as model parameters - input gets passed in as numpy array
        return self.model(velocity_field)


if __name__ == '__main__':

    model = IRCNN()
    dummy_input = torch.randn(10, 1, 64, 64)
    output = model(dummy_input)
    print(output.shape)
