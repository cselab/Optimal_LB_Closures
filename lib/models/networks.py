import numpy as np
import torch
from torch import nn
from tianshou.data.batch import Batch

SIGMA_MIN = -20
SIGMA_MAX = 2


class FcNN(nn.Module):

    def __init__(self, device="cpu", in_channels=1, feature_dim=3, out_channels=1, padding_mode="circular"):
        super(FcNN, self).__init__()
        self.device = device
        
        ### Convolutional section
        self.fcnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=feature_dim, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True,padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]
        logits = self.fcnn(obs.reshape(batch, 1, 128, 128))

        return logits, state



# different to tianshou, this network has activation functions in the last layer such that 
# the constraints |mu| <= 1, 0<=sigma<=1 are satisfyed automatically
class MyFCNNActorProb(nn.Module):

    def __init__(self, action_shape, device="cpu", in_channels=1, feature_dim=3, out_channels=1, padding_mode="circular"):
        super(MyFCNNActorProb, self).__init__()
        self.device = device
        self.output_shape = action_shape
        
        ### Convolutional section
        self.fcnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=feature_dim, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True,padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
        )

        self.mu = nn.Sequential(nn.Conv2d(in_channels=feature_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
                         nn.Tanh()
        )
        self.sigma = nn.Sequential(nn.Conv2d(in_channels=feature_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
                         nn.Sigmoid()
        )
        #initialize bias to quarantee that the network starts with max standard deviation
        #TODO: maybe torch.no_grad disables changing of beta at all times -> check this
        #with torch.no_grad():
        #   self.sigma[0].bias.fill_(0.1)
        
        print(f"bias is initialized to {self.sigma[0].bias}")
        
        
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]

        logits = self.fcnn(obs.reshape(batch, 1, 128, 128))
        mu = self.mu(logits)
        sigma = self.sigma(logits)
        #if not self._unbounded:
        #    mu = self._max * torch.tanh(mu)
        #if self._c_sigma:
        #    sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        #else:
        #    shape = [1] * len(mu.shape)
        #    shape[1] = -1
        #    sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        
        #mu, sigma = mu.reshape(self.output_shape), sigma.reshape(self.output_shape)
        mu, sigma = mu.reshape(batch,128,128), sigma.reshape(batch,128,128)
        return (mu, sigma), state



# 2nd version of myFCNN. It outputs a constant sigma to check for high variance errors
class MyFCNNActorProb2(nn.Module):

    def __init__(self, action_shape, device="cpu", in_channels=1, feature_dim=3, out_channels=1, padding_mode="circular"):
        super(MyFCNNActorProb2, self).__init__()
        self.device = device
        self.output_shape = action_shape
        
        ### Convolutional section
        self.fcnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=feature_dim, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True,padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
            nn.Tanh()
        )

        
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]

        mu = self.fcnn(obs.reshape(batch, 1, 128, 128)).reshape(batch,128,128)
        sigma = torch.ones(batch, 128,128, device=self.device)*0.1
        
        return (mu, sigma), state



