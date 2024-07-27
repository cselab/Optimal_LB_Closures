import numpy as np
import torch
from torch import nn
from tianshou.data.batch import Batch


class Backbone(nn.Module):

    def __init__(self, device="cpu", in_channels=1, feature_dim=64, out_channels=1, padding_mode="circular"):
        super(Backbone, self).__init__()
        
        ### Convolutional section
        self.fcnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=feature_dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
                      padding_mode=padding_mode)),
            nn.ReLU(True),
            nn.Conv2d(2, 4, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )
        
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(512, out_size),
            nn.ReLU(True),
        )
        

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=device)
        batch = obs.shape[0]

        obs = self.encoder_cnn(obs.reshape(batch, 1, 128, 128))
        obs = obs.reshape(batch, -1)
        logits = self.encoder_lin(obs)

        return logits, state