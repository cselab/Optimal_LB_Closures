import numpy as np
import torch
from torch import nn
from tianshou.data.batch import Batch

SIGMA_MIN = -20
SIGMA_MAX = 2



class local_actor_net(nn.Module):

    def __init__(self, device="cpu", in_channels=9, feature_dim=128, out_channels=1, padding_mode="circular", nx=128):
        super().__init__()
        self.device = device
        self.nx = nx
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=feature_dim, kernel_size=1, stride=1,
                       padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1, stride=1,
                       padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        self.mu = nn.Sequential(nn.Conv2d(in_channels=feature_dim, out_channels=out_channels, kernel_size=1, stride=1,
                       padding=0, bias=True),
                         nn.Tanh()
        )
        self.sigma = nn.Sequential(nn.Conv2d(in_channels=feature_dim, out_channels=out_channels, kernel_size=1, stride=1,
                       padding=0, bias=True),
                         nn.Softplus()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize the weights of the last layer of self.fcnn
        with torch.no_grad():
            #self.fcnn[4].weight *= 1/100
            self.mu[0].weight *= 1/100
            self.sigma[0].weight *= 1/100
            self.sigma[0].bias.fill_(-0.9)
        
        
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]

        logits = self.model(obs.reshape(batch, -1, self.nx, self.nx))
        mu = self.mu(logits)
        sigma = self.sigma(logits)
        mu, sigma = mu.reshape(batch,self.nx,self.nx), sigma.reshape(batch,self.nx,self.nx)
        sigma = torch.max(sigma, torch.full_like(sigma, 1e-6))
        return (mu, sigma), state
    

#central critic for PPO
class central_actor_net2(nn.Module):

    def __init__(self, device="cpu", in_channels=9, feature_dim=32, out_channels=1, padding_mode="circular", nx=128):
        super().__init__()
        self.device = device
        self.nx=nx
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=9, stride=4,
                       padding=4, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2,
                       padding=2, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )

        self.fcnn = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        self.mu = nn.Sequential(
            nn.Linear(64,1),
            nn.Tanh()
        )

        self.sigma = nn.Sequential(
            nn.Linear(64,1),
            nn.Softplus()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize the weights of the last layer of self.fcnn
        with torch.no_grad():
            #self.fcnn[4].weight *= 1/100
            self.mu[0].weight *= 1/100
            self.sigma[0].weight *= 1/100
            self.sigma[0].bias.fill_(-0.9)

    
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]

        logits = self.model(obs.reshape(batch, -1, self.nx, self.nx))
        logits = logits.reshape(batch, -1)
        logits = self.fcnn(logits)
        mu = self.mu(logits)
        sigma = self.sigma(logits)
        sigma = torch.max(sigma, torch.full_like(sigma, 1e-6))
        return (mu, sigma), state


#central critic for PPO
class central_critic_net2(nn.Module):

    def __init__(self, device="cpu", in_channels=9, feature_dim=32, out_channels=1, padding_mode="circular", nx=128):
        super().__init__()
        self.device = device
        self.nx = nx
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=9, stride=4,
                       padding=4, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2,
                       padding=2, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )

        self.fcnn = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)

        batch = obs.shape[0]
        obs = obs.reshape(batch,-1,self.nx,self.nx)
        
        values = self.model(obs)
        values = values.reshape(batch, -1)
        values = self.fcnn(values)
        return values


class FullyConvNet_interpolating_agents(nn.Module):

    def __init__(self, N, in_channels=1, device="cpu", padding_mode="circular", nx=128):
        super().__init__()
        self.N = N #number of agents
        self.nx = nx
        receptive_field = nx // N  # Receptive field calculation based on N
        self.device = device

        layers = []
        # First convolutional block (no downsampling yet)
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode)) 
        layers.append(nn.ReLU())

        # Second convolutional block with downsampling
        layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode)) 
        layers.append(nn.ReLU())
        
        # Third block: Downsampling to achieve the right spatial size
        layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode))  # No downsampling here
        layers.append(nn.ReLU())

        # Final convolution to map to output N x N (adjust stride to 1 to avoid further downsampling)
        layers.append(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode))
        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

        # Separate layers for mu and sigma
        self.mu = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True, padding_mode=padding_mode),
            nn.Tanh()
        )
        self.sigma = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True, padding_mode=padding_mode),
            nn.Softplus()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights for output layers
        with torch.no_grad():
            self.mu[0].weight *= 1/100
            self.sigma[0].weight *= 1/100
            self.sigma[0].bias.fill_(-0.9)
        
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]

        # Pass through the model
        logits = self.model(obs.reshape(batch, -1, self.nx, self.nx))  # Reshape to correct input shape
        mu = self.mu(logits)
        sigma = self.sigma(logits)
        
        # Ensure output shape is (batch, N, N)
        mu, sigma = mu.reshape(batch, self.N, self.N), sigma.reshape(batch, self.N, self.N)
        sigma = torch.max(sigma, torch.full_like(sigma, 1e-6))
        return (mu, sigma), state


#correcting the perceptive filed
class FullyConvNet_interpolating_agents3(nn.Module):

    def __init__(self, N, in_channels=1, device="cpu", padding_mode="circular", nx=128):
        super().__init__()
        self.N = N #number of agents
        self.nx = nx
        self.device = device
        k = nx // N + 1

        layers = []
        # convolutional filter
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=k, stride=k-1, padding=1, padding_mode=padding_mode))
        layers.append(nn.ReLU())

        # Second convolutional
        layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)) 
        layers.append(nn.ReLU())
        
        # Third block
        layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0))
        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

        self.mu = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
                            nn.Tanh()
        )
        self.sigma = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1,padding=0, bias=True),
                            nn.Softplus()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights for output layers
        with torch.no_grad():
            self.mu[0].weight *= 1/100
            self.sigma[0].weight *= 1/100
            self.sigma[0].bias.fill_(-0.9)
        
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]

        # Pass through the model
        logits = self.model(obs.reshape(batch, -1, self.nx, self.nx))  # Reshape to correct input shape
        mu = self.mu(logits)
        sigma = self.sigma(logits)
        
        # Ensure output shape is (batch, N, N)
        mu, sigma = mu.reshape(batch, self.N, self.N), sigma.reshape(batch, self.N, self.N)
        sigma = torch.max(sigma, torch.full_like(sigma, 1e-6))
        return (mu, sigma), state
