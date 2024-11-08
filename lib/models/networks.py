import numpy as np
import torch
from torch import nn
from tianshou.data.batch import Batch

SIGMA_MIN = -20
SIGMA_MAX = 2


class FcNN(nn.Module):

    def __init__(self, in_channels=1, feature_dim=3, out_channels=1, padding_mode="circular", device="cpu"):
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
            print("assertion")
        #   obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        #batch = obs.shape[0]
        #logits = self.fcnn(obs.reshape(batch, 1, 128, 128))
        #return logits, state
        return self.fcnn(obs)


# different to tianshou, this network has activation functions in the last layer such that 
# the constraints |mu| <= 1, 0<=sigma<=1 are satisfyed automatically
class MyFCNNActorProb(nn.Module):

    def __init__(self, device="cpu", in_channels=1, feature_dim=3, out_channels=1, padding_mode="circular"):
        super(MyFCNNActorProb, self).__init__()
        self.device = device
        
        ### Convolutional section
        self.fcnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=feature_dim, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True,padding_mode=padding_mode),
            nn.Tanh(),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
            nn.Tanh(),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
            nn.Tanh(),
        )

        self.mu = nn.Sequential(nn.Conv2d(in_channels=feature_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
                         nn.Tanh()
        )
        self.sigma = nn.Sequential(nn.Conv2d(in_channels=feature_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
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

        logits = self.fcnn(obs.reshape(batch, -1, 128, 128))
        mu = self.mu(logits)
        sigma = self.sigma(logits)
        mu, sigma = mu.reshape(batch,128,128), sigma.reshape(batch,128,128)
        return (mu, sigma), state
    

# different to tianshou, this network has activation functions in the last layer such that 
# the constraints |mu| <= 1, 0<=sigma<=1 are satisfyed automatically
class MyFCNNCriticProb(nn.Module):

    def __init__(self, device="cpu", in_channels=1, feature_dim=3, out_channels=1, padding_mode="circular"):
        super(MyFCNNCriticProb, self).__init__()
        self.device = device
        
        ### Convolutional section
        self.fcnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=feature_dim, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True,padding_mode=padding_mode),
            nn.Tanh(),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
            nn.Tanh(),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
            nn.Tanh(),
        )

        self.value = nn.Sequential(nn.Conv2d(in_channels=feature_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
                         nn.Tanh()
        )
        
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]

        logits = self.fcnn(obs.reshape(batch, -1, 128, 128))
        values = self.value(logits)
        values = values.reshape(batch,128,128)
        return values


class MyFcnnActor(nn.Module):

    def __init__(self, backbone, device="cpu", in_channels=1, feature_dim=1, out_channels=1, padding_mode="circular"):
        super(MyFcnnActor, self).__init__()
        self.device = device
        self.backbone = backbone

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
        #print(f"bias is initialized to {self.sigma[0].bias}")
        
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]

        logits = self.backbone(obs.reshape(batch, 1, 128, 128))
        mu = self.mu(logits).reshape(batch,128,128)
        sigma = self.sigma(logits).reshape(batch,128,128)
    
        return (mu, sigma), state


class MyCritc(nn.Module):

    def __init__(self, backbone, device="cpu", in_features=16384, out_features=1):
        super(MyCritc, self).__init__()
        self.device = device
        self.backbone = backbone

        self.linear = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(True),
        )

        
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]

        logits = self.backbone(obs.reshape(batch, 1, 128, 128))
        obs = obs.reshape(batch, -1)
        print("logits shape: ", logits.shape)
        adv = self.linear(logits.reshape(batch, -1))
        print("adv shape: ", adv.shape)
        adv = adv.reshape(batch)
        print(adv)
    
        return adv, state


class FcNN_flattend(nn.Module):

    def __init__(self, in_channels=1, feature_dim=3, out_channels=1, padding_mode="circular", device="cpu"):
        super(FcNN_flattend, self).__init__()
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
            print("reshaping")
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]
        logits = self.fcnn(obs.reshape(batch, 1, 128, 128))
        logits = logits.reshape(batch, -1)
        print("logits shape: ", logits.shape)
        return logits, state


class Backbone(nn.Module):

    def __init__(self, in_channels=1, out_size=64, device="cpu"):
        super(Backbone, self).__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 2, 3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(2, 4, 3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2,2)
        )
        
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(512, out_size),
            nn.Tanh(),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=device)
        batch = obs.shape[0]

        obs = self.encoder_cnn(obs.reshape(batch, -1, 128, 128))
        obs = obs.reshape(batch, -1)
        logits = self.encoder_lin(obs)

        return logits, state


class FcNN_to_critic_converter(nn.Module):

    def __init__(self, fcnn_backbone, device="cpu"):
        super().__init__()
        self.device = device
        self.fcnn_backbone = fcnn_backbone
        
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]
        logits = self.fcnn_backbone(obs.reshape(batch, 1, 128, 128))
        logits = logits.reshape(batch, -1)
        return logits, state


# deeper networks!
class MyFCNNActorProb2(nn.Module):

    def __init__(self, device="cpu", in_channels=1, feature_dim=3, out_channels=1, padding_mode="circular"):
        super().__init__()
        self.device = device
        self.fcnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=2, dilation=2, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=3, dilation=3, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=4, dilation=4, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=3, dilation=3, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=2, dilation=2, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            )

        self.mu = nn.Sequential(nn.Conv2d(in_channels=feature_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
                         nn.Tanh()
        )
        self.sigma = nn.Sequential(nn.Conv2d(in_channels=feature_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                         bias=True, padding_mode=padding_mode),
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

        logits = self.fcnn(obs.reshape(batch, -1, 128, 128))
        mu = self.mu(logits)
        sigma = self.sigma(logits)
        mu, sigma = mu.reshape(batch,128,128), sigma.reshape(batch,128,128)
        return (mu, sigma), state
    

# different to tianshou, this network has activation functions in the last layer such that 
# the constraints |mu| <= 1, 0<=sigma<=1 are satisfyed automatically
class MyFCNNCriticProb2(nn.Module):

    def __init__(self, device="cpu", in_channels=1, feature_dim=64, out_channels=1, padding_mode="circular"):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=2, dilation=2, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=3, dilation=3, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=4, dilation=4, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=3, dilation=3, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=2, dilation=2, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=out_channels, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode)
            )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]

        values = self.model(obs.reshape(batch, -1, 128, 128))
        values = values.reshape(batch,128,128)
        return values
    

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
        return (mu, sigma), state
    

#local actor net with bigger perceptive field
class local_actor_net2(nn.Module):

    def __init__(self, device="cpu", in_channels=9, feature_dim=32, out_channels=1, padding_mode="circular"):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
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

        logits = self.model(obs.reshape(batch, -1, 128, 128))
        mu = self.mu(logits)
        sigma = self.sigma(logits)
        mu, sigma = mu.reshape(batch,128,128), sigma.reshape(batch,128,128)
        return (mu, sigma), state
    

#local actor net with bigger perceptive field
class local_critic_net2(nn.Module):

    def __init__(self, device="cpu", in_channels=9, feature_dim=16, out_channels=1, padding_mode="circular"):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=out_channels, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode)
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]

        values = self.model(obs.reshape(batch, -1, 128, 128))
        values = values.reshape(batch,128,128)
        return values


#local critic net with bigger perceptive field
class local_critic_net3(nn.Module):

    def __init__(self, device="cpu", in_channels=9, feature_dim=16, out_channels=1, padding_mode="circular"):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=feature_dim, kernel_size=9, stride=1,
                       padding=4, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=9, stride=1,
                       padding=4, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=7, stride=1,
                       padding=3, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=7, stride=1,
                       padding=3, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=out_channels, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]

        values = self.model(obs.reshape(batch, -1, 128, 128))
        values = values.reshape(batch,128,128)
        return values



class walker_actor(nn.Module):

    def __init__(self, device="cpu", in_channels=24, feature_dim=128, out_channels=4):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(in_channels, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
        )

        self.mu = nn.Sequential(nn.Linear(feature_dim, out_channels),
                         nn.Tanh(),
        )
        self.sigma = nn.Sequential(nn.Linear(feature_dim, out_channels),
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

        logits = self.model(obs.reshape(batch, -1))
        mu = self.mu(logits)
        sigma = self.sigma(logits)
        mu, sigma = mu.reshape(batch,-1), sigma.reshape(batch,-1)
        return (mu, sigma), state
    


class walker_critic(nn.Module):
    def __init__(self, device="cpu", in_channels=24, feature_dim=128, out_channels=1):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(in_channels, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, out_channels),
        )
                
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]

        logits = self.model(obs.reshape(batch, -1))
        return logits



class central_actor_net(nn.Module):

    def __init__(self, device="cpu", in_channels=9, feature_dim=128, out_channels=1, padding_mode="circular"):
        super().__init__()
        self.device = device
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
        #self.sigma = nn.Sequential(nn.Conv2d(in_channels=feature_dim, out_channels=out_channels, kernel_size=1, stride=1,
        #               padding=0, bias=True),
        #                 nn.Softplus()
        #)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize the weights of the last layer of self.fcnn
        with torch.no_grad():
            #self.fcnn[4].weight *= 1/100
            self.mu[0].weight *= 1/100
            #self.sigma[0].weight *= 1/100
            #self.sigma[0].bias.fill_(-0.9)
        
        
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]

        logits = self.model(obs.reshape(batch, -1, 128, 128))
        mu = self.mu(logits)
        #sigma = self.sigma(logits)
        mu = mu.reshape(batch,128,128)
        return mu, state


#central critic
class central_critic_net1(nn.Module):

    def __init__(self, device="cpu", in_channels=10, feature_dim=32, out_channels=1, padding_mode="circular"):
        super().__init__()
        self.device = device
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

    def forward(self, obs, act, state=None, info={}):
        #print(obs.shape, act.shape)
        batch = obs.shape[0]
        act = act.reshape(batch,-1,128,128)
        obs = obs.reshape(batch,-1,128,128)
    
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        if not isinstance(act, torch.Tensor):
            act = torch.tensor(act, dtype=torch.float, device=self.device)
        
        state = torch.cat((obs, act), 1)
        #print(state.shape)
        values = self.model(state)
        #print(values.shape)
        values = values.reshape(batch, -1)
        #print(values.shape)
        values = self.fcnn(values)
        return values


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
        return (mu, sigma), state


#central critic for TD3
class central_actor_net3(nn.Module):

    def __init__(self, device="cpu", in_channels=9, feature_dim=32, out_channels=1, padding_mode="circular"):
        super().__init__()
        self.device = device
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

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize the weights of the last layer of self.fcnn
        with torch.no_grad():
            #self.fcnn[4].weight *= 1/100
            self.mu[0].weight *= 1/100

    
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]

        logits = self.model(obs.reshape(batch, -1, 128, 128))
        logits = logits.reshape(batch, -1)
        logits = self.fcnn(logits)
        mu = self.mu(logits)
        return mu, state


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

        # **Adjust the number of stride=2 layers based on the receptive field**
        # Skip additional downsampling, ensure you don't reduce beyond N x N
        for _ in range(max(int(receptive_field / 4) - 1, 0)):
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
        
        return (mu, sigma), state


class FullyConvNet_interpolating_agents2(nn.Module):

    def __init__(self, N, in_channels=1, device="cpu", padding_mode="circular", nx=128):
        super().__init__()
        self.N = N #number of agents
        self.nx = nx
        receptive_field = nx // N  # Receptive field calculation based on N
        self.device = device

        layers = []
        # First convolutional block (no downsampling yet)
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode))
        layers.append(nn.ELU())
        layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode)) 
        layers.append(nn.ELU())

        # Second convolutional block with downsampling
        layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode)) 
        layers.append(nn.ELU())
        
        # Third block: Downsampling to achieve the right spatial size
        layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode))
        layers.append(nn.ELU())

        # **Adjust the number of stride=2 layers based on the receptive field**
        # Skip additional downsampling, ensure you don't reduce beyond N x N
        for _ in range(max(int(receptive_field / 4) - 1, 0)):
            layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode))  # No downsampling here
            layers.append(nn.ELU())

        # Final convolution to map to output N x N (adjust stride to 1 to avoid further downsampling)
        layers.append(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode))
        layers.append(nn.ELU())

        self.model = nn.Sequential(*layers)

        # Separate layers for mu and sigma
        self.mu = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True, padding_mode=padding_mode),
            #nn.Tanh()
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
        
        return (mu, sigma), state
    

#central critic for PPO
class central_critic_net4(nn.Module):

    def __init__(self, device="cpu", in_channels=9, feature_dim=32, out_channels=1, padding_mode="circular", nx=128):
        super().__init__()
        self.device = device
        self.nx = nx
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=9, stride=4,
                       padding=4, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2,
                       padding=2, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=True, padding_mode=padding_mode),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2,2),
        )

        self.fcnn = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
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
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=k, stride=k-1, padding=1, padding_mode=padding_mode))
        layers.append(nn.ELU())

        # Second convolutional
        layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)) 
        layers.append(nn.ELU())
        
        # Third block
        layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0))
        layers.append(nn.ELU())

        self.model = nn.Sequential(*layers)

        self.mu = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
                            #nn.Tanh()
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
        
        return (mu, sigma), state



class local_actor_net_fast(nn.Module):

    def __init__(self, device="cpu", in_channels=6, feature_dim=128, out_channels=1, padding_mode="circular", nx=128):
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
        #if not isinstance(obs, torch.Tensor):
        #obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        batch = obs.shape[0]
        logits = self.model(obs)
        mu = self.mu(logits)
        #sigma = self.sigma(logits)
        #mu, sigma = mu.reshape(batch,self.nx,self.nx), sigma.reshape(batch,self.nx,self.nx)
        return mu.reshape(self.nx, self.nx)