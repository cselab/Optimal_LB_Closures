from lib.models.pretrained_denoisers import AdvectionIRCNN, BurgersIRCNN
from lib.models.wrappers import MarlModel
from tianshou.utils.net.common import ActorCritic
#from lib.models.networks import FcNN, MyFCNNActorProb, MyFCNNActorProb2, MyFcnnActor, MyCritc, Backbone
from lib.models.networks import *

def get_actor_critic(name, device, env, action_dim):
    if name == "IRCNN":
        if env == "advection":
            backbone = AdvectionIRCNN()
        elif env == "burgers":
            backbone = BurgersIRCNN()
        else:
            raise NotImplementedError(f"No backbone wrapper for {env} implemented.")

        actor = MarlModel(backbone=backbone,
                          _is="actor",
                          action_dim=action_dim).to(device)
        critic = MarlModel(backbone=backbone,
                           _is="critic",
                           action_dim=action_dim).to(device)
        actor_critic = ActorCritic(actor, critic)
        return actor_critic
    else:
        raise ValueError(f'No model with name {name} found')

