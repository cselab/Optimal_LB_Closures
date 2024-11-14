import os
import jax
import numpy as np
import argparse
import torch

from XLB.src.utils import *
from XLB.src.lattice import LatticeD2Q9
from xlb_flows.utils import *
from xlb_flows.kolmogorov_2d import *

from lib.models import local_actor_net_fast
from collections import OrderedDict

import jax
import jax.numpy as jnp
from flax import linen as nn
import time


device = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(42)
jax.config.update('jax_enable_x64', True)

flow_classes = {
    "Kolmogorov_ClosureRL": Kolmogorov_flow_ClosureRL,
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--Re", type=int, default=1e4)
    parser.add_argument("--upsilon", type=int, default=1) #scales non-dim velocity    
    parser.add_argument("--lamb", type=int, default=1) #scales spacial resolution
    parser.add_argument("--t_wish", type=int, default=18) # or 227 for visualization
    parser.add_argument("--print_rate", type=int, default=32) # take 1 for dataset creation
    parser.add_argument("--flow", type=str, default="Kolmogorov") 
    parser.add_argument("--model", type=str, default="ClosureRL")
    parser.add_argument("--setup", type=str, default="loc")
    parser.add_argument("--measure_speedup", type=int, default=False)

    return parser.parse_known_args()[0]


class LocalActorNetFast(nn.Module):
    in_channels: int = 6
    feature_dim: int = 128
    out_channels: int = 1
    nx: int = 128
    
    def setup(self):
        # Define the layers (Conv2D, ReLU, etc.) in Flax
        self.conv1 = nn.Conv(features=self.feature_dim, kernel_size=(1, 1), strides=(1, 1), padding='VALID')
        self.conv2 = nn.Conv(features=self.feature_dim, kernel_size=(1, 1), strides=(1, 1), padding='VALID')
        self.mu_conv = nn.Conv(features=self.out_channels, kernel_size=(1, 1), strides=(1, 1), padding='VALID')
        self.sigma_conv = nn.Conv(features=self.out_channels, kernel_size=(1, 1), strides=(1, 1), padding='VALID')
    
    def _initialize_weights(self, prng_key):
        # For custom weight initialization, you can manipulate the weights of conv layers directly
        params = self.init(prng_key, jnp.ones((1, self.nx, self.nx, self.in_channels)))
        
        # Initialize the weights of mu and sigma layers with custom scaling
        params['mu_conv']['kernel'] *= 1 / 100
        params['sigma_conv']['kernel'] *= 1 / 100
        params['sigma_conv']['bias'] = jnp.full_like(params['sigma_conv']['bias'], -0.9)
        
        return params
    
    def __call__(self, obs, state=None, info={}):
        # Forward pass for the model
        x = self.conv1(obs)
        x = nn.relu(x)  # ReLU activation
        x = self.conv2(x)
        x = nn.relu(x)  # ReLU activation
        
        mu = self.mu_conv(x)
        mu = jax.nn.tanh(mu)  # tanh activation
        
        sigma = self.sigma_conv(x)
        sigma = jax.nn.softplus(sigma)  # softplus activation
        
        # Optionally reshape if necessary (matching original output shape)
        return mu.reshape((self.nx, self.nx))


def convert_pytorch_to_flax(actor_parameters, flax_model, N):
    # Initialize the Flax model with dummy input
    key = jax.random.PRNGKey(0)
    input_shape = (1, N, N, 6)  # Example input shape
    flax_params = flax_model.init(key, jnp.ones(input_shape))

    # Initialize a copy of the Flax params to be updated
    flax_params_copy = flax_params.copy()

    # Manually map the PyTorch weights to Flax keys
    for pytorch_key, pytorch_value in actor_parameters.items():
        if pytorch_key == "model.0.weight":
            flax_params_copy['params']["conv1"]["kernel"] = jnp.array(pytorch_value.permute(2, 3, 1, 0).cpu().numpy())
        elif pytorch_key == "model.0.bias":
            flax_params_copy['params']["conv1"]["bias"] = jnp.array(pytorch_value.cpu().numpy())
        elif pytorch_key == "model.2.weight":
            flax_params_copy['params']["conv2"]["kernel"] = jnp.array(pytorch_value.permute(2, 3, 1, 0).cpu().numpy())
        elif pytorch_key == "model.2.bias":
            flax_params_copy['params']["conv2"]["bias"] = jnp.array(pytorch_value.cpu().numpy())
        elif pytorch_key == "mu.0.weight":
            flax_params_copy['params']["mu_conv"]["kernel"] = jnp.array(pytorch_value.permute(2, 3, 1, 0).cpu().numpy())
        elif pytorch_key == "mu.0.bias":
            flax_params_copy['params']["mu_conv"]["bias"] = jnp.array(pytorch_value.cpu().numpy())
        elif pytorch_key == "sigma.0.weight":
            flax_params_copy['params']["sigma_conv"]["kernel"] = jnp.array(pytorch_value.permute(2, 3, 1, 0).cpu().numpy())
        elif pytorch_key == "sigma.0.bias":
            flax_params_copy['params']["sigma_conv"]["bias"] = jnp.array(pytorch_value.cpu().numpy())
        else:
            print(f"Warning: No matching Flax parameter for PyTorch key {pytorch_key}")

    return flax_params_copy


if __name__ == "__main__":

    args = get_args()
    print(args)

    try:
        
        u0_path = os.path.expanduser(
             f"~/CNN-MARL_closure_model_discovery/"
             "xlb_flows/init_fields/"
             f"velocity_kolmogorov_2d_910368_s{args.seed}.npy"
             )
        rho0_path = os.path.expanduser(
             f"~/CNN-MARL_closure_model_discovery/"
             "xlb_flows/init_fields/"
             f"density_kolmogorov_2d_910368_s{args.seed}.npy"
             )
    except:
        print(f"no file found for given seed = {args.seed}")
    
    print(f"seed = {args.seed}")

    precision = "f64/f64"
    lattice = LatticeD2Q9(precision)
    seed = args.seed
    Re=args.Re
    upsi=args.upsilon
    lamb = args.lamb # N = lamb*128 
    kwargs, endTime, T, N = get_kwargs(u0_path=u0_path,
                                       rho0_path=rho0_path,
                                       T_wish=args.t_wish,
                                       lamb=lamb,
                                       Re=Re, n=4,
                                       upsilon=upsi,
                                       seed=seed,
                                       print_rate=args.print_rate,
                                       measure_speedup=args.measure_speedup)

    # Retrieve the class based on args.flow
    flow_type = args.flow + '_' + args.model
    flow_class = flow_classes.get(flow_type)

    #get acotor:
    DUMP_PATH = "../dump/Kolmogorov22_ppo_cgs1_fgs16/"
    ID = "20241021-110311"
    
    #actor = FullyConvNet_interpolating_agents(in_channels=6, N=args.num_agents, device=device).to(device)
    actor = local_actor_net_fast(in_channels=6, device=device, nx=(128*args.lamb)).to(device)
    #load actor from state_dict
    actor_parameters = OrderedDict()
    state_dict = torch.load(DUMP_PATH+'policy_'+ID+'.pth')
    for key, value in state_dict.items():
        if key.startswith("actor."):
            new_key = key.replace("actor.", "")
        elif key.startswith("_actor_critic.actor."):
            new_key = key.replace("_actor_critic.actor.", "")
        else:
            continue  # Skip keys not related to the actor
        actor_parameters[new_key] = value
    model = LocalActorNetFast(nx=N)
    flax_params = convert_pytorch_to_flax(actor_parameters, model, N)

    # Check if the flow exists and create an instance
    if flow_class:
        sim = flow_class(model, flax_params, **kwargs)
    else:
        print("flow does not exist")

    if args.flow == "Burn_in":
        create_and_navigate_to("init_fields")
    
    else:
        # Define folder paths
        main_folder = (f"../results/re{int(Re)}_T{int(T)}_S{seed}_{args.flow}_runs")
        run_folder = f"re{int(Re)}_T{int(T)}_N{int(N)}_S{seed}_U{upsi}_{args.model+'-'+args.setup}"

        create_and_navigate_to(main_folder)
        create_and_navigate_to(run_folder)

        # remove all .png files in the current directory
        if not sim.restore_checkpoint:
            os.system("rm -rf ./*.png")
            os.system("rm -rf ./*.npy")
            os.system("rm -rf ./*.json")


    start_time=time.time()
    sim.run(endTime)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time}")