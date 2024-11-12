import os
import sys
import json
import jax
import numpy as np
import argparse

from XLB.src.utils import *
from XLB.src.lattice import LatticeD2Q9
from xlb_flows.utils import *
from xlb_flows.kolmogorov_2d import Kolmogorov_flow, Decaying_flow, Burn_in_Kolmogorov_flow

np.random.seed(42)
jax.config.update('jax_enable_x64', True)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--Re", type=int, default=1e4)
    parser.add_argument("--upsilon", type=int, default=1) #scales non-dim velocity    
    parser.add_argument("--lamb", type=int, default=1) #scales spacial resolution
    parser.add_argument("--t_wish", type=int, default=18) # or 227 for visualization
    parser.add_argument("--print_rate", type=int, default=32) # take 1 for dataset creation
    parser.add_argument("--flow", type=str, default="Kolmogorov") #opitons "Kolmogorov", "Decaying", "Burn_in"

    return parser.parse_known_args()[0]


if __name__ == "__main__":

    args = get_args()
    print(args)

    try:
        u0_path = os.path.expanduser(f"~/XLB/vel_init/velocity_burn_in_909313_s{args.seed}.npy")
        rho0_path = os.path.expanduser(f"~/XLB/vel_init/density_burn_in_909313_s{args.seed}.npy")
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
                                       print_rate=args.print_rate)

    # run the simulation
    if args.flow == "Kolmogorov":
        sim = Kolmogorov_flow(**kwargs)
    elif args.flow == "Decaying":
        sim = Decaying_flow(**kwargs)
    elif args.flow == "Burn_in":
        sim = Burn_in_Kolmogorov_flow(**kwargs)
    else:
        print("flow does not exist")


     # if the folder f"re{int(Re)}_dump" exitst delete it recursively
    if not os.path.exists(f"re{int(Re)}_T{int(T)}_N{int(N)}_S{seed}_U{upsi}_dump"):
        os.makedirs(f"re{int(Re)}_T{int(T)}_N{int(N)}_S{seed}_U{upsi}_dump")
    os.chdir(f"re{int(Re)}_T{int(T)}_N{int(N)}_S{seed}_U{upsi}_dump")
    # remove all .png files in the current directory
    if not sim.restore_checkpoint:
        os.system("rm -rf ./*.png")
        os.system("rm -rf ./*.npy")
        os.system("rm -rf ./*.json")

    if sim.restore_checkpoint:
        sim.kin_list = [] # list that stores kinetic energy values 
        sim.ens_list = [] # list that stores enstrophy values
        sim.scale = None

    sim.run(endTime)
