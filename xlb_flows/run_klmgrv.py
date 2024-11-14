import os
import jax
import numpy as np
import argparse

from XLB.src.utils import *
from XLB.src.lattice import LatticeD2Q9
from xlb_flows.utils import *
from xlb_flows.kolmogorov_2d import *

import time

np.random.seed(42)
jax.config.update('jax_enable_x64', True)

flow_classes = {
    "Kolmogorov_BGK": Kolmogorov_flow,
    "Decaying_BGK": Decaying_flow,
    "Burn_in_BGK": Burn_in_Kolmogorov_flow,
    "Kolmogorov_KBC": Kolmogorov_flow_KBC,
    "Decaying_KBC": Decaying_flow_KBC
}

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--Re", type=int, default=1e4)
    parser.add_argument("--upsilon", type=int, default=1) #scales non-dim velocity    
    parser.add_argument("--lamb", type=int, default=1) #scales spacial resolution
    parser.add_argument("--t_wish", type=int, default=18) # non-dimensional time
    parser.add_argument("--print_rate", type=int, default=32) # take 1 for dataset creation
    parser.add_argument("--flow", type=str, default="Kolmogorov") 
    parser.add_argument("--model", type=str, default="BGK")
    parser.add_argument("--measure_speedup", type=int, default=False)
   

    return parser.parse_known_args()[0]


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
    
    #correct downsampling & info prints for burn in simulation
    if args.flow == "Burn_in":
        kwargs["downsampling_factor"] = 1

    flow_type = args.flow + '_' + args.model
    flow_class = flow_classes.get(flow_type)
    # Check if the flow exists and create an instance
    if flow_class:
        sim = flow_class(**kwargs)
    else:
        print("flow does not exist")

    if args.flow == "Burn_in":
        create_and_navigate_to("init_fields")
    
    else:
        # Define folder paths
        main_folder = f"../results/re{int(Re)}_T{int(T)}_S{seed}_{args.flow}_runs"
        run_folder = f"re{int(Re)}_T{int(T)}_N{int(N)}_S{seed}_U{upsi}_{args.model}"

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