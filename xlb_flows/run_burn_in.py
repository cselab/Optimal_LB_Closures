"""
this script iterates through the seeds and runs a burn in simulation 
and saves the final velocity and density fields. 
The seed is used to initilaize the burn in simulation. 
By default the burn in simulation has resolution 2048 and runs
for T=645. 

Expect the execution for one seed to take several hours (if executed
on one GPU similar to a NVIDIA A100 80GB)
"""

import numpy as np
import subprocess

# seeds used to reproduce results:
seeds = np.array([102, 99, 33])

# define GPU number used for execution
DEVICE = 1 # for execution on multiple GPUs try e.g. DEVICE = 2,3

# Iterate through each seed and execute the command
for seed in seeds:
    command = f'CUDA_VISIBLE_DEVICES={DEVICE} PYTHONPATH=..:../XLB python run_klmgrv.py --t_wish 645 --lamb 16 --Re 10000 --seed {seed} --flow "Burn_in"'
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        break
    print(f"Finished execution for seed: {seed}")