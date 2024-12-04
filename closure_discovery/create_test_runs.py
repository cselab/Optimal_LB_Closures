"""
this script creates runs the reference simulations used for testing closure models.
"""

import numpy as np
import subprocess

# Seed and common parameters
seed = 33
T = 227
DEVICE = 3  # For execution on multiple GPUs, adjust this number as needed

# Flow configurations
flow_configs = [
    {"flow": "Kolmogorov", "Re": 10000, "lambda": 1, "setups": ["log", "glob", "interp"]},
    {"flow": "Decaying", "Re": 10000, "lambda": 1, "setups": ["log", "glob", "interp"]},
    {"flow": "Kolmogorov", "Re": 100000, "lambda": 2, "setups": ["log", "glob", "interp"]},
]

# Function to execute command for a given configuration
def execute_command(flow, Re, lamb, setup, T, seed, device):
    command = (
        f'CUDA_VISIBLE_DEVICES={device} PYTHONPATH=..:../XLB python execute_model.py'
        f' --t_wish {T} --lamb {lamb} --seed {seed} --flow {flow} --setup {setup} --Re {Re}'
    )
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        exit

# Iterate over each flow configuration and run the commands
for config in flow_configs:
    print(f"Running for flow: {config['flow']} with Re: {config['Re']}")
    for setup in config["setups"]:
        execute_command(config["flow"], config["Re"], config["lambda"], setup, T, seed, DEVICE)
