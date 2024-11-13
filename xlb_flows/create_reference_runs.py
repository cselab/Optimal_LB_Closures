"""
this script creates runs the reference simulations used for testing closure models.
"""

import numpy as np
import subprocess

# Seed and common parameters
seed = 33
T = 227
DEVICE = 1  # For execution on multiple GPUs, adjust this number as needed

# Flow configurations
flow_configs = [
    {"flow": "Kolmogorov", "Re": 1e4, "lambs": [1, 1, 2, 16], "models": ["", "KBC", "", ""]},
    {"flow": "Decaying", "Re": 1e4, "lambs": [1, 1, 2, 16], "models": ["", "KBC", "", ""]},
    {"flow": "Kolmogorov", "Re": 1e5, "lambs": [2, 2, 4, 16], "models": ["", "KBC", "", ""]}
]

# Function to execute command for a given configuration
def execute_command(flow, Re, lamb, model, T, seed, device):
    flow_model = f"{flow}_{model}"
    command = (
        f'CUDA_VISIBLE_DEVICES={device} PYTHONPATH=..:../XLB python xlb_flows/run_klmgrv.py'
        f'--t_wish {T} --lamb {lamb} --seed {seed} --flow {flow_model} --Re {Re}'
    )
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        exit

# Iterate over each flow configuration and run the commands
for config in flow_configs:
    print(f"Running for flow: {config['flow']} with Re: {config['Re']}")
    for lamb, model in zip(config["lambs"], config["models"]):
        execute_command(config["flow"], config["Re"], lamb, model, T, seed, DEVICE)
