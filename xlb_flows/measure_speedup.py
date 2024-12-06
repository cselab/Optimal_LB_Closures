import subprocess
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Seed and common parameters
seed = 33
T = 100
DEVICE = 3  # For execution on multiple GPUs, adjust this number as needed


def run_command(command):
    try:
        # Run the command and capture output
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # Extract execution time from output
        for line in result.stdout.splitlines():
            if "Execution time:" in line:
                # Parse and convert execution time to a float
                execution_time = float(line.split(":")[1].strip())
                return execution_time
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
    return None
# Define the commands and arrays to store times
CGS_times = []
RL_times = []

# Commands for ClosureRL
# Assuming `T`, `seed`, and `lambs` are already defined
lambs = np.array([1, 2, 4, 8])
RL_commands = [
    (f"CUDA_VISIBLE_DEVICES={DEVICE} PYTHONPATH=..:../XLB python "
     f"run_klmgrv_RL.py --flow 'Kolmogorov' --model 'ClosureRL' --measure_speedup 1 "
     f"--t_wish {T} --lamb {scale_factor} --seed {seed}")
    for scale_factor in lambs
]

# Commands for CGS
CGS_commands = [
    (f"CUDA_VISIBLE_DEVICES={DEVICE} PYTHONPATH=..:../XLB python "
     f"run_klmgrv.py --flow 'Kolmogorov' --measure_speedup 1 "
     f"--t_wish {T} --lamb {2 * scale_factor} --seed {seed}")
    for scale_factor in lambs
]

# Run RL commands and store execution times
print("running ClosureRL")
for command in tqdm(RL_commands):
    time_taken = run_command(command)
    if time_taken is not None:
        RL_times.append(time_taken)
    else:
        print(f"Failed to get execution time for command: {command}")

# Run CGS commands and store execution times
print("running CGS")
for command in tqdm(CGS_commands):
    time_taken = run_command(command)
    if time_taken is not None:
        CGS_times.append(time_taken)
    else:
        print(f"Failed to get execution time for command: {command}")

# Convert times to numpy arrays
CGS_times = np.array(CGS_times)
RL_times = np.array(RL_times)
#lamb = np.array([1, 2, 4, 8])

# Plotting
RL_names = ["128", "256", "512", "1024"]
CGS_names = ["256", "512", "1024", "2048"]

plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(10, 6))
plt.plot(lambs, RL_times, 'o-', label='ReLBM', color='#d62728')
plt.plot(lambs, CGS_times, 's-', label='LBGK', color='#1f77b4')

# Adding labels to each point
for i, (x, y, rl_name, cgs_name) in enumerate(zip(lambs, RL_times, RL_names, CGS_names)):
    plt.text(x, y, rl_name, color='#d62728', ha='right', va='bottom')
    plt.text(x, CGS_times[i], cgs_name, color='#1f77b4', ha='right', va='top')

# Labels and title
plt.xlabel('Grid scaling (λ)')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.yscale("log")  # Log scale for better visibility of data spread
plt.tight_layout()
plt.savefig("../results/figures/execution_time_comparison.png")  # Save the plot to a file
