# Adaptive Closure Modeling for Lattice Boltzmann Methods using Multi-Agent Reinforcement Learning

## Description
This is the official implementation of the MSc thesis *Adaptive Closure Modeling for Lattice Boltzmann Methods using Multi-Agent Reinforcement Learning*.

Reinforcement Learning (RL) is used for automatic discovery of hybrid turbulence models for Lattice Boltzmann Methods (LBM). LBM offers advantages such as easy access to macroscopic flow features and its complete locality, which allows efficient parallelization. RL eliminates the need for costly direct numerical simulation data during training by using an energy spectrum similarity measure as a reward. We have implemented several multi-agent RL models (ReLBM) with fully convolutional networks, achieving stabilization of turbulent 2D Kolmogorov flows and more accurate energy spectra than traditional LBM turbulence models. An example of the performance of a ReLBM at resolution $N = 128$, compared to a coarse LBGK simulation (128_BGK) and a resolved direct numerical simulation (2048_GK), is shown below. For RL training we used [Tianshou](https://tianshou.org/en/stable/) and for the LBM simulations we used [XLB](https://github.com/Autodesk/XLB). 

![](results/figures/model_eval.gif)



## Getting Started

Clone repository and import XLB
```console
$ git clone git@github.com:cselab/CNN-MARL_closure_model_discovery.git
$ cd CNN-MARL_closure_model_discovery
$ git clone git@github.com:Autodesk/XLB.git -b main-old
```
To disable print statements in XLB, optionally modify `XLB/src/base.py` by removing or commenting out:
```python
self.show_simulation_parameters()
...
print("Time to create the grid mask:", time.time() - start)
...
print("Time to create the local masks and normal arrays:", time.time() - start)

```

## Setup (Optional)
### 1. Burn in simulation
A burn in simulation is used to statistically stabalize the Kolmogorov flow. Run:
```console
$ cd xlb_flows
$ python run_burn_in.py
```
to run an XLB simulation of the 2D Kolmogorov flow for $T=645$ at resolution $N=2048$ for seeds $s \in \{102, 99, 33\}$. The final density and velocity fields will be used to initialize all future Kolmogorov flows. $s=102$ is used for training, $s=99$ for validation, and $s=33$ for testing.  This step is optional as we have included the resulting fields from the burn in simulation in [results/init_fields](results/init_fields).


## Model Training (Optional)
To reproduce the training of the policies run:
```console
cd closure_discovery
````
Global:
```console
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=..:../XLB python rl_klmgrv_ppo.py --max_epoch 100 --steup "glob" --num_agents 1 --ent_coef -0.005 --seed 66
```
Local:
```console
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=..:../XLB python rl_klmgrv_ppo.py --max_epoch 300 --steup "loc" --num_agents 128 --lr_decay 1 --seed 44
```
Interpolating:
```console
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=..:../XLB python rl_klmgrv_ppo.py --max_epoch 200 --steup "interp" --num_agents 16 --seed 33
```
This steps are optional as we provided the weights of the trained models in [results/weights](results/weights).


## Model Testing
### 1. Create references for testing
To create all the reference solutions used for testing the ClosureRL models, run:
```console
$ cd xlb_flows
$ python create_reference_runs.py
```
 It runs a BGK and a KBC simulation at the same resolution as the ClosureRL model, a BGK simulation at twice the resolution, and a BGK simulation at DNS resolution $N=2048$. This is done for all 3 test cases: Kolmogorov flow at $Re=10^4$ and $Re=10^5$ and a decaying flow at $10^4$. All simulations run for $T=227$, and the velocity file is saved every $32$ steps for the CGS resolution.


### 2. Evaluate ClosureRL models
-To evaluate the trained models run:
```console
$ cd closure_discovery
$ python create_test_runs.py
```
This will evaluate all 3 models (global, interpolating and local) on the 3 test cases and store the velocity fields used to create the figures.

### 3. Create figures
- The testing figures are plottet in [results/analysis.ipynb](results/analysis.ipynb).
- The action interpretation figures are plottet in [closure_discovery/action_analysis.ipynb](closure_discovery/action_analysis.ipynb).

### 4. Measure speedup
To measure the speedup and create the speedup plot, run:
```console
$ cd xlb_flows
$ python measure_speedup.py
```


## Acknowledgements
- the RL part is from [Tianshou](https://tianshou.org/en/stable/).
- The LBM part is from [XLB](https://github.com/Autodesk/XLB).