# Closure Discovery for Coarse Grained Partial Differential Equations using Multi-Agent Reinforcement Learning

## Description
This is the official PyTorch/tianshou implementation of the paper *Closure Discovery for Coarse Grained Partial Differential Equations using Multi-Agent Reinforcement Learning*.


This repository allows to train a multi-agent reinforcement learning (MARL) agents to discover a closure model for a multiscale system.
As a result, the agents are able to improve the accuracy of a coarse-grained simulation (CGS) significantly.
A qualitative example of the improvement is shown in the figure below. The CGS combined with the agents is referred to as CNN-MARL.



## Getting Started

Clone repository and import XLB
```console
$ git clone git@github.com:cselab/CNN-MARL_closure_model_discovery.git
$ cd CNN-MARL_closure_model_discovery
$ git clone git@github.com:Autodesk/XLB.git -b main-old
```
To turn of print statements in XLB, optinally modify `XLB/src/base.py` by removing/commenting:
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
to run a XLB simulaton for of the 2d Kolmogorov flow for $T=645$ at resolution $N=2048$ for seeds $s \in \{102, 33\}$. The final density and velocity fields are used to initialize all future kolmgorov flows. This step is optional, as we included the reulting fields of the burn in simulation in [xlb_flows/init_fields](xlb_flows/init_fields).


## Model Training (Optional)


## Model Testing
### 1. Create references for testing
To create all the reference solutions used for testing the ClosureRL models, run:
```console
$ cd xlb/flows
$ python create_reference_runs.py
```
 Namely it runs a BGK and a KBC simulation at the same resolution as the ClosureRL model, a BGK at twice the resolution, and a BGK simulation at DNS resolution $N=2048$. This is done for all 3 test cases: Kolmogorov flow at $Re=10^4$ and $Re=10^5$ and a Decaying flow at $10^4$. All simulations run for $T=227$, and the velocity filed is saved every $32$ steps for the CGS resolution.


### 2. Evaluate ClosureRL models

### 3. Create figures

### 4. Measure speedup

## Some usefull RL resources
- [Andrej Karpahty's blog post](http://karpathy.github.io/2016/05/31/rl/)
- [Jared Tumiel's blog post](https://www.alexirpan.com/2018/02/14/rl-hard.html)

- [OpenAI's Spinning up documentation](https://spinningup.openai.com/en/latest/)
- David Silver's course on RL, [recordings](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) and [slides](https://www.davidsilver.uk/teaching/)
- [Berkley DRL course](http://rail.eecs.berkeley.edu/deeprlcourse/)
- [RL in practice - tips & tricks](https://www.youtube.com/watch?v=Ikngt0_DXJg)


## Acknowledgements
