import numpy as np
import sys
import os
import seaborn as sn
from jax.experimental.multihost_utils import process_allgather
import gymnasium as gym
from gymnasium import spaces
from functools import partial
import jax.numpy as jnp
from jax import jit

sys.path.append(os.path.abspath(os.path.expanduser('~/XLB')))
from src.utils import *
from src.boundary_conditions import *
from src.lattice import LatticeD2Q9


# creates kwargs for burn in simulation
def get_burn_in_kwargs(lamb=16,
                        desired_time=40,
                        Re=1000, n=4,
                        vel_ref=0.1*(1/np.sqrt(3)),
                        seed=42):
    
    twopi = 2.0 * np.pi
    N = int(128*lamb) # domain length in LB units
    dx = twopi/N # space conversion factor 
    dx_eff = twopi/128 # effective space conversion factor for the 128x128 simulation 
    l = N/(twopi*n) # characteristic length in LB units
    visc = vel_ref * l / Re # viscosity
    omega = 1.0 / (3. * visc + 0.5) # relaxation rate
    C_u = vel_ref/n # velocity transformation factor

    # compute burn in time s.t. the simulation reaches the desired time
    # and the outputs are always at the same timestep independent of the resolution
    kappa = np.ceil(desired_time/(twopi*vel_ref))
    actual_time = kappa*twopi*vel_ref 
    m = N*actual_time*14/twopi # number of conventional time steps
    tau = 14*C_u # time conversion factor
    m_prime = m/tau # number of LB time steps
    tc = vel_ref/l # time conversion factor for non-dimensional time

    T = int(np.ceil(m_prime*tc)) # non-dimensional time
    N_prints = m_prime//(lamb*64) # number of prints -> needs to be sufficiently large to produce smooth moovie
    endTime = int(np.ceil(m_prime))
    io_rate = endTime
    print(f"m = {m}, m_prime = {m_prime},
           end time = {endTime} steps, T={T},
             io_rate = {io_rate}, Number of outputs = {endTime//io_rate + 2}")

    coord = np.array([(i, j) for i in range(N) for j in range(N)])
    xx, yy = coord[:, 0], coord[:, 1]
    kx, ky = n * twopi / N, n * twopi / N
    xx = xx.reshape((N,N)) * kx
    yy = yy.reshape((N,N)) * ky
    #force parameter
    chi = C_u**2 * twopi/N
    alpha = 0.1 * C_u * twopi/N
    v_max = C_u * 7.0

    precision = "f64/f64"
    lattice = LatticeD2Q9(precision)
    # definedirectory
    checkpoint_dir = os.path.abspath("./checkpoints")
    
    kwargs = {
    'lattice': lattice,
    'omega': omega,
    'nx': N,
    'ny': N,
    'nz': 0,
    'precision': precision,
    'io_rate': int(io_rate),
    'print_info_rate': int(10*io_rate),
    'checkpoint_rate': endTime,
    'checkpoint_dir': checkpoint_dir,
    'restore_checkpoint': False,
    'C_u' : C_u,
    'vel_ref' : vel_ref,
    'chi' : chi,
    'alpha' : alpha,
    'yy' : yy,
    'dx_eff': dx_eff,
    'v_max': v_max,
    'seed': seed,
    }

    return kwargs, endTime, T, N



#create kwargs for dataset creating e.g. for supervised learning
def get_kwargs4(u0_path,
                rho0_path,
                T_wish=18,
                lamb=1,
                Re=1000,
                n=4,
                upsilon=1,
                seed=42):

    twopi = 2.0 * np.pi
    vel_ref= upsilon*0.1*(1/np.sqrt(3))
    N = int(128*lamb) # domain length in LB units
    dx_eff = twopi/128 # effective space conversion factor, used in vorticity computation which is always performed on the downsampled field
    l = N/(twopi*n) # characteristic length in LB units
    visc = vel_ref * l / Re # viscosity
    omega = 1.0 / (3. * visc + 0.5) # relaxation rate
    C_u = vel_ref/n # velocity transformation factor
    kappa = np.ceil(T_wish*l/(lamb*vel_ref))
    T = kappa * (lamb*vel_ref/l)
    m_prime = kappa * lamb
    endTime = int(np.ceil(m_prime))
    io_rate = lamb
    print(rf"Re={Re}, m_prime={endTime}, T={T}, omega={omega}")
    coord = np.array([(i, j) for i in range(N) for j in range(N)])
    xx, yy = coord[:, 0], coord[:, 1]
    kx, ky = n * twopi / N, n * twopi / N
    xx = xx.reshape((N,N)) * kx
    yy = yy.reshape((N,N)) * ky
    #force parameter
    chi = C_u**2 * twopi/N
    alpha = 0.1 * C_u * twopi/N
    #v_max = C_u * 7.0
    precision = "f64/f64"
    lattice = LatticeD2Q9(precision)
    
    kwargs = {
    'lattice': lattice,
    'omega': omega,
    'nx': N,
    'ny': N,
    'nz': 0,
    'precision': precision,
    'io_rate': int(io_rate),
    'print_info_rate': int(10*io_rate),
    'downsampling_factor': lamb,
    #'checkpoint_rate': endTime,
    #'checkpoint_dir': checkpoint_dir,
    #'restore_checkpoint': False,
    'u0_path' : u0_path,
    'rho0_path' : rho0_path,
    'C_u' : C_u,
    'vel_ref' : vel_ref,
    'chi' : chi,
    'alpha' : alpha,
    'yy' : yy,
    'dx_eff': dx_eff,
    'seed': seed
    }
    
    return kwargs, endTime, T, N


#less io prints so used for evaluation and visualization
def get_kwargs5(u0_path,
                rho0_path,
                T_wish=18,
                lamb=1,
                Re=1000,
                n=4,
                upsilon=1,
                seed=42):

    twopi = 2.0 * np.pi
    vel_ref= upsilon*0.1*(1/np.sqrt(3))
    N = int(128*lamb) # domain length in LB units
    dx_eff = twopi/128 # effective space conversion factor, used in vorticity computation which is always performed on the downsampled field
    l = N/(twopi*n) # characteristic length in LB units
    visc = vel_ref * l / Re # viscosity
    omega = 1.0 / (3. * visc + 0.5) # relaxation rate
    C_u = vel_ref/n # velocity transformation factor
    kappa = np.ceil(T_wish*l/(lamb*vel_ref))
    T = kappa * (lamb*vel_ref/l)
    m_prime = kappa * lamb
    endTime = int(np.ceil(m_prime))
    N_prints = m_prime//(lamb*32)  #N_prints = m_prime//(lamb*8)
    io_rate = m_prime / N_prints
    print(rf"Re={Re}, m_prime={endTime}, T={T}, omega={omega}, N_prints={N_prints}, io_rate={io_rate}")
    coord = np.array([(i, j) for i in range(N) for j in range(N)])
    xx, yy = coord[:, 0], coord[:, 1]
    kx, ky = n * twopi / N, n * twopi / N
    xx = xx.reshape((N,N)) * kx
    yy = yy.reshape((N,N)) * ky
    #force parameter
    chi = C_u**2 * twopi/N
    alpha = 0.1 * C_u * twopi/N
    #v_max = C_u * 7.0

    precision = "f64/f64"
    lattice = LatticeD2Q9(precision)
    
    kwargs = {
    'lattice': lattice,
    'omega': omega,
    'nx': N,
    'ny': N,
    'nz': 0,
    'precision': precision,
    'io_rate': int(io_rate),
    'print_info_rate': int(10*io_rate),
    'downsampling_factor': lamb,
    #'checkpoint_rate': endTime,
    #'checkpoint_dir': checkpoint_dir,
    #'restore_checkpoint': False,
    'u0_path' : u0_path,
    'rho0_path' : rho0_path,
    'C_u' : C_u,
    'vel_ref' : vel_ref,
    'chi' : chi,
    'alpha' : alpha,
    'xx' : xx,
    'yy' : yy,
    'dx_eff': dx_eff,
    'seed': seed
    }
    
    return kwargs, endTime, T, N


def get_velocity(f, sim):
        rho, u = sim.update_macroscopic(f)
        rho = downsample_field(rho, sim.downsamplingFactor)
        u = downsample_field(u, sim.downsamplingFactor)
        rho = process_allgather(rho)
        u = process_allgather(u)/sim.C_u
        return rho, u


def get_vorticity(f, sim):
        rho, u = sim.update_macroscopic(f)
        rho = downsample_field(rho, sim.downsamplingFactor)
        u = downsample_field(u, sim.downsamplingFactor)
        rho = process_allgather(rho)
        u = process_allgather(u)/sim.C_u
        v = vorticity_2d(u, sim.dx_eff)
        return v


def get_moments(f, sim):
        rho, u = sim.update_macroscopic(f)
        fneq = f - sim.equilibrium(rho, u, cast_output=False)
        P_neq = sim.momentum_flux(fneq)
        #rho = downsample_field(rho, sim.downsamplingFactor)
        #u = downsample_field(u, sim.downsamplingFactor)
        #P_neq = downsample_field(P_neq, sim.downsamplingFactor)
        rho = process_allgather(rho)
        u = process_allgather(u)/sim.C_u
        P_neq = process_allgather(P_neq)
        return rho, u, P_neq


@partial(jit, inline=True)
def update_macroscopic(f, c):
    """
    This function computes the macroscopic variables (density and velocity) based on the 
    distribution functions (f).

    The density is computed as the sum of the distribution functions over all lattice directions. 
    The velocity is computed as the dot product of the distribution functions and the lattice 
    velocities, divided by the density.

    Parameters
    ----------
    f: jax.numpy.ndarray
        The distribution functions.

    Returns
    -------
    rho: jax.numpy.ndarray
        Computed density.
    u: jax.numpy.ndarray
        Computed velocity.
    """
    rho =jnp.sum(f, axis=-1, keepdims=True)
    u = jnp.dot(f, c.T) / rho

    return rho, u

@partial(jit, inline=True)
def momentum_flux(fneq, cc):
    """
    This function computes the momentum flux, which is the product of the non-equilibrium 
    distribution functions (fneq) and the lattice moments (cc).

    The momentum flux is used in the computation of the stress tensor in the Lattice Boltzmann 
    Method (LBM).

    Parameters
    ----------
    fneq: jax.numpy.ndarray
        The non-equilibrium distribution functions.

    Returns
    -------
    jax.numpy.ndarray
        The computed momentum flux.
    """
    return jnp.dot(fneq, cc)

@partial(jit, inline=True)
def equilibrium(rho, u, c, w):
    """
    This function computes the equilibrium distribution function in the Lattice Boltzmann Method.
    The equilibrium distribution function is a function of the macroscopic density and velocity.
    The function first casts the density and velocity to the compute precision if the cast_output flag is True.
    The function finally casts the equilibrium distribution function to the output precision if the cast_output 
    flag is True.
    Parameters
    ----------
    rho: jax.numpy.ndarray
        The macroscopic density.
    u: jax.numpy.ndarray
        The macroscopic velocity.
    cast_output: bool, optional
        A flag indicating whether to cast the density, velocity, and equilibrium distribution function to the 
        compute and output precisions. Default is True.
    Returns
    -------
    feq: ja.numpy.ndarray
        The equilibrium distribution function.
    """

    # Cast c to compute precision so that XLA call FXX matmul, 
    # which is faster (it is faster in some older versions of JAX, newer versions are smart enough to do this automatically)
    cu = 3.0 * jnp.dot(u, c)
    usqr = 1.5 * jnp.sum(jnp.square(u), axis=-1, keepdims=True)
    feq = rho * w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)

    return feq