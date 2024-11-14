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
from tqdm import tqdm
import matplotlib.pyplot as plt



from XLB.src.utils import *
from XLB.src.boundary_conditions import *
from XLB.src.lattice import LatticeD2Q9


#create kwargs for dataset creating e.g. for supervised learning
def get_kwargs(u0_path,
                rho0_path,
                T_wish=18,
                lamb=1,
                Re=1000,
                n=4,
                upsilon=1,
                seed=42,
                print_rate=32,
                measure_speedup=False):

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
    if measure_speedup:
         N_prints=1
    else:
        N_prints = m_prime//(lamb*print_rate)  #print factor for superviese = 1
    io_rate = m_prime / N_prints

    print(rf"Re={Re}, m_prime={endTime}, T={T}, omega={omega}")
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
    'v_max': v_max,
    'seed': seed,
    'endTime': endTime,
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


@partial(jit)
def vorticity_2d(u, dx=1.0):
    u_x_dy, u_y_dx = jnp.gradient(u[..., 0], dx, axis=1), jnp.gradient(u[..., 1], dx, axis=0)
    return u_y_dx - u_x_dy


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



# computes energy spectrum of a 2d velocity field
def energy_spectrum_2d(u):

    n_x, n_y, _ = u.shape
    nn = max(n_x,n_y)
    uh = np.fft.ifft2(u[...,0],norm='backward')
    vh = np.fft.ifft2(u[...,1], norm='backward')
    uhat = np.stack([uh, vh], axis=-1)
    E_k = np.sum(np.conj(uhat)*uhat, axis=-1).real/2.
    freq = np.fft.fftfreq(nn, d=1/nn)
    kx, ky = np.meshgrid(freq, freq)
    knorm = np.sqrt(kx**2 + ky**2)
    ks = np.arange(1,int(nn/2))
    Ek = np.zeros(len(ks))
    for i in range(0,int(nn/2)-1):
        k = ks[i]
        mask = (knorm > k-0.5) & (knorm <= k+0.5)
        Ek[i] = np.mean(E_k[mask])

    return ks, 2*np.pi*Ek



@partial(jit, static_argnums=(1, 2))
def downsample_vorticity(field, factor, method='bicubic'):
    """
    Downsample a JAX array by a factor of `factor` along each axis.

    Parameters
    ----------
    field : jax.numpy.ndarray
        The input vector field to be downsampled. This should be a 3D or 4D JAX array where the last dimension is 2 or 3 (vector components).
    factor : int
        The factor by which to downsample the field. The dimensions of the field will be divided by this factor.
    method : str, optional
        The method to use for downsampling. Default is 'bicubic'.

    Returns
    -------
    jax.numpy.ndarray
        The downsampled field.
    """
    if factor == 1:
        return field
    else:
        new_shape = tuple(dim // factor for dim in field.shape)
        downsampled_components = []
        resized = resize(field, new_shape, method=method)
        downsampled_components.append(resized)

        return jnp.stack(downsampled_components, axis=-1)


def save_image(timestep, fld, prefix=None, scale=None, show_time=False, tc = 1.0):
    """
    Save an image of a field at a given timestep.

    Parameters
    ----------
    timestep : int
        The timestep at which the field is being saved.
    fld : jax.numpy.ndarray
        The field to be saved. This should be a 2D or 3D JAX array. If the field is 3D, the magnitude of the field will be calculated and saved.
    prefix : str, optional
        A prefix to be added to the filename. The filename will be the name of the main script file by default.

    Returns
    -------
    None

    Notes
    -----
    This function saves the field as an image in the PNG format. The filename is based on the name of the main script file, the provided prefix, and the timestep number.
    If the field is 3D, the magnitude of the field is calculated and saved. The image is saved with the 'nipy_spectral' colormap and the origin set to 'lower'.
    """
    #fname = os.path.basename(__main__.__file__)
    script_filename = 'klmgrv.py'
    fname = os.path.basename(script_filename)
    fname = os.path.splitext(fname)[0]
    if prefix is not None:
        fname = prefix + fname
    fname = fname + "_" + str(timestep).zfill(6)

    if len(fld.shape) > 3:
        raise ValueError("The input field should be 2D!")
    elif len(fld.shape) == 3:
        fld = np.sqrt(fld[..., 0] ** 2 + fld[..., 1] ** 2)

    plt.clf()
    #plt.imsave(fname + '.png', fld.T, cmap=cm.jet, vmin=-0.001, vmax=0.001, origin='lower')
    if scale is None:
        plt.imsave(fname + '.png', fld.T, cmap=sn.cm.icefire, vmin=-10, vmax=10, origin='lower')
        #plt.imsave(fname + '.png', fld.T, cmap=seaborn.cm.icefire, origin='lower')
    else:
        
        if show_time == True:
            # translate timestep to non-dimensional time
            non_dim_time = int(round(timestep * tc,0))
            fig, ax = plt.subplots()
            im = ax.imshow(fld.T, cmap=sn.cm.icefire, vmin=-scale, vmax=scale, origin='lower')
            bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="black", alpha=0.7)
            ax.text(0.1, 0.1, f"T = {non_dim_time}", ha='left', va='bottom', fontsize=33, color='white', weight='bold', bbox=bbox_props, transform=ax.transAxes)
            ax.axis('off')
            plt.savefig(fname+'.png', bbox_inches='tight', pad_inches= 0., transparent=True)
            plt.close()
        
        else:
            plt.imsave(fname + '.png', fld.T, cmap=sn.cm.icefire, vmin=-scale, vmax=scale, origin='lower')


def create_and_navigate_to(folder_name):
    # Create and navigate to folder_name
    os.makedirs(folder_name, exist_ok=True)
    with os.scandir(folder_name):
        os.chdir(folder_name)

@partial(jit)
def vorticity_2d(u, dx=1.0):
    u_x_dy, u_y_dx = jnp.gradient(u[..., 0], dx, axis=1), jnp.gradient(u[..., 1], dx, axis=0)
    return u_y_dx - u_x_dy