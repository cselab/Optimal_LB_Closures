import os
import sys
import json
import jax
import numpy as np
import jax_cfd.base as cfd
from jax import tree_util
from xlb_flows.utils import *
from XLB.src.utils import *
from XLB.src.models import BGKSim, KBCSim, AdvectionDiffusionBGK

np.random.seed(42)
jax.config.update('jax_enable_x64', True)


# base class initialized from density and velocity field
class Kolmogorov_flow(BGKSim):

    def show_simulation_parameters(self):
        pass

    def __init__(self, **kwargs):
        self.u0_path = kwargs.get("u0_path")
        self.rho0_path = kwargs.get("rho0_path")
        self.C_u = kwargs.get("C_u")
        self.vel_ref = kwargs.get("vel_ref")
        self.chi = kwargs.get("chi")
        self.alpha = kwargs.get("alpha")
        self.yy = kwargs.get("yy")
        self.dx_eff = kwargs.get("dx_eff")
        super().__init__(**kwargs)

        
    def set_boundary_conditions(self):
        # no boundary conditions implying periodic BC in all directions
        return
    
    def initialize_macroscopic_fields(self):
        u = np.load(self.u0_path)*self.C_u
        rho = np.load(self.rho0_path)
        #downsample u to the desired resolution
        if(u.shape[0]/self.nx > 1):
            ux = downsample_vorticity(u[...,0], int(u.shape[0]/self.nx))[...,0]
            uy = downsample_vorticity(u[...,1], int(u.shape[0]/self.ny))[...,0]
            u = np.stack([ux, uy], axis=-1)
            rho = downsample_vorticity(rho[...,0], int(rho.shape[0]/self.nx))
        u = self.distributed_array_init(u.shape, self.precisionPolicy.output_dtype, init_val=u, sharding=self.sharding)
        rho = self.distributed_array_init(rho.shape, self.precisionPolicy.output_dtype, init_val=rho, sharding=self.sharding)
        return rho, u
 

    def initialize_populations(self, rho, u):
        omegaADE = 1.0
        kwargs = {'lattice': self.lattice,
                   'nx': self.nx,
                   'ny': self.ny,
                   'nz': self.nz, 
                   'precision': self.precision,
                   'omega': omegaADE,
                   'vel': u,
                   'print_info_rate': 0,
                   'io_rate': 0}
        
        ADE = AdvectionDiffusionBGK(**kwargs)
        ADE.initialize_macroscopic_fields = self.initialize_macroscopic_fields
        f = ADE.run(int(self.vel_ref))
        return f
    
    
    def get_force(self, u=None):
        force = np.zeros((self.nx, self.ny, 2))
        force[..., 0] = self.chi * np.sin(self.yy)
        if u is not None:
            force = force - self.alpha * u
        return self.precisionPolicy.cast_to_output(force)


    def output_data(self, **kwargs):
            u = np.array(kwargs["u"])/self.C_u
            timestep = kwargs["timestep"]

            #save velocity field as npy
            #fname = os.path.basename(__file__)
            #fname = os.path.splitext(fname)[0]
            #fname = "velocity_" + fname
            #fname = fname + "_" + f"s{self.seed}"
            #fname = fname + "_" + str(timestep).zfill(6)
            #np.save(fname, u)

            v = vorticity_2d(u, self.dx_eff)
            save_image(timestep, v, "vort_")

            ## compute and save the energy spectrum
            #_, energy_spectrum = energy_spectrum_2d(u)
            #save_npy(timestep, energy_spectrum, "spec_")


#decaying flow
class Decaying_flow(Kolmogorov_flow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_force(self, u=None):
        pass


# burn in simulation is initialized randomly and saves only the final fields
class Burn_in_Kolmogorov_flow(Kolmogorov_flow):

    def __init__(self, **kwargs):
        self.v_max = kwargs.get("v_max")
        self.seed = kwargs.get("seed")
        self.endTime = kwargs.get("endTime")
        super().__init__(**kwargs)

    
    # initalized field by sampling big fields of size "size" and downsampling to the actual desired size "n"
    # this allows to have same initial fields for differnec grid resolutions
    def random_initial_fields(self, vmax, n_small=128, size=2048, seed=42):
        rho = np.ones((n_small,n_small,1))
        grid = cfd.grids.Grid((size, size), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
        v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(seed), grid, vmax)
        ux = tree_util.tree_flatten(v0)[0][0]
        uy = tree_util.tree_flatten(v0)[0][1]
        if(size//n_small > 1):
            ux = downsample_vorticity(ux, int(size//n_small))[...,0]
            uy = downsample_vorticity(uy, int(size//n_small))[...,0]
        return ux, uy, rho
    

    def initialize_macroscopic_fields(self):
        ux, uy, rho = self.random_initial_fields(self.v_max, n_small=self.nx, seed=self.seed)
        rho = self.distributed_array_init(rho.shape, self.precisionPolicy.output_dtype, init_val=1.0, sharding=self.sharding)
        u = np.stack([ux, uy], axis=-1)
        u = self.distributed_array_init(u.shape, self.precisionPolicy.output_dtype, init_val=u, sharding=self.sharding)
        return rho, u
    

    def output_data(self, **kwargs):
        u = np.array(kwargs["u"])/self.C_u
        rho = np.array(kwargs["rho"])
        timestep = kwargs["timestep"]

        if timestep == 0:
            self.kin_list = [] # list that stores kinetic energy values 
            self.ens_list = [] # list that stores enstrophy values
            self.diss_list = [] # list that stores dissipation values
            self.ein_list = [] # list that stores energy injection values
            v = vorticity_2d(u, self.dx_eff)
            self.scale = np.max(np.abs(v)) * 0.4

        if timestep == self.endTime:
            #save u as npy
            fname = os.path.basename(__file__)
            fname = os.path.splitext(fname)[0]
            fname = "velocity_" + fname
            fname = fname + "_" + str(timestep).zfill(6)
            fname = fname + "_" + f"s{self.seed}"
            np.save(fname, u)

            #save rho as npy
            fname = os.path.basename(__file__)
            fname = os.path.splitext(fname)[0]
            fname = "density_" + fname
            fname = fname + "_" + str(timestep).zfill(6)
            fname = fname + "_" + f"s{self.seed}"
            np.save(fname, rho)


# Create Kolmogorov_flow_KBC by inheriting from Kolmogorov_flow and KBCSim
class Kolmogorov_flow_KBC(KBCSim, Kolmogorov_flow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# Create Kolmogorov_flow_KBC by inheriting from Kolmogorov_flow and KBCSim
class Decaying_flow_KBC(KBCSim, Decaying_flow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
