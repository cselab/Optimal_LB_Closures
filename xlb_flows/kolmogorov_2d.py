import os
import sys
import json
import jax
import numpy as np
from termcolor import colored

#sys.path.append(os.path.abspath(os.path.expanduser('~/XLB')))
#from src.utils import *
#from src.models import BGKSim, AdvectionDiffusionBGK

from xlb_flows.utils import *
from XLB.src.utils import *
from XLB.src.models import BGKSim, AdvectionDiffusionBGK

np.random.seed(42)
jax.config.update('jax_enable_x64', True)


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
        kwargs = {'lattice': self.lattice, 'nx': self.nx, 'ny': self.ny, 'nz': self.nz,  'precision': self.precision, 'omega': omegaADE, 'vel': u, 'print_info_rate': 0, 'io_rate': 0}
        ADE = AdvectionDiffusionBGK(**kwargs)
        ADE.initialize_macroscopic_fields = self.initialize_macroscopic_fields
        #print("Initializing the distribution functions using the specified macroscopic fields....")
        f = ADE.run(int(self.vel_ref))
        return f
    
    
    def get_force(self, u=None):
        # define the external force
        force = np.zeros((self.nx, self.ny, 2))
        force[..., 0] = self.chi * np.sin(self.yy) # force in x-direction
        # apply friction damping
        if u is not None:
            force = force - self.alpha * u
        return self.precisionPolicy.cast_to_output(force)


    def output_data(self, **kwargs):
        """
        u = np.array(kwargs["u"])/self.C_u
        timestep = kwargs["timestep"]

        if timestep == 0:
            self.kin_list = [] # list that stores kinetic energy values 
            self.ens_list = [] # list that stores enstrophy values
            v = vorticity_2d(u, self.dx_eff)
            self.scale = np.max(np.abs(v)) * 0.4
    
        # compute vorticity
        v = vorticity_2d(u, self.dx_eff)
        # compute the total kinetic energy
        t_kin = kinetic_energy_2d(u)
        t_kin = np.sum(np.array(t_kin), axis=0) # just for io convenience transform jax array to numpy array
        self.kin_list.append(t_kin)
        #compute total enstrophy
        ens = enstrophy_2d(v)
        ens = np.sum(np.array(ens), axis=0)
        self.ens_list.append(ens)
        # save field
        if self.scale is None:
            save_image(timestep, v, "vort_")
        else:
            #save_image(timestep, v, "vort_", self.scale, show_time=True, tc=tc)
            save_image(timestep, v, "vort_", self.scale)

        #save_npy(timestep, v, "npy_vort_")

        ## compute and save the energy spectrum
        #_, energy_spectrum = energy_spectrum_2d(u)
        #save_npy(timestep, energy_spectrum, "spec_")
        """
        return 0


class Decaying_flow(BGKSim):
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
        kwargs = {'lattice': self.lattice, 'nx': self.nx, 'ny': self.ny, 'nz': self.nz,  'precision': self.precision, 'omega': omegaADE, 'vel': u, 'print_info_rate': 0, 'io_rate': 0}
        ADE = AdvectionDiffusionBGK(**kwargs)
        ADE.initialize_macroscopic_fields = self.initialize_macroscopic_fields
        #print("Initializing the distribution functions using the specified macroscopic fields....")
        f = ADE.run(int(self.vel_ref))
        return f


    def output_data(self, **kwargs):
        """
        u = np.array(kwargs["u"])/self.C_u
        timestep = kwargs["timestep"]

        if timestep == 0:
            self.kin_list = [] # list that stores kinetic energy values 
            self.ens_list = [] # list that stores enstrophy values
            v = vorticity_2d(u, self.dx_eff)
            self.scale = np.max(np.abs(v)) * 0.4
    
        # compute vorticity
        v = vorticity_2d(u, self.dx_eff)
        # compute the total kinetic energy
        t_kin = kinetic_energy_2d(u)
        t_kin = np.sum(np.array(t_kin), axis=0) # just for io convenience transform jax array to numpy array
        self.kin_list.append(t_kin)
        #compute total enstrophy
        ens = enstrophy_2d(v)
        ens = np.sum(np.array(ens), axis=0)
        self.ens_list.append(ens)
        # save field
        if self.scale is None:
            save_image(timestep, v, "vort_")
        else:
            #save_image(timestep, v, "vort_", self.scale, show_time=True, tc=tc)
            save_image(timestep, v, "vort_", self.scale)

        #save_npy(timestep, v, "npy_vort_")

        ## compute and save the energy spectrum
        #_, energy_spectrum = energy_spectrum_2d(u)
        #save_npy(timestep, energy_spectrum, "spec_")
        """
        return 0
