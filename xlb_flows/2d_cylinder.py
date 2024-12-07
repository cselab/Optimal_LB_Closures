"""
This script conducts a 2D simulation of flow around a cylinder using the lattice Boltzmann method (LBM). This is a classic problem in fluid dynamics and is often used to examine the behavior of fluid flow over a bluff body.

In this example you'll be introduced to the following concepts:

1. Lattice: A D2Q9 lattice is used, which is a two-dimensional lattice model with nine discrete velocity directions. This type of lattice allows for a precise representation of fluid flow in two dimensions.

2. Boundary Conditions: The script implements several types of boundary conditions:

    BounceBackHalfway: This condition is applied to the cylinder surface, simulating a no-slip condition where the fluid at the cylinder surface has zero velocity.
    ExtrapolationOutflow: This condition is applied at the outlet (right boundary), where the fluid is allowed to exit the simulation domain freely.
    Regularized: This condition is applied at the inlet (left boundary) and models the inflow of fluid into the domain with a specified velocity profile. Another Regularized condition is used for the stationary top and bottom walls.
3. Velocity Profile: The script uses a Poiseuille flow profile for the inlet velocity. This is a parabolic profile commonly seen in pipe flow.

4. Drag and lift calculation: The script computes the lift and drag on the cylinder, which are important quantities in fluid dynamics and aerodynamics.

5. Visualization: The simulation outputs data in VTK format for visualization. It also generates images of the velocity field. The data can be visualized using software like ParaView.

# To run type:
nohup python3 examples/CFD/cylinder2d.py > logfile.log &
"""
import os
import sys
import json
import jax
from time import time
from jax import config
import numpy as np
import jax.numpy as jnp

from XLB.src.utils import *
from XLB.src.boundary_conditions import *
from XLB.src.models import BGKSim
from XLB.src.lattice import LatticeD2Q9
from xlb_flows.utils import vorticity_2d

# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
jax.config.update('jax_enable_x64', True)

class Cylinder(BGKSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        # Define the cylinder surface
        coord = np.array([(i, j) for i in range(self.nx) for j in range(self.ny)])
        xx, yy = coord[:, 0], coord[:, 1]
        cx, cy = 2.*diam, 2.*diam
        cylinder = (xx - cx)**2 + (yy-cy)**2 <= (diam/2.)**2
        cylinder = coord[cylinder]
        implicit_distance = np.reshape((xx - cx)**2 + (yy-cy)**2 - (diam/2.)**2, (self.nx, self.ny))
        self.BCs.append(InterpolatedBounceBackBouzidi(tuple(cylinder.T), implicit_distance, self.gridInfo, self.precisionPolicy))

        # Outflow BC
        outlet = self.boundingBoxIndices['right']
        rho_outlet = np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        self.BCs.append(ExtrapolationOutflow(tuple(outlet.T), self.gridInfo, self.precisionPolicy))
        # self.BCs.append(ZouHe(tuple(outlet.T), self.gridInfo, self.precisionPolicy, 'pressure', rho_outlet))

        # Inlet BC
        inlet = self.boundingBoxIndices['left']
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)
        yy_inlet = yy.reshape(self.nx, self.ny)[tuple(inlet.T)]
        vel_inlet[:, 0] = poiseuille_profile(yy_inlet,
                                             yy_inlet.min(),
                                             yy_inlet.max()-yy_inlet.min(), 3.0 / 2.0 * prescribed_vel)
        self.BCs.append(Regularized(tuple(inlet.T), self.gridInfo, self.precisionPolicy, 'velocity', vel_inlet))

        # No-slip BC for top and bottom
        wall = np.concatenate([self.boundingBoxIndices['top'], self.boundingBoxIndices['bottom']])
        vel_wall = np.zeros(wall.shape, dtype=self.precisionPolicy.compute_dtype)
        self.BCs.append(Regularized(tuple(wall.T), self.gridInfo, self.precisionPolicy, 'velocity', vel_wall))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using bounce-back)
        rho = np.array(kwargs["rho"][..., 1:-1, :])
        u = np.array(kwargs["u"][..., 1:-1, :])
        timestep = kwargs["timestep"]
        u_prev = kwargs["u_prev"][..., 1:-1, :]

        if timestep == 0:
            self.lift_list = [] # list that stores lift coefficient values
            self.drag_list = [] # list that stores drag coefficient values
            self.time_list = [] # list that stores time values

        else:
            # compute lift and drag over the cyliner
            cylinder = self.BCs[0]
            boundary_force = cylinder.momentum_exchange_force(kwargs['f_poststreaming'], kwargs['f_postcollision'])
            boundary_force = np.sum(np.array(boundary_force), axis=0)
            drag = boundary_force[0]
            lift = boundary_force[1]
            cd = 2. * drag / (prescribed_vel ** 2 * diam)
            cl = 2. * lift / (prescribed_vel ** 2 * diam)

            #compute vorticity field
            v = vorticity_2d(u)

            u_old = np.linalg.norm(u_prev, axis=2)
            u_new = np.linalg.norm(u, axis=2)
            err = np.sum(np.abs(u_old - u_new))
            
            self.lift_list.append(cl)
            self.drag_list.append(cd)
            self.time_list.append(timestep)

            #print('error= {:07.6f}, CL = {:07.6f}, CD = {:07.6f}'.format(err, cl, cd))
            save_image(timestep, v, "vort_")
            #save_npy(timestep, v ,"vort_")
            #save_npy(timestep, rho[..., 0], "rho_")


# Helper function to specify a parabolic poiseuille profile
poiseuille_profile  = lambda x,x0,d,umax: np.maximum(0.,4.*umax/(d**2)*((x-x0)*d-(x-x0)**2))

if __name__ == '__main__':

    # takes lambda as command line argument and checks if it is valid
    # if no argument is given, default value of 1 is used
    if len(sys.argv) == 1:
        lamb = 1.0
    else:
        lamb = float(sys.argv[1])

    # check if lamb is smaller than 10 and larger than 0
    if lamb < 0 or lamb > 10:
        raise ValueError("Lambda must be between 0 and 10")
    
    
    # solely dimensionless description
    Re = 100.0 # fixed Reynolds number
    scale_factor = lamb # scaling factor to refine resolution if accuracy and stability needs to be improved, set to 4 to get original results

    diam = int(0.2 * Re * scale_factor) # diameter of the cylinder -> corresponds to non-dimensionalized length l
    prescribed_vel = 0.012 / scale_factor # prescribed velocity at the inlet
    r_x = 22 # ratio of domain length in x-direction to length of the cylinder
    r_y = 4.1 # ratio of domain length in y-direction to length of the cylinder
    nx = int(r_x*diam)
    ny = int(r_y*diam)
    visc = prescribed_vel * diam / Re # viscosity
    omega = 1.0 / (3. * visc + 0.5) # relaxation rate, i.e. 1/tau = omega
    
    # other specifications
    precision = 'f32/f32'
    CL_list, CD_list = [], []
    result_dict = {}  
    lattice = LatticeD2Q9(precision)

    # characteristic time
    T = 50 # non-dimensionalized time -> needs to be sufficiently large to reach steady state
    N_prints = 30*T # number of prints -> needs to be sufficiently large to produce smooth moovie
    tc = prescribed_vel/diam
    niter_max = int(T//tc) # number of iterations
    
    # set specifications as kwargs
    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': precision,
        'io_rate': int(niter_max // N_prints),
        'print_info_rate': int(10*(niter_max // N_prints)),
        'return_fpost': True    # Need to retain fpost-collision for computation of lift and drag
    }

    #run simulation
    # if it doenst exist create a folder to store the data
    # the folders name should be "re<value of Re>_lamb<value of lamb>"
    if not os.path.exists(f"re{int(Re)}_lamb{int(lamb)}"):
        os.makedirs(f"re{int(Re)}_lamb{int(lamb)}")
    os.chdir(f"re{int(Re)}_lamb{int(lamb)}")

    os.system('rm -rf ./*.vtk && rm -rf ./*.png, && rm -rf ./*.npy')
    sim = Cylinder(**kwargs)
    sim.run(niter_max)

    #save lift and drag coefficients to array
    CL_list.append(sim.lift_list)
    CD_list.append(sim.drag_list)
    result_dict['CL'] = CL_list
    result_dict['CD'] = CD_list
    with open(f'data_re{int(Re)}_lamb{int(lamb)}.json', 'w') as fp:
        json.dump(result_dict, fp)
