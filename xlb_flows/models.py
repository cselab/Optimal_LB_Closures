import jax.numpy as jnp
from jax import jit, grad, vmap
from functools import partial
import torch
import jax
from XLB.src.base import LBMBase
jax.config.update("jax_spmd_mode", 'allow_all')


class ClosureRLSim(LBMBase):
    """
    ClosureRL LBM simulation.

    Samples overrelaxation parameter from a policy 
    """
    def __init__(self, actor, flax_params, **kwargs):
        self.actor = actor
        self.flax_params = flax_params
        self.C_u = kwargs.get("C_u")
        self.N = kwargs.get("nx")
        super().__init__(**kwargs)

    #@partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f):
        """
        BGK collision step for lattice.

        The collision step is where the main physics of the LBM is applied. In the BGK approximation, 
        the distribution function is relaxed towards the equilibrium distribution function.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        rho, u = self.update_macroscopic(f)
        feq = self.equilibrium(rho, u, cast_output=False)
        fneq = f - feq
        P_neq = self.momentum_flux(fneq)
        state = jnp.concatenate((rho, u/self.C_u, P_neq), axis=-1)
        alpha = self.actor.apply(self.flax_params, state).reshape(self.N, self.N, 1)
        #TODO :scale alpha to range here
        alpha = alpha * 0.005
        #
        fout = f - (self.omega*(1+alpha))  * fneq
        if self.force is not None:
            fout = self.apply_force(fout, feq, rho, u)
        return self.precisionPolicy.cast_to_output(fout)