from __future__ import annotations

import numpy as np
from scipy.optimize import root

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array
from stanshock.system.base import RightHandSide
from stanshock.models.boundary_layer import SkinFriction
from stanshock.system.geometry import Geometry
from scipy.integrate import solve_ivp
"""
Key elements of pseudoshock solution:
Cf
M
p
q
Dh
xShock

"""
class Pseudoshock(RightHandSide):

    """
    This function computes the pressure profile found in a pseudoshock, according to the analysis performed by Fievet et. al (2018).

    """
    def __init__(
            self,
            hydraulic_diameter,
            characteristic_length,
            skin_friction_coefficient=None,
    ) -> None:
        self.hydraulic_diameter = hydraulic_diameter
        self.characteristic_length = characteristic_length
        self.skin_friction_coefficient = skin_friction_coefficient
        self.xShock_array = [0]
        self.sInd_array = [0]
        if  self.skin_friction_coefficient is None:
            self.skin_friction_coefficient = SkinFriction()  # initialize the function

    def get_shock_location(self, _time: float,  state: FluidState, physics: FluidPhysics, geometry: Geometry
    ) -> None:
        p = state.pressure
        x = geometry.x
        dpdx = np.gradient(p,x)
        shock_inds = np.where(dpdx > 0)[0]
        sInd = shock_inds[np.argmin(np.abs(x[shock_inds] - self.xShock_array[-1]))]
        xShock = x[sInd]
        self.xShock_array.append(xShock)
        self.sInd_array.append(sInd)
        
    def get_pseudoshock_profile(self, _time: float, state: FluidState, physics: FluidPhysics, geometry: Geometry
    ) -> Array:
        x = geometry.x
        dp = np.zeros_like(x)
        sIndc = self.sInd_array[-1]
        M1 = state.velocity[sIndc] / state.sound_speed[sIndc]
        if M1 < 1.3:
            return dp
        else:
            parameters = 1.0e+02 * np.array([0.0347, 0.0000170, 0.0151, 0.259, 1.355, 0.0164, 0.0176, 0.01428, 0.0106])
            Dp = parameters[0]
            Ep = parameters[1]
            Cp = parameters[2]
            Bp = parameters[3]
            k_ref = parameters[4]
            Ap = parameters[5]
            Beta_p = parameters[6]
            M_ref = parameters[7]
            Alpha_p = parameters[8]
            p1 = state.pressure[sIndc]
            g1 = state.gamma[sIndc]
            Re0 = (state.velocity[sIndc]*self.characteristic_length*state.density[sIndc]) / state.viscosity[sIndc]
            cf0 = SkinFriction(Re0)     
            q1 = p1 * M1**2 * p1 /2
            kappa = Bp*(1 - np.tanh(Cp*(M1 - M_ref)))
            sigma = 0.769520803266958
            cf_model = Ep + (Dp*cf0)
            k0 = k_ref*sigma**Beta_p
            norm_int = ((Ap + 1)**(kappa+1) - Ap**(kappa+1))/(kappa+1)
            Dh = self.hydraulic_diameter(x[sIndc],_time)
            def M_Ar_Derivatives(x,y):
                    M2, Aratio, p = y
                    q = g1*M2*p/2
                    k_q = k0*((Ap + q/q1)**kappa) / norm_int
                    dP_dx = p*k_q/Dh*cf0**Alpha_p * g1 * M2/2
                    dM2 =  - M2*((1 + (g1 -1)/2 * M2)* ((2/g1/M2/Aratio)*(dP_dx/p)) + 4*cf_model/Dh *1/Aratio)
                    dAratio = Aratio*((1 - M2* (1 - g1 * (1 - Aratio))) / (g1 * M2 * Aratio) * (dP_dx/p) + (1 + (g1 - 1) * M2) / (2 * Aratio) * 4 * cf_model / Dh)
                    return np.array([dM2, dAratio, dP_dx],dtype=float).flatten()
            y0 = [float(M1**2), 1.000, p1]
            pseudo_solve = solve_ivp(fun=lambda x, y: M_Ar_Derivatives(x,y), t_span=[0,self.x[-1]],y0=y0,method='RK45',t_eval=self.x)
            i_end = len(x) - sIndc
            dp[sIndc:] = (pseudo_solve.y[2])[:i_end] - state.pressure[sIndc:]
            return dp
    def source_from_primitives(
        self, _time: float, state: FluidState, physics: FluidPhysics, geometry: Geometry
        ) -> Array:
            """Boundary layer contribution to RHS."""
            if self.hydraulic_diameter is None or self.characteristic_length is None:
                msg = "Combustor improperly initialized for boundary layer terms"
                raise Exception(msg)

            rhs = np.zeros((*state.shape, 3 + physics.n_scalars))
            #Pseudoshock pressure addition
            xShock = self.get_shock_location(_time, state, physics, geometry)
            p_pseudo = self.get_pseudoshock_profile(_time,  state, physics, geometry)
            tau = 1e-3
            rhs[:, 1] = p_pseudo*np.exp(-_time/tau)
            return rhs