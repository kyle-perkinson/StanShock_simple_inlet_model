from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from stanshock.models.boundary_layer import SkinFriction
from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array
from stanshock.system.base import RightHandSide
from stanshock.system.geometry import Geometry
from matplotlib import pyplot as plt


class Pseudoshock(RightHandSide):
    """
    This function computes the pressure profile found in a pseudoshock, according to the analysis performed by Fievet et. al (2018).

    Key elements of pseudoshock solution:
    Cf
    M
    p
    q
    Dh
    xShock
    """

    def __init__(
        self,
        skin_friction_coefficient: SkinFriction | None = None,
        parameters: dict[str, float] | None = None,
    ) -> None:
        self.xShock_array = []
        self.sInd_array = []

        self.skin_friction_coefficient = skin_friction_coefficient
        if self.skin_friction_coefficient is None:
            self.skin_friction_coefficient = SkinFriction()  # initialize the function

        # Use all default values if parameters not provided
        if parameters is None:
            parameters = {}

        # Extract values from parameters dictionary if provided
        self.Dp = parameters.get("Dp", 3.47)
        self.Ep = parameters.get("Ep", 0.00170)
        self.Cp = parameters.get("Cp", 1.51)
        self.Bp = parameters.get("Bp", 25.9)
        self.k_ref = parameters.get("k_ref", 135.5)
        self.Ap = parameters.get("Ap", 1.64)
        self.Beta_p = parameters.get("Beta_p", 1.76)
        self.M_ref = parameters.get("M_ref", 1.428)
        self.Alpha_p = parameters.get("Alpha_p", 1.06)

    def get_shock_location(self, time: float, state: FluidState, geometry: Geometry) -> None:
        p = state.pressure
        x = geometry.x
        dpdx = np.gradient(p, x)
        shock_inds = np.where(dpdx > 0)[0]
        if time == 0:
            sInd = shock_inds[0]
        else:
            sInd = shock_inds[np.argmin(np.abs(x[shock_inds] - self.xShock_array[-1]))]
        xShock = x[sInd]
        self.xShock_array.append(xShock)
        self.sInd_array.append(sInd)
    def M_Ar_Derivatives(self, _x, y, g1, q1, k0, kappa, norm_int, cf0, cf_model, Dh):
        M2, Aratio, p = y
        q = g1 * M2 * p / 2
        k_q = k0 * ((self.Ap + q / q1) ** kappa) / norm_int
        dP_dx = p * k_q / Dh * cf0**self.Alpha_p * g1 * M2 / 2
        dM2 = -M2 * (
            (1 + (g1 - 1) / 2 * M2) * ((2 / g1 / M2 / Aratio) * (dP_dx / p))
            + 4 * cf_model / Dh * 1 / Aratio
        )
        dAratio = Aratio * (
            (1 - M2 * (1 - g1 * (1 - Aratio))) / (g1 * M2 * Aratio) * (dP_dx / p)
            + (1 + (g1 - 1) * M2) / (2 * Aratio) * 4 * cf_model / Dh
        )
        return np.array([dM2, dAratio, dP_dx])

    def get_pseudoshock_profile(
        self,
        time: float,
        state: FluidState,
        physics: FluidPhysics,
        geometry: Geometry,
    ) -> Array:
        x = geometry.x
        dp = np.zeros_like(x)
        sIndc = self.sInd_array[-1]
        M_arr = state.velocity/physics.get_sound_speed(state)
        M1 = state.velocity[sIndc] / physics.get_sound_speed(state)[sIndc]
        if M1 < 1.3:
            return dp
        else:
            Dh = geometry.d_outer(x[sIndc], time)
            p1 = state.pressure[sIndc]
            g1 = state.gamma[sIndc]
            Re0 = (state.velocity[sIndc] * Dh * state.density[sIndc]) / physics.get_mu(
                state
            )[sIndc]
            cf0 = self.skin_friction_coefficient(Re0)
            q1 = g1 * M1**2 * p1 / 2
            kappa = self.Bp * (1 - np.tanh(self.Cp * (M1 - self.M_ref)))
            sigma = 0.769520803266958
            cf_model = self.Ep + (self.Dp * cf0)
            k0 = self.k_ref * sigma**self.Beta_p
            norm_int = ((self.Ap + 1) ** (kappa + 1) - self.Ap ** (kappa + 1)) / (kappa + 1)
            def reattachment(_x,y,*_):
                Aratio = y[1]
                # p2 = y[2]
                return (Aratio - 1)
            reattachment.terminal = False
            reattachment.direction = 1

                
            y0 = [float(M1**2), 1.000, p1]
            pseudo_solve = solve_ivp(
                fun=self.M_Ar_Derivatives,
                t_span=[x[0], x[-1]],
                y0=y0,
                method="RK45",
                # t_eval=x,
                t_eval=x,
                events=reattachment,
                args=(g1, q1, k0, kappa, norm_int, cf0, cf_model, Dh),
            )
            # if time == 0:
            #     i_end = sIndc + len(pseudo_solve.t)
            #     dp_init = np.copy(state.pressure)
            #     dp_init[sIndc:i_end] = pseudo_solve.y[2]
            #     state.pressure = dp_init
            # else:
            L_ps = pseudo_solve.t_events[0]
            x2 = L_ps + x[sIndc]
            w_mask = (x > (x2 - L_ps/2)) & (x < (x2 + L_ps/2))
            p_pseudo = pseudo_solve.y[2]
            p_pseudo_windowed = p_pseudo[w_mask]
            p_curr_windowed = state.pressure[w_mask]
            i_local_min = np.argmin(np.abs(p_pseudo_windowed - p_curr_windowed))
            global_indices = np.where(w_mask)[0]
            i_end = global_indices[i_local_min]
            # if time == 0:
            dp = np.copy(state.pressure)
            dp[sIndc:i_end] = (pseudo_solve.y[2])[:i_end-sIndc]
            # else:
            #     dp[sIndc:i_end] = (pseudo_solve.y[2])[:i_end-sIndc]  - state.pressure[sIndc:i_end]

            
            

            # plt.figure()
            # plt.plot(x,dp,label='pseudoshock')
            # plt.plot(x,state.pressure,label='existing')
            # # plt.plot(x,pseudo_solve.y[1])
            # plt.legend()
            # plt.show()
            return dp

    def source_from_primitives(
        self,
        time: float,
        state: FluidState,
        physics: FluidPhysics,
        geometry: Geometry,
    ) -> Array:
        """Pseudoshock contribution to RHS."""
        rhs = np.zeros((*state.shape, 3 + physics.n_scalars))
        # Pseudoshock pressure addition
        self.get_shock_location(time, state, geometry)
        p_pseudo = self.get_pseudoshock_profile(time, state, physics, geometry)
        tau = 1e-3
        rhs[:, 1] = p_pseudo#  * np.exp(-time / tau)
        return rhs
