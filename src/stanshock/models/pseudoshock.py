from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from stanshock.models.boundary_layer import SkinFriction
from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array
from stanshock.system.base import RightHandSide
from stanshock.system.geometry import Geometry
from matplotlib import pyplot as plt
import copy



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
        self.sf_array = []
        self.se_array = []
        self.time_array = []

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
        from scipy.signal import savgol_filter

# Apply Savitzky-Golay filter to smooth pressure data
        smoothed_p = savgol_filter(p, window_length=5, polyorder=3)
        dpdx = np.gradient(smoothed_p, x)
        dpdx = dpdx / max(dpdx)
        d2pdx2 = np.gradient(dpdx, x) / max(np.gradient(dpdx, x))
        average_dpdx = np.mean(dpdx)

        threshold = 5 * average_dpdx
        
        shock_foot_inds = np.where((dpdx > threshold) & (d2pdx2 > 0))[0]
        shock_foot_ind = shock_foot_inds[0] if len(shock_foot_inds) > 0 else np.argmax(dpdx)
        if shock_foot_ind != 0:
            shock_foot_ind = shock_foot_ind - 2 #buffer for preventing wrong inlet conditions
        shock_end_inds_after_foot = np.where((dpdx <= 0) & (d2pdx2 >= 0) & (np.arange(len(dpdx)) > shock_foot_ind))[0]
        shock_end_ind = shock_end_inds_after_foot[0]

        self.sf_array.append(shock_foot_ind)
        self.se_array.append(shock_end_ind)
        self.time_array.append(time)

            
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
    
    def get_preshock_properties(self, time: float, state: FluidState,
        physics: FluidPhysics,
        geometry: Geometry,
        sInd: int):
        x = geometry.x
        M1 = state.velocity[sInd] / physics.get_sound_speed(state)[sInd]
        Dh = geometry.d_outer(x[sInd], time)
        p1 = state.pressure[sInd]
        g1 = state.gamma[sInd]
        q1 = g1 * M1**2 * p1 / 2
        Re0 = (state.velocity[sInd] * Dh * state.density[sInd]) / physics.get_mu(state)[sInd]
        cf0 = self.skin_friction_coefficient(Re0)
        kappa = self.Bp * (1 - np.tanh(self.Cp * (M1 - self.M_ref)))
        sigma = 0.769520803266958
        cf_model = self.Ep + (self.Dp * cf0)
        k0 = self.k_ref * sigma**self.Beta_p
        norm_int = ((self.Ap + 1) ** (kappa + 1) - self.Ap ** (kappa + 1)) / (kappa + 1)
        args = (g1, q1, k0, kappa, norm_int, cf0, cf_model, Dh)
        y0 =  [float(M1**2), 1.000, p1]
        return args, y0
    
    def pseudoshock_solver(self,
        time: float,
        state: FluidState,
        physics: FluidPhysics,
        geometry: Geometry
    ): #returns p_pseudo_profile, predicted p2, and pseudoshock length
            def reattachment(_x,y,*_):
                Aratio = y[1]
                return (Aratio - 1)
            reattachment.terminal = False
            reattachment.direction = 1
            args, y0 = self.get_preshock_properties(time, state, physics, geometry, self.sf_array[-1])
            pseudo_solve = solve_ivp(fun=self.M_Ar_Derivatives, t_span=[geometry.x[0], geometry.x[-1]],
            y0=y0,
            method="RK45",
            t_eval=geometry.x,
            events=reattachment,
            args=args,
            )
            L_ps = pseudo_solve.t_events[0][0] #pseudoshock length
            p2 = pseudo_solve.y_events[0][0][2] #predicted final pressure rise
            p_pseudo = pseudo_solve.y[2] #pseudoshock profile
            return p_pseudo, p2, L_ps

    
    def location_optimizer(self, time: float, state: FluidState,
        physics: FluidPhysics,
        geometry: Geometry):
            self.get_shock_location(time, state, geometry)
            args, y0 = self.get_preshock_properties(time, state, physics, geometry, self.sf_array[-1])
            if np.sqrt(y0[0]) < 1.3:
                dp_arr = np.zeros_like(state.pressure)
                return dp_arr
            else:
                x = geometry.x
                dp_arr = np.copy(state.pressure)
                p_pseudo, p2, L_ps = self.pseudoshock_solver(time, state, physics, geometry)
                error_array = [1, 0]
                while error_array[-1] < error_array[-2] and len(error_array) < 12: #resolves until shock location is closest match to existing profile
                    x2 = L_ps + x[self.sf_array[-1]]
                    w_mask = (x > (x2 - L_ps)) & (x < (x2 + L_ps))
                    p_curr_windowed = state.pressure[w_mask]
                    i_min_p = np.argmin(np.abs(p2 - p_curr_windowed))
                    global_indices = np.where(w_mask)[0]
                    i_end_p = global_indices[i_min_p]
                    error_array.append(np.abs(state.pressure[i_end_p] - p2))
                    new_xShock = x[i_end_p] - L_ps
                    sInd_new = np.argmin(np.abs(new_xShock - x))
                    self.sf_array[-1] = sInd_new
                    self.se_array[-1] = i_end_p
                    p_pseudo, p2, L_ps = self.pseudoshock_solver(time, state, physics, geometry)
                # print("It took %0.0f iterations to find the best solution." %(len(error_array) - 2))
                Re_prior = (state.velocity * args[7] * state.density) / physics.get_mu(state)
                cf_prior = self.skin_friction_coefficient(Re_prior)
                shear_prior = (cf_prior * 0.5 * state.density * state.velocity**2) * np.sign(state.velocity)
                sfc = self.sf_array[-1]
                sec = self.se_array[-1]
                tau = 1e-4
                dp_arr = np.zeros_like(state.pressure)
                dp_arr[sfc:sec] = (p_pseudo[:(sec-sfc)] - state.pressure[sfc:sec])*(1 - np.exp(-time/tau)) - shear_prior[sfc:sec]
                return dp_arr


    def source_from_primitives(
        self,
        time: float,
        state: FluidState,
        physics: FluidPhysics,
        geometry: Geometry,
    ) -> Array:
        """Pseudoshock contribution to RHS."""
        rhs = np.zeros((*state.shape, 3 + physics.n_scalars))
        deltap = self.location_optimizer(time, state, physics, geometry)
        Dh = geometry.d_outer(geometry.x, time)
        rhs[:, 1] =  -4.0 / Dh * deltap
        return rhs
