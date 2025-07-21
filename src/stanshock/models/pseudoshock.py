from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
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
        self.sf_array = [np.nan]
        self.se_array = [np.nan]
        self.us = []
        self.time_array = [np.nan]
        self.t_ps = 0

        self.skin_friction_coefficient = skin_friction_coefficient
        if self.skin_friction_coefficient is None:
            self.skin_friction_coefficient = SkinFriction()  # initialize the function

        # Use all default values if parameters not provided
        if parameters is None:
            parameters = {}

        # Extract values from parameters dictionary if provided
        # self.Ap = parameters.get("Ap", 1.64)       # 0.0043
        # self.Bp = parameters.get("Bp", 25.9)       # 2.47

        # self.Cp = parameters.get("Cp", 1.51)       # 157.3
        # self.Dp = parameters.get("Dp", 3.47)       # 0.195
        # self.Ep = parameters.get("Ep", 0.00170)    # 0.000

        # self.k_ref = parameters.get("k_ref", 135.5)  # 98.2

        # self.Beta_p = parameters.get("Beta_p", 1.76)  # 0.0
        # self.M_ref = parameters.get("M_ref", 1.428)   # 2.07
        # self.Alpha_p = parameters.get("Alpha_p", 1.06)  # 0.0043
        self.Ap = parameters.get("Ap", 0.0043) #1.64
        self.Bp = parameters.get("Bp", 2.47) #25.9

        self.Cp = parameters.get("Cp", 157.3) #1.51
        self.Dp = parameters.get("Dp", 0.195) #3.47
        self.Ep = parameters.get("Ep", 0.000) #0.00170  

        self.k_ref = parameters.get("k_ref", 98.2) #135.5
        
        self.Beta_p = parameters.get("Beta_p", 0.0) #1.76
        self.M_ref = parameters.get("M_ref", 2.07) #1.428
        self.Alpha_p = parameters.get("Alpha_p", 0.0043) #1.06"

    def get_shock_location(self, time: float, state: FluidState, physics: FluidPhysics, geometry: Geometry) -> None:
        p = state.pressure
        x = geometry.x
        x_iso_start, x_iso_end = geometry.regions["isolator"]
        xf = x[:-1]
        p2p1a = p[1:] / p[:-1]
        sf_options = np.where((p2p1a > 1.15) & (xf >= x_iso_start) & (xf <= x_iso_end))[0] 
        if len(sf_options) > 0: # and np.isnan(self.sf_array[-1]):
            max_jump_idx = sf_options[np.argmax(p2p1a[sf_options])]
            # p1 = p[sf_options[0]]
            # p2 = p[sf_options[0] + 1]
            self.sf_array.append(max_jump_idx -2)
            self.time_array.append(time)


        # elif len(sf_options) > 0 and ~np.isnan(self.sf_array[-1]):
        #     closest_idx = sf_options[np.argmin(np.abs(sf_options - self.sf_array[-1]))]
        #     self.sf_array.append(closest_idx)
        #     self.time_array.append(time)
        else:
            self.sf_array.append(np.nan)

        
        

        
         

        # if len(self.time_array) == 1000:
        #     plt.figure()
        #     mask = ~np.isnan(self.sf_array)
        #     sf = np.array(self.sf_array)[mask]
        #     sf = sf.astype(np.int64)
        #     plt.scatter(np.array(self.time_array)[mask], x[sf])
        #     plt.show()
        #     print('blah')

            
    def M_Ar_Derivatives(self, x, y, g1, q1, k0, kappa, norm_int, cf0, cf_model, Dh_func):
        Dh = Dh_func(x)
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
        def Dh_func(x_val):
            return geometry.d_outer(x_val +x[sInd], time)
        Dh0 = geometry.d_outer(x[sInd], time)


        # kappa = self.Bp * (1 - np.tanh(self.Cp * (M1 - self.M_ref)))

        if len(self.sf_array) >= 2 and not np.isnan(self.sf_array[-2:]).any():
            u_s = (x[self.sf_array[-1]] - x[self.sf_array[-2]]) / (self.time_array[-1] - self.time_array[-2])
            if np.abs(u_s) <= state.velocity[sInd]:
                M1 = np.abs(state.velocity[sInd] - u_s) / physics.get_sound_speed(state)[sInd]
                self.us.append(u_s)
                # print(f"Shock Velocity u_s = {u_s:.2f} m/s")
        # kappa = self.Bp * (1 - np.tanh(self.Cp * (self.M_ref - M1)))
        kappa = self.Bp * (1 - np.tanh(self.Cp * (M1 - self.M_ref)))
        sigma = 0.769521
        cf_model = self.Ep + (self.Dp * cf0)
        k0 = self.k_ref * sigma**self.Beta_p
        norm_int = ((self.Ap + 1) ** (kappa + 1) - self.Ap ** (kappa + 1)) / (kappa + 1)
        

        p1 = state.pressure[sInd]
        g1 = state.gamma[sInd]
        q1 = g1 * M1**2 * p1 / 2
        Re0 = (state.velocity[sInd] * Dh0 * state.density[sInd]) / physics.get_mu(state)[sInd]
        cf0 = self.skin_friction_coefficient(Re0)
        args = (g1, q1, k0, kappa, norm_int, cf0, cf_model, Dh_func)
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
            pseudo_solve = solve_ivp(fun=self.M_Ar_Derivatives, t_span=[geometry.x[0],geometry.x[-1]],
            y0=y0,
            method="RK45",
            t_eval=geometry.x,
            events=reattachment,
            args=args,
            )
            L_ps = pseudo_solve.t_events[0][0] #pseudoshock length
            p2 = pseudo_solve.y_events[0][0][2] #predicted final pressure rise
            p_pseudo = pseudo_solve.y[2] #pseudoshock profile
            x_eval = pseudo_solve.t
            p_pseudo[pseudo_solve.t >= L_ps] = p2
            return p_pseudo, x_eval, p2, L_ps

    
    def location_optimizer(self, time: float, state: FluidState,
        physics: FluidPhysics,
        geometry: Geometry):
        self.get_shock_location(time, state, physics, geometry)
        dp_arr = np.zeros_like(state.pressure)

        if np.isnan(self.sf_array[-1]): 
            return dp_arr

        _, y0 = self.get_preshock_properties(time, state, physics, geometry, self.sf_array[-1])
        if np.sqrt(y0[0]) < 1.3:
            return dp_arr
        
        else:
            x = geometry.x
            if self.t_ps == 0: self.t_ps = time
            p_pseudo, x_pseudo, p2, L_ps = self.pseudoshock_solver(time, state, physics, geometry)
            x2 = x[self.sf_array[-1]] + L_ps
            # sec_temp = np.argmin(np.abs(x - x2))
            # self.se_array.append(sec_temp)
            error_array = [1, 0]
            while error_array[-1] < error_array[-2] and len(error_array) < 5: #resolves until shock location is closest match to existing profile
                w_mask = (x > (x2 - 0.5*L_ps)) & (x < (x2 + 0.5*L_ps))
                p_curr_windowed = state.pressure[w_mask]
                if np.any(p_curr_windowed):
                    i_min_p = np.argmin(np.abs(p2 - p_curr_windowed))
                    global_indices = np.where(w_mask)[0]
                    i_end_p = global_indices[i_min_p]
                    error_array.append(np.abs(state.pressure[i_end_p] - p2))
                    new_xShock = x[i_end_p] - L_ps
                    sInd_new = np.argmin(np.abs(new_xShock - x))
                    self.sf_array[-1] = sInd_new
                    self.se_array.append(i_end_p)
                    p_pseudo, x_pseudo, p2, L_ps = self.pseudoshock_solver(time, state, physics, geometry)
                else:
                    return dp_arr

            # x1 = x[self.sf_array[-1]]
            # L_flat = np.abs(x_iso_end - x_iso_start) - L_ps
            # x_flat_mp = L_ps + (L_flat/2)
            # x2 = x1 + x_flat_mp
            # L_range = L_ps + L_flat
            # curr_mask = (x > (x2 - L_range)) & (x <= (x2 + L_range))
            # x_pseudo_ref = x_pseudo + x1

            # ps_mask = (x_pseudo_ref > (x2 - L_range)) & (x_pseudo_ref < (x2 + L_range))
            # x_pseudo_windowed = x_pseudo_ref[ps_mask]
            # p_pseudo_windowed = p_pseudo[ps_mask]
            # p_pseudo_interp = np.interp(x[curr_mask], x_pseudo_windowed, p_pseudo_windowed)
            # i_min_p = np.argmin(np.abs(p_pseudo_interp - state.pressure[curr_mask]))
            # L_ps_total = x_pseudo_windowed[i_min_p] - x1
            # global_indices = np.where(curr_mask)[0]
            # i_end_p = global_indices[i_min_p]
            # self.se_array.append(i_end_p)
                # new_xShock = x[i_end_p] - L_ps_total
                # sInd_new = np.argmin(np.abs(new_xShock - x))


                
                # error_array.append(np.abs(state.pressure[i_end_p] - p2))
                # p_pseudo, x_pseudo, p2, L_ps = self.pseudoshock_solver(time, state, physics, geometry)


            # if len(self.time_array) == 20:
            plt.figure()
            plt.plot(x, state.pressure / state.pressure[0], label=r"p_{c}")
            plt.plot((x_pseudo + x[self.sf_array[-1]]), p_pseudo / state.pressure[0], label=r'p_{pseudo}')
            plt.axvline(x2, linestyle='dashed',c='k',label=r'L_{ps}')
            # plt.plot(x[curr_mask], p_curr_windowed,c='r', label='windowed pressure')
            # plt.axvline(new_xShock, c='g',label='new xshock')
            plt.legend()
            plt.xlabel('x [m]')
            plt.ylabel(r"$p / p_1$")
            plt.show()
            print('blah')

            sfc = self.sf_array[-1]
            if len(self.time_array) % 50 == 0: print(f"x_s = {x[sfc]:.2f} | u_s = {self.us[-1]:.2e} m/s")
            sec = self.se_array[-1]
            Re_prior = (state.velocity * geometry.d_outer(geometry.x, time) * state.density) / physics.get_mu(state)
            cf_prior = self.skin_friction_coefficient(Re_prior)
            shear_prior = (cf_prior * 0.5 * state.density * state.velocity**2) * np.sign(state.velocity)
            # tau = 10*L_ps_total / state.velocity[self.sf_array[-1]] #
            tau = 1e-7
            dp_arr[sfc:sec] = (p_pseudo[:(sec-sfc)]- state.pressure[sfc:sec] - shear_prior[sfc:sec])  *(1 - np.exp(-(time-self.t_ps)/tau))  
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
        
        if np.any(deltap != 0.0):
            Dh = geometry.d_outer(geometry.x, time)
            rhs[:, 1] =  - 4.0 / Dh * deltap
            # rhs = deltap
        return rhs
