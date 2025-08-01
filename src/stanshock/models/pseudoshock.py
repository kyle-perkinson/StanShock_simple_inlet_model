from __future__ import annotations

import numpy as np
from scipy.integrate import odeint
from scipy.signal import savgol_filter

from stanshock.models.boundary_layer import SkinFriction
from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array
from stanshock.system.base import RightHandSide
from stanshock.system.geometry import Geometry


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
        self.us = []
        self.x_pred = []
        self.t_ps = []
        self.t_us = []
        self.t_start = 0

        self.skin_friction_coefficient = skin_friction_coefficient
        if self.skin_friction_coefficient is None:
            self.skin_friction_coefficient = SkinFriction()  # initialize the function

        # Use all default values if parameters not provided
        if parameters is None:
            parameters = {}

        # Extract values from parameters dictionary if provided
        self.Ap = parameters.get("Ap", 1.64)  # 0.0043
        self.Bp = parameters.get("Bp", 25.9)  # 2.47

        self.Cp = parameters.get("Cp", 1.51)  # 157.3
        self.Dp = parameters.get("Dp", 3.47)  # 0.195
        self.Ep = parameters.get("Ep", 0.00170)  # 0.000

        self.k_ref = parameters.get("k_ref", 135.5)  # 98.2

        self.Beta_p = parameters.get("Beta_p", 1.76)  # 0.0
        self.M_ref = parameters.get("M_ref", 1.428)  # 2.07
        self.Alpha_p = parameters.get("Alpha_p", 1.06)  # 0.0043
        self.sigma = parameters.get("sigma", 0.769521)
        self.k0 = self.k_ref * self.sigma**self.Beta_p

    def get_shock_derivatives(self, time: float, x):
        window_len = len(self.sf_array)

        if len(self.sf_array) % 2 == 0:
            window_len -= 1

        window_len = int(np.clip(window_len, None, 101))
        x_in = x[self.sf_array]
        x_smooth = savgol_filter(x_in, window_length=window_len, polyorder=2)
        coeffs = np.polyfit(self.t_ps[-window_len:], x_smooth[-window_len:], deg=2)
        x_s = coeffs[0] * time**2 + coeffs[1] * time + coeffs[0]
        u_s = 2 * coeffs[0] * time + coeffs[1]
        a_s = 2 * coeffs[0]
        dt = time - self.t_ps[-1]
        if np.abs(u_s) > 1e4:
            u_s = np.gradient(x_smooth)[-1] / (self.t_ps[-1] - self.t_ps[-2])

        dx = np.abs(u_s * dt)
        x_low_bound = np.clip(x_in[-1] - dx, x[0], None)
        x_high_bound = np.clip(x_in[-1] + dx, None, x[-1])
        x_pred = np.clip(
            x_in[-1] + u_s * dt + 0.5 * a_s * dt**2, x_low_bound, x_high_bound
        )
        window = np.rint((1.1 * u_s * dt) / (x[1] - x[0]))
        window = int(np.clip(np.abs(window), 5, 15) * np.sign(u_s))
        bound = np.clip((self.sf_array[-1] + window), 0, len(x))

        self.x_pred.append(x_pred)
        self.us.append(u_s)
        self.t_us.append(time)
        # if len(self.x_pred) % 50 == 0:
        #     plt.figure()
        #     plt.plot(self.t_ps, x_in,c='r')
        #     plt.plot(self.t_us, self.x_pred)
        #     plt.ylim([x[0], x[-1]])
        #     plt.show()
        return x_pred, u_s, bound

    def get_shock_location(
        self, time: float, state: FluidState, physics: FluidPhysics, geometry: Geometry
    ) -> None:
        p = state.pressure
        x = geometry.x
        x_iso_start, x_iso_end = geometry.regions["isolator"]
        xf = x[:-1]
        gamma_all = physics.get_gamma(state)
        p2p1a = np.maximum(p[1:], p[:-1]) / np.minimum(
            p[1:], p[:-1]
        )  # actual cell-by-cell pressure gradient
        M_all = state.velocity / physics.get_sound_speed(state)
        M_in = np.maximum(M_all[1:], M_all[:-1])
        g_in = np.maximum(gamma_all[1:], gamma_all[-1:])
        p2p1t = ((2 * g_in * M_in**2) - (g_in - 1)) / (
            g_in + 1
        )  # theoretical normal shock pressure gradient
        sf_options = np.where(
            (p2p1a >= p2p1t) & (xf >= x_iso_start) & (xf <= x_iso_end)
        )[0]
        if len(sf_options) > 0:
            if len(self.sf_array) >= 15:
                x_pred, u_s_mean, bound = self.get_shock_derivatives(time, x)
                search_start = min(self.sf_array[-1], bound)
                search_end = max(self.sf_array[-1], bound)
                sf_options = np.asarray(sf_options)
                valid_sf = sf_options[
                    (sf_options >= search_start) & (sf_options <= search_end)
                ]
                if len(valid_sf) > 0:
                    closest_idx = valid_sf[np.argmin(np.abs(x[valid_sf] - x_pred))]
                else:
                    closest_idx = sf_options[
                        np.argmin(np.abs(x[sf_options] - x_pred))
                    ]  # fallback

                self.sf_array.append(max(closest_idx - 2, 0))
            else:
                max_jump_idx = sf_options[np.argmax(p2p1a[sf_options])]
                self.sf_array.append(max(max_jump_idx - 2, 0))
            self.t_ps.append(time)
            return True
        return False

    def M_Ar_Derivatives(
        self, y, x, g1, q1, k0, kappa, norm_int, cf0, cf_model, Dh_func
    ):
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

    def get_preshock_properties(
        self,
        time: float,
        state: FluidState,
        physics: FluidPhysics,
        geometry: Geometry,
        sInd: int,
    ):
        x = geometry.x
        M1 = state.velocity[sInd] / physics.get_sound_speed(state)[sInd]

        def Dh_func(x_val):
            return geometry.d_outer(x_val + x[sInd], time)

        Dh0 = geometry.d_outer(x[sInd], time)
        kappa = self.Bp * (1 - np.tanh(self.Cp * (self.M_ref - M1)))
        # kappa = self.Bp * (1 - np.tanh(self.Cp * (M1 - self.M_ref))) #this may be needed if we change dictionaries. sometimes this works better.

        norm_int = ((self.Ap + 1) ** (kappa + 1) - self.Ap ** (kappa + 1)) / (kappa + 1)

        p1 = state.pressure[sInd]
        g1 = state.gamma[sInd]
        q1 = g1 * M1**2 * p1 / 2
        Re0 = (state.velocity[sInd] * Dh0 * state.density[sInd]) / physics.get_mu(
            state
        )[sInd]
        cf0 = self.skin_friction_coefficient(Re0)
        cf_model = self.Ep + (self.Dp * cf0)
        args = (g1, q1, kappa, norm_int, cf0, cf_model, Dh_func)
        y0 = [float(M1**2), 1.000, p1]
        return args, y0

    def pseudoshock_solver(
        self, time: float, state: FluidState, physics: FluidPhysics, geometry: Geometry
    ):  # returns pseudoshock profile, predicted p2, and pseudoshock length
        args, y0 = self.get_preshock_properties(
            time, state, physics, geometry, self.sf_array[-1]
        )

        y_out = odeint(self.M_Ar_Derivatives, y0, geometry.x, args)

        A_ratio = y_out[:, 1]
        search_inds = np.where(np.gradient(A_ratio) > 0)[0]
        if search_inds.size > 0:
            abs_diff = np.abs(A_ratio[search_inds] - 1.0)
            end_ind = search_inds[np.argmin(abs_diff)]
            if np.abs(A_ratio[end_ind] - 1) > 0.1:
                print("Warning! Pseudoshock solver not fully resolved.")
            p2 = y_out[end_ind, 2]
            L_ps = geometry.x[end_ind]
            p_pseudo = y_out[:, 2]
            p_pseudo[geometry.x >= L_ps] = p2
        return p_pseudo, p2, L_ps

    def location_optimizer(
        self, time: float, state: FluidState, physics: FluidPhysics, geometry: Geometry
    ):
        found_shock = self.get_shock_location(time, state, physics, geometry)
        dp_arr = np.zeros_like(state.pressure)

        if not found_shock:
            return dp_arr
        x = geometry.x
        if self.t_start == 0:
            self.t_start = time
        p_pseudo, p2, L_ps = self.pseudoshock_solver(time, state, physics, geometry)
        x2 = x[self.sf_array[-1]] + L_ps
        i_end_p = np.argmin(np.abs(x2 - x))
        if len(self.us) > 0:
            w_mask = x > x2 - 0.5 * L_ps if self.us[-1] > 0 else x < x2 + 0.5 * L_ps
            p_curr_windowed = state.pressure[w_mask]
            if np.any(p_curr_windowed):
                i_min_p = np.argmin(np.abs(p2 - p_curr_windowed))
                global_indices = np.where(w_mask)[0]
                i_end_p = global_indices[i_min_p]
                new_xShock = x[i_end_p] - L_ps
                sInd_new = np.argmin(np.abs(new_xShock - x))
                self.sf_array[-1] = sInd_new
                p_pseudo, p2, L_ps = self.pseudoshock_solver(
                    time, state, physics, geometry
                )
        self.se_array.append(i_end_p)
        sfc = self.sf_array[-1]
        if len(self.t_ps) % 500 == 0:
            print(f"x_s = {x[sfc]:.2f} | u_s = {self.us[-1]:.2e} m/s")
        sec = self.se_array[-1]
        Re_prior = (
            state.velocity * geometry.d_outer(geometry.x, time) * state.density
        ) / physics.get_mu(state)

        cf_prior = self.skin_friction_coefficient(Re_prior)
        shear_prior = (cf_prior * 0.5 * state.density * state.velocity**2) * np.sign(
            state.velocity
        )

        tau = 1e-7

        dp_arr[sfc:sec] = (
            p_pseudo[: (sec - sfc)] - state.pressure[sfc:sec] - shear_prior[sfc:sec]
        ) * (1 - np.exp(-(time - self.t_start) / tau))
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
            rhs[:, 1] = -4.0 / Dh * deltap
        return rhs
