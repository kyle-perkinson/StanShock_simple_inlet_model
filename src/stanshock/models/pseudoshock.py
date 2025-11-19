from __future__ import annotations
from matplotlib import pyplot as plt
import numpy as np

from scipy.integrate import odeint


from stanshock.models.boundary_layer import SkinFriction
from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array
from stanshock.system.base import RightHandSide
from stanshock.system.geometry import Geometry
import pandas as pd


class Pseudoshock(RightHandSide):
    """
    This function computes the pressure profile found in a pseudoshock, according to the analysis performed by Fievet et. al (2018).
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
        self.norm_int = 0

        self.skin_friction_coefficient = skin_friction_coefficient
        if self.skin_friction_coefficient is None:
            self.skin_friction_coefficient = SkinFriction()  # initialize the function

        # Use all default values if parameters not provided
        if parameters is None:
            parameters = {}

        # Extract values from parameters dictionary if provided
        self.Ap = parameters.get("Ap", 1.64)       # 0.0043
        self.Bp = parameters.get("Bp", 25.9)       # 2.47

        self.Cp = parameters.get("Cp", 1.660)       # 157.3
        self.Dp = parameters.get("Dp", 3.544)       # 0.195
        self.Ep = parameters.get("Ep", 1.69e-3)    # 0.000

        self.k_ref = parameters.get("k_ref", 137.5) # 98.2

        self.Beta_p = parameters.get("Beta_p", 1.71) # 0.0
        self.M_ref = parameters.get("M_ref", 1.424) # 2.07
        self.Alpha_p = parameters.get("Alpha_p", 1.064) # 0.0043
        self.sigma = parameters.get("sigma", 0.769521)
        self.k0 = self.k_ref * self.sigma**self.Beta_p


    def get_shock_derivatives(self,time: float, x):
            dt = time - self.t_ps[-1]
            x_pred = x[self.sf_array][-1] + self.us[-1] * dt
            u_s = self.us[-1]
            window = np.rint((1.1 * u_s * dt) / (x[1] - x[0]))
            window = int(np.clip(np.abs(window), 5, 15)*np.sign(u_s))
            bound = np.clip((self.sf_array[-1] + window), 0, len(x))
            self.x_pred.append(x_pred)
            return x_pred, bound

    def M_Ar_Derivatives(self, y, x, k1, q1, kappa, norm_int, cf0, cf_model, Dh_func):
        Dh = Dh_func(x)
        M2, Ar, p = y
        q = k1 * M2 * p / 2
        k_q = self.k0 * ((self.Ap + q / q1) ** kappa) / norm_int
        dp_dx = q * (k_q / Dh) * cf0**(self.Alpha_p)
        dM2_dx = -M2 * ((1 + M2*(k1-1)/2) * (
             (2)/(k1 * M2 * Ar) * (dp_dx / p) + (4 * cf_model)/(Dh * Ar) ) )
        dAr_dx = Ar*((1 - M2*(1 - k1*(1-Ar))) * (dp_dx / p) + (
             (1 + (k1 - 1)*M2) / (2 * Ar) * (4 * cf_model / Dh) ))
        return np.array([dM2_dx, dAr_dx, dp_dx])


    def get_shock_location(
            self,
            time: float,
            state: FluidState, 
            physics: FluidPhysics, 
            geometry: Geometry
            ) -> None:
        idx = geometry.idx_cells
        p = state.pressure[idx]
        u = state.velocity[idx]
        r = state.density[idx]
        k = physics.get_gamma(state)[idx]
        c = physics.get_sound_speed(state)[idx]
        mu = physics.get_mu(state)[idx]

        x = geometry.xc[idx]

        p2p1a = np.maximum(p[1:], p[:-1]) / np.minimum(p[1:], p[:-1])
        pR = p[1:]; pL = p[:-1]
        

        direction = np.sign((p[1:] - p[:-1]))
        us = np.full_like(p2p1a, np.nan)

        uL = u[:-1]; uR = u[1:]
        cL = c[:-1]; cR = c[1:]

        kL = k[:-1]; kR = k[1:]
        maskL = direction == 1
        maskR = direction == -1

        ent_cond_L = (uL > uR) & (uL >= cL)
        ent_cond_R = (uR > uL) & (uR >= cR)

        maskL_final = maskL & ent_cond_L
        maskR_final = maskR & ent_cond_R

        us[maskL_final] = (uL[maskL_final] - cL[maskL_final] * np.sqrt(
            1 + ((kL[maskL_final] + 1)/(2*kL[maskL_final]) * (p2p1a[maskL_final] - 1))
        ))
        us[maskR_final] = uR[maskR_final] + cR[maskR_final] * np.sqrt(
            1 + ((kR[maskR_final] + 1)/(2*kR[maskR_final]) * (p2p1a[maskR_final] - 1))
        )

        pre_idx = np.arange(len(direction))
        pre_idx[maskR] += 1
        valid = (~np.isnan(us)) & (direction != 0)
        us = us[valid]

        ut = np.where(direction==1, uL,uR)
        ct = np.where(direction==1, cL,cR)
        kt = np.where(direction==1, kL,kR)

        ut = ut[valid]; ct = ct[valid]; kt = kt[valid]
        pre_idx = pre_idx[valid]
        dir_v = direction[valid]
        M_rel = (ut - us) / ct
        p2p1t = ((2 * kt * M_rel**2) - (kt - 1)) / (kt + 1)
        
        
        p2p1a = p2p1a[valid]
        p_mask = (p2p1a >= 1.15) & (dir_v == 1)
        if not np.any(p_mask):
            return None
        du_dx = np.gradient(u, x)
        du_dx_sf = du_dx[pre_idx]
        final_mask = p_mask & (du_dx_sf < 0)
        pre_idx_masked = pre_idx[final_mask]
        if not np.any(final_mask):
            return None
        
        
        us_final = us[final_mask]
        grad_strength = np.abs(du_dx_sf[final_mask])
        imax = np.argmax(grad_strength)
        us_temp = us_final[imax]
        shock_idx = np.clip(pre_idx_masked[imax]-1,0,len(x)-1)

        def Dh_func(x_val):
            return geometry.hydraulic_diameter(time, x_val +x0)
        
        x0 = x[shock_idx]
        c1 = c[shock_idx]
        u1 = u[shock_idx]
        r1 = r[shock_idx]
        k1 = k[shock_idx]
        p1 = p[shock_idx]
        mu1 = mu[shock_idx]

        M1_rel = u1 / c1
        q1 = k1 * M1_rel**2 * p1 / 2
        Re0 = (u1 * Dh_func(x0)* r1) /mu1
        kappa = self.Bp * (1 - np.tanh(self.Cp * (M1_rel - self.M_ref)))
        norm_int = ((self.Ap + 1) ** (kappa + 1) - self.Ap ** (kappa + 1)) / (kappa + 1)
        cf0 = self.skin_friction_coefficient(Re0)
        cf_model = self.Ep + (self.Dp * cf0)
        args = (k1, q1, kappa, norm_int, cf0, cf_model, Dh_func)
        y0 =  [float(M1_rel**2), 1.000, p1]

        self.sf_array.append(shock_idx)
        self.us.append(us_temp)
        self.t_ps.append(time)
        return args, y0, shock_idx, us_temp

        # plt.figure()
        # plt.plot(x, p,c='k',label=r"t = 0 ms")
        # plt.scatter(x[pre_idx-1], p[pre_idx-1],c='b',label='Possible Shocks') 
        # plt.scatter(x[shock_idx],p[shock_idx],c='r',label='Selected')
        # plt.legend(loc='best')
        # plt.xlabel('x [m]')
        # plt.ylabel('P [Pa]')
        # plt.ylim([0,1.1*max(p)])
        # plt.tight_layout()
        # plt.show()

        
    def pseudoshock_solver(self, x: Array, args, y0):
        y_out = odeint(self.M_Ar_Derivatives, y0, x, args)
        A_ratio = y_out[:,1]
        M_out = np.sqrt(y_out[:,0])
        search_inds = np.where(np.gradient(A_ratio) > 0)[0]
        if search_inds.size > 0:
            abs_diff = np.abs(A_ratio[search_inds] - 1.0)
            end_ind = search_inds[np.argmin(abs_diff)]
            if np.abs(A_ratio[end_ind] - 1) > 0.1:
                print("Warning! Pseudoshock solver not fully resolved.")
            p2 = y_out[end_ind,2]
            M_end = M_out[end_ind]
            
            L_ps = x[end_ind]
            A_ratio[x >= L_ps] = 1.0
            M_out[x>=L_ps] = M_end
            p_pseudo = y_out[:,2]
            p_pseudo[x >= L_ps] = p2
            return p_pseudo, p2, L_ps, A_ratio, M_out
    
    def get_effective_area(self, time: float, state: FluidState, physics: FluidPhysics, geometry: Geometry):
        shock_info = self.get_shock_location(time, state, physics, geometry)
        if shock_info is None:
            return None
        args, y0, shock_idx, us_temp = shock_info
        x = geometry.xc[geometry.idx_cells]
        p_pseudo, p2, L_ps, A_ratio, M_out = self.pseudoshock_solver(x, args, y0)
        x2 = x[shock_idx] + L_ps
        i_end_p = np.argmin(np.abs(x2 - x))
        
        self.se_array.append(i_end_p)
        sfc = shock_idx
        sec = i_end_p

        if len(self.t_ps) % 50 == 0:
            print(f"x_s = {x[sfc]:.2f} | u_s = {self.us[-1]:.2e} m/s")



        #     se_arr = np.array([x[sfc], x[sec]])
        #     # plt.scatter(se_arr, np.zeros_like(se_arr), c='r')
        #     plt.plot(x[sfc:sec],(p_pseudo[:(sec-sfc)]/p1),'r',label='Fievet Model')
        #     plt.plot(x_ep,p2p1_e,'b',label='Experimental')
        #     plt.legend(loc='lower right')
        #     plt.xlabel('x [m]')
        #     plt.ylabel(r"P / $P_1$")
        #     plt.ylim([0, 4])
        #     plt.grid()
        #     plt.show()
        #     plt.close()
        if sfc == sec:
            return None
        A_actual = geometry.area(time, x)
        if np.size(A_actual) == 1:
            A_actual = np.full_like(x, float(A_actual))
        A_eff = A_actual.copy()
        A_eff[sfc:sec] = A_actual[sfc:sec] * A_ratio[:(sec-sfc)]
        # Ac = A_eff / A_actual
        # df = pd.read_csv('data/fievet_data_case4.csv')
        # xDh_f = df['xDh'].values
        # AcA_f = df['AcA'].values
        # M_f = df['M'].values
        # Prat = df['PP1_f'].values
        # Dh_array = geometry.hydraulic_diameter(time, x)
        # xDh = x / Dh_array
        # fig, ax1 = plt.subplots()
        # ax1.plot(xDh,A_ratio,c='r',label=r"$A_c$ StanShock")
        # ax1.plot(xDh_f, AcA_f,c='r',linestyle='--',label=r"$A_c$ Fievet")
        # ax1.plot(xDh, M_out, c='b', label="M StanShock")
        # ax1.plot(xDh_f, M_f,c='b',linestyle='--',label="M Fievet")
        # ax1.set_xlabel(r"$x /D_h$")
        # ax1.set_xlim([0, 10])
        # ax1.set_ylabel(r"$A_c, M$")
        # ax1.set_ylim([0, 2])
        # ax1.grid(True)

        # lines1, labels1 = ax1.get_legend_handles_labels()
        # p1 = p_pseudo[0]
        # ax2 = ax1.twinx()
        # ax2.plot(xDh, (p_pseudo/p1),c='k',label=r"$P/P_1$ StanShock")
        # ax2.plot(xDh_f, Prat, c='k',linestyle='--',label=r"$P/P_1$ Fievet")
        # ax2.set_ylabel(r"$P / P_1$")
        # # ax2.axvline(5.48, c='g')
        # ax2.set_ylim([0, 3])
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
        # plt.show()

        dlnAcA_dx = np.gradient(np.log(A_eff), x)
        return dlnAcA_dx