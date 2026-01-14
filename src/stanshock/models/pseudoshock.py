from __future__ import annotations
import numpy as np

from scipy.integrate import odeint

from stanshock.models.boundary_layer import SkinFriction
from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array
from stanshock.system.base import RightHandSide
from stanshock.system.geometry import Geometry

class Pseudoshock(RightHandSide):
    """
    This function computes the pressure profile found in a pseudoshock, according to the analysis performed by Fievet et. al (2018).
    """

    def __init__(
        self,
        wall_temperature=None,
        skin_friction_coefficient: SkinFriction | None = None,
        parameters: dict[str, float] | None = None,
    ) -> None:
        self.sf_array = []
        self.se_array = []
        self.us = []
        self.t_ps = []
        self.L_scale = []
        self.sigma_ss = []

        self.skin_friction_coefficient = skin_friction_coefficient
        if self.skin_friction_coefficient is None:
            self.skin_friction_coefficient = SkinFriction()  # initialize the function
        self.wall_temperature = wall_temperature
        if self.wall_temperature is None:
            pass
            #Implement some check here. If adiabatic wall, we can just set T_rat=  (T/T_wall) = 1 + (gamma - 1)/2 * M**2

        if parameters is None:
            parameters = {}


        self.Ap = parameters.get("Ap",1.64)  
        self.Bp = parameters.get("Bp",25.9)       

        self.Cp = parameters.get("Cp", 2.5)
        self.Dp = parameters.get("Dp", 3.5)
        self.Ep = parameters.get("Ep", 0)

        self.k_ref = parameters.get("k_ref", 135.5)

        self.Beta_p = parameters.get("Beta_p", 1.1)
        self.M_ref = parameters.get("M_ref", 1.90)
        self.Alpha_p = parameters.get("Alpha_p", 1.064)
        self.sigma = parameters.get("sigma", 0.769521)




    def reduce_sequential(self,arr):
        runs = np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)
        out = []
        for run in runs:
            out.append(run[0])
        return np.array(out, dtype=int)

    def get_constants(self, M1, cf0, sigma = None):
        if sigma is None:
            sigma = self.sigma
        kappa = self.Bp * (1 - np.tanh(self.Cp * (M1 - self.M_ref)))
        norm_int = ((self.Ap + 1) ** (kappa + 1) - self.Ap ** (kappa + 1)) / (kappa + 1)
        k_c = (self.k_ref * cf0**self.Alpha_p * sigma**self.Beta_p) / norm_int
        cf_model = self.Ep + (self.Dp * cf0)        
        return kappa, k_c, cf_model


    def M_Ar_Derivatives(self, y, x, gamma, q1, kappa, k_c, cf_model, Dh_func):
        Dh = Dh_func(x)
        M2, AcA, p = y
        q = gamma * M2 * p / 2
        dp_dx = (q / Dh) * k_c * (self.Ap + (q / q1))**kappa

        mult = -M2 * (1 + ((gamma - 1)/2)*M2)
        term1M2 = (2 / (gamma * M2))*(dp_dx / p)
        term2M2 = (4 * cf_model / Dh)
        dM2_dx = mult * ((term1M2 + term2M2) / AcA)

        term1AcA = (1 - M2*(1 - gamma*(1 - AcA))) / (gamma * M2 * AcA)
        term2AcA = ((1 + (gamma - 1)*M2) / (2 * AcA)) * (4*cf_model/Dh)
        dAcA_dx = ((term1AcA)*(dp_dx/p) + term2AcA)*AcA

        return np.array([dM2_dx, dAcA_dx, dp_dx])


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

        a = physics.get_sound_speed(state)[idx]


        p2p1a = np.maximum(p[1:], p[:-1]) / np.minimum(p[1:], p[:-1])
        dp_norm = np.sign((p[1:] - p[:-1])) #compute d(p_R - p_L) / x (implicitly assumed dx > 0)

        uL = u[:-1]; uR = u[1:]
        aL = a[:-1]; aR = a[1:]

        maskLP = (np.abs(p2p1a) >= 1.05) & (dp_norm == 1)
        l_entropy_condn = (uL - aL) > (uR - aR) #gamma- char.
        l_states = maskLP & l_entropy_condn
        L = np.nonzero(l_states)[0]   # left-facing shock candidates
        if L.size == 0: 
            return None
        else:
            if L.size >1:
                L = self.reduce_sequential(L.copy())
            i_best = L[0]
            final_idx = i_best - 4

        
        shock_idx = int(np.clip(final_idx, 0, len(dp_norm) - 1))  
     
        x0 = geometry.xc[idx][shock_idx]
        p1 = p[shock_idx]
        a1 = a[shock_idx]
        u1 = u[shock_idx]

        gamma1 = physics.get_gamma(state)[idx][shock_idx]
        r1 = state.density[idx][shock_idx]
        mu1 = physics.get_mu(state)[idx][shock_idx]
        cp1 = physics.get_cp(state)[idx][shock_idx]
        T1 = state.temperature[idx][shock_idx]

        def Dh_func(x_val):
            return geometry.hydraulic_diameter(time, x_val + x0)
    
        if not self.L_scale:
            self.L_scale = float(Dh_func(0))

        d_ind = np.rint(self.L_scale / (2 * geometry.dx)).astype(int)
        i0 = max(shock_idx - d_ind, 0)
        i1 = min(shock_idx + d_ind, len(p))
        p_wind = p[i0 : i1]
        p_rat = np.max(p_wind / p1)
        T_rat =  T1 / self.wall_temperature

        us = u1 - a1*np.sqrt((p_rat * (gamma1 + 1)/(2*gamma1)) + (gamma1 - 1)/(2 * gamma1))
        M1 = u1/a1
        M1_rel = M1 - (us / a1)
        if np.abs(us) >= 600 or M1_rel <= 1.3:
            return None
        T0 = T1 * (1 + 0.5*(gamma1 - 1)*M1_rel**2)
        q1 = gamma1 * M1_rel**2 * p1 / 2
        Re0 = (u1*Dh_func(x0)*r1)/mu1
        cf0 = float(self.skin_friction_coefficient(Re0, M1, T_rat)) / 3
        R_s = (cp1 * (gamma1 - 1)) / gamma1

        

        kappa, k_c, cf_model = self.get_constants(M1_rel, cf0)
        args = (gamma1, q1, kappa, k_c, cf_model, Dh_func)

        y0 =  [float(M1_rel**2), 1.000, float(p1)]
        self.sf_array.append(shock_idx)
        self.us.append(us)
        self.t_ps.append(time)
        return args, y0, shock_idx, us, T0, R_s
    

    def pseudoshock_solver(self, x: Array, args, y0):
        y_out = odeint(self.M_Ar_Derivatives, y0, x, args)
        M_out_temp = np.sqrt(y_out[:,0])
        AcA_temp = y_out[:,1]
        P_out_temp = y_out[:,2]

        search_inds = np.where(np.gradient(AcA_temp) > 0)[0]
        if search_inds.size > 0:
            abs_diff = np.abs(AcA_temp[search_inds] - 1.0)
            end_ind = search_inds[np.argmin(abs_diff)] + 1
            M_out = M_out_temp[:end_ind]
            AcA = AcA_temp[:end_ind]
            P_out = P_out_temp[:end_ind]
        else:
            M_out, AcA, P_out = M_out_temp, AcA_temp, P_out_temp
        return M_out, AcA, P_out
    
    def get_source_terms(self, time: float, state: FluidState, physics: FluidPhysics, geometry: Geometry):
        shock_info = self.get_shock_location(time, state, physics, geometry)

        if shock_info is None:
            return None
        args, y0, s_idx, us, T0, R_s = shock_info

        idx = geometry.idx_cells
        x = geometry.xc[idx]
        
        M_out, AcA, P_out = self.pseudoshock_solver(x, args, y0)
        n_ps = len(P_out)
        if n_ps < 2:
            return None
        
        r_idx = np.clip(s_idx + n_ps, 0, len(x)-1)
        ssMask = np.arange(s_idx, r_idx)
        l_idx = r_idx - s_idx

        x_ps = x[ssMask]
        M_out = M_out[:l_idx]
        P_out = P_out[:l_idx]
        AcA = AcA[:l_idx]

        #Get Stanshock state; masked:
        p_ss = state.pressure[idx]
        r_ss = state.density[idx]
        u_ss = state.velocity[idx]
        q_ss = (0.5 * r_ss * u_ss**2)[ssMask]

        Dh = geometry.hydraulic_diameter(time, x_ps)

        gamma, q1, kappa, k_c, cf_model, Dh_func = args
        q = gamma * M_out**2 * P_out / 2


        K_PS = k_c * (self.Ap + (q /   q1))**kappa

        i_exp = np.argmax(K_PS)
        mask1 = np.arange(0, i_exp+1)

        dp_dx_SS = q_ss * K_PS / Dh
        dp_dx_SS[mask1] = np.zeros_like(mask1)

        mask2 = dp_dx_SS < 0
        dp_dx_SS[mask2] = 0
        
        sigma_ps = (p_ss[r_idx]) / P_out[-1]
        self.sigma_ss.append(sigma_ps)

        rhs = np.zeros((len(x), 2 + physics.n_scalars))
        rhs[ssMask, 0] = dp_dx_SS
        shear_mask = rhs[:,0] != 0

        #WIP
        # if np.abs(error) <= 0.1:
        # elif error > 0.1: #if backpressure greater than predicted
        #     #have to shift pseduoshock profile upstream.
        #     pass
        # else: #backpressure less than predicted
        #     pass
        return (rhs, shear_mask)
    






