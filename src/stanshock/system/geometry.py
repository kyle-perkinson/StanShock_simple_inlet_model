from __future__ import annotations

import numpy as np
from scipy import integrate

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array
from stanshock.system.base import RightHandSide
from scipy.sparse import diags


class Geometry(RightHandSide):
    def __init__(
        self,
        t: float,
        x: Array,
        h=None,
        w=None,
        d_inner=None,
        d_outer=None,
        dlnA_dt=None,
        dlnA_dx=None,
        upper_wall = None,
        lower_wall = None,
        regions: dict[str, tuple[float, float]] = None
    ) -> None:
        self.time = t
        self.x = x
        self.n = len(self.x)
        self.dx = self.x[1] - self.x[0]
        if upper_wall is not None and lower_wall is not None:
            self.lower_wall = lower_wall
            self.upper_wall = upper_wall
        self.regions = regions or {}

        self.h = h
        self.w = w
        self.d_inner = d_inner
        self.d_outer = d_outer
        self.dlnA_dt = dlnA_dt
        self.dlnA_dx = dlnA_dx

        if self.h is not None and self.w is not None:
            self.hydraulic_diameter = (
                2
                * self.h(self.x, self.time)
                * self.w
                / (self.h(self.x, self.time) + self.w)
            )
            self.characteristic_length = self.hydraulic_diameter.copy()
        elif self.d_outer is not None:
            self.hydraulic_diameter = self.d_outer(self.x, self.time)
            self.characteristic_length = self.hydraulic_diameter.copy()

            if self.d_inner is not None:
                self.hydraulic_diameter -= self.d_inner(self.x, self.time)
                self.characteristic_length = 0.5 * self.hydraulic_diameter

                noInsert = self.d_inner(self.x, self.time) == 0.0
                self.characteristic_length[noInsert] = self.hydraulic_diameter[noInsert]

        # self.integrator = integrate.ode(self.source_fast).set_integrator("lsoda")
        self.integrator = integrate.ode(self.source_fast, jac=self.source_fast_jacobian_banded).set_integrator("lsoda", lband=0, uband=0)
        # self.integrator = integrate.ode(self.source_fast, jac=None).set_integrator("lsoda", lband=0, uband=0)
        # Define global indices
        self.idx_locations = np.s_[:]
        self.idx_source_terms = np.s_[:3]

    def source(
        self,
        time: float,
        state_array: Array,
        physics: FluidPhysics,
        gamma_star: Array,
        dt: float,
    ):
        # Divide domain between explicit and implicit source terms
        idx_explicit = np.arange(self.x.shape[0])
        idx_implicit = []
        rhs = np.zeros(state_array[self.idx_locations, self.idx_source_terms].shape)
        if self.dlnA_dt is not None:
            dlnA_dt = self.dlnA_dt(self.x, time)
            idx_implicit = np.where(dlnA_dt != 0.0)[0]
            idx_explicit = np.where(dlnA_dt == 0.0)[0]
            if idx_implicit.size != 0:
                y0 = state_array[idx_implicit, self.idx_source_terms]
                args = [self.x[idx_implicit], gamma_star[idx_implicit]]
                self.integrator.set_initial_value(y0, time)
                self.integrator.set_f_params(args)
                self.integrator.set_jac_params(args)
                self.integrator.integrate(time + dt)
                rhs[idx_implicit] = (
                    self.integrator.y - state_array[idx_implicit, self.idx_source_terms]
                ) / dt

        # Add slow source terms
        state = physics.conservative_to_primitive(state_array, gamma_star)
        rhs[idx_explicit, :] = self.source_slow(time, state_array, state, idx_explicit)

        return rhs

    def source_slow(
        self, time: float, state_array: Array, state: FluidState, idx: Array
    ) -> Array:
        """Area change contributions to RHS."""
        rhs = np.zeros((idx.shape[0], 3))

        if self.dlnA_dt is not None:
            dlnA_dt = self.dlnA_dt(self.x, time)[idx]
            dlnA_dt = dlnA_dt[:, None]
            rhs -= state_array[idx, :3] * dlnA_dt

        if self.dlnA_dx is not None:
            dlnA_dx = np.array(self.dlnA_dx(self.x, time)[idx])
            rhs[:, 0] -= state_array[idx, 1] * dlnA_dx
            rhs[:, 1] -= (state_array[idx, 1] ** 2.0 / state_array[idx, 0]) * dlnA_dx
            rhs[:, 2] -= (
                state.velocity[idx] * (state_array[idx, 2] + state.pressure[idx])
            ) * dlnA_dx

        return rhs

    def source_fast(self, time: float, y: Array, args: tuple[Array, Array]):
        x = args[0]
        gamma = args[1]
        n = len(x)
        r = y[0:n]
        ru = y[n : 2 * n]
        E = y[2 * n : 3 * n]
        p = (gamma - 1) * (E - 0.5 * ru**2 / r)
        rhs = np.zeros_like(y)
        if self.dlnA_dt is not None:
            dlnAdt = self.dlnA_dt(x, time)
            rhs[0:n] -= r * dlnAdt
            rhs[n : 2 * n] -= ru * dlnAdt
            rhs[2 * n : 3 * n] -= E * dlnAdt
        if self.dlnA_dx is not None:
            dlnAdx = self.dlnA_dx(x, time)
            rhs[0:n] -= ru * dlnAdx
            rhs[n : 2 * n] -= (ru**2.0 / r) * dlnAdx
            rhs[2 * n : 3 * n] -= (ru / r * (E + p)) * dlnAdx
        return rhs
    

    def source_fast_jacobian_banded(self, time: float, y: Array, args: tuple[Array, Array]):
        x = args[0]
        gamma = args[1]
        n = len(x)
        r = y[0:n]
        ru = y[n : 2 * n]
        E = y[2 * n : 3 * n]
        p = (gamma - 1) * (E - 0.5 * ru**2 / r)

        dlnAdt = self.dlnA_dt(x, time) if self.dlnA_dt is not None else np.zeros_like(x)
        dlnAdx = self.dlnA_dx(x, time) if self.dlnA_dx is not None else np.zeros_like(x)

        J_diag = np.zeros(3 * n)

        # Diagonal entries per variable
        J_diag[0:n]       = -dlnAdt                         # ∂R/∂ρ
        J_diag[n:2*n]     = -dlnAdt                         # ∂R/∂(ρu)
        J_diag[2*n:3*n]   = -dlnAdt - (ru / r) * gamma * dlnAdx     # ∂R/∂E
        return J_diag.reshape(1, 3*n)  # shape (1, 3n): banded with 0 lower and upper bandwidth