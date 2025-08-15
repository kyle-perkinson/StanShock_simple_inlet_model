import numpy as np
from flowstate import FlowState
import comp_flow_fxns as cff

class Feature:
    """
    Represents a geometric or flow feature (shock, expansion, corner).
    Knows its category, start point, angle, pre/post FlowState, and characteristic line.
    """
    def __init__(self, pre_state: FlowState, theta_new: float, sigma: float, x_start: float, y_start: float, wall_type: str):
        self.x_start = x_start
        self.y_start = y_start
        self.theta_new = theta_new     #new flow angle
        self.sigma = sigma             # Feature angle (radians)
        self.pre_state = pre_state
        self.post_state = None
        self.wall_type = wall_type

    def compute_post_state(self):
        """Generate post_state from pre_state using sigma (angle change)."""
        self.post_state = self.pre_state.through_feature(self)
        return self.post_state

    def y_at(self, x: float) -> float:
        """Return y-position of the feature characteristic line at given x."""
        return self.y_start + np.tan(self.sigma) * (x - self.x_start)
    
    def through_feature(self, T_new, P_new, M_new):
        """
        Return a new FlowState after passing through a feature (shock or expansion).
        theta_new: flow angle after feature (radians)
        """
        delta = self.theta_new - self.pre_state.theta

        if delta > 0: #shock
            T_rat, P_rat, M2, h1 = cff.oblique_solver(self.pre_state.k, self.pre_state.M, delta)
        elif delta < 0: 
            T_rat, P_rat, M2, nu1, nu2 = cff.pm_solver(self.pre_state.k, self.pre_state.M, np.abs(delta))
        else:
            return FlowState(
                T=self.pre_state.T,
                P=self.pre_state.P,
                M=self.pre_state.M,
                theta=self.pre_state.theta,
                k=self.pre_state.k
            )

        return FlowState(
            T=self.pre_state.T * T_rat,
            P=self.pre_state.P * P_rat,
            M=M2,
            theta=self.theta_new,
            k=self.pre_state.k
        )
    

