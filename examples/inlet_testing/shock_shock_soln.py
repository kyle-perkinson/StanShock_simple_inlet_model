import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils.comp_flow_fxns as cff
from scipy.optimize import root_scalar

class FlowState:
    def __init__(self, composition: str = "air"):
        if composition == "air":
            mech = "../Stanshock/data/mechanisms/N2O2HeAr.yaml"
            X = "O2:0.21 N2:0.79"
        else:
            mech = composition
            X = composition
        self.gas = ct.Solution(mech)
        self.X = X
        self.k = None
        self.M = None
        self.theta = None
        self.P = None
        self.T = None
        self.rho = None

    def set_state(self, T: float, P: float, M: float, theta: float):
        self.gas.TPX = T, P, self.X
        self.k = self.gas.cp_mass / self.gas.cv_mass
        self.M = M
        self.theta = theta
        self.P = P
        self.T = T
        self.rho = self.gas.density


def feature_calc(inflow, theta_guess):
    delta = theta_guess - inflow.theta
    if delta > 0:
        T_rat, P_rat, M2, h = cff.oblique_solver(inflow.k, inflow.M, delta)
    else:
        T_rat, P_rat, M2, h, h2 = cff.pm_solver(inflow.k, inflow.M, delta)
    T2, P2 = inflow.T * T_rat, inflow.P * P_rat
    outflow = FlowState(composition="air")
    outflow.set_state(T=T2, P=P2, M=M2,theta=theta_guess)
    return outflow

ambient = FlowState(composition="air")
ambient.set_state(T=300, P=1e5, M=3.0,theta=0.0)

def p_theta(theta_guess, upper_state, lower_state):
    #Next time: 
    outflow1 = feature_calc(upper_state, theta_guess - upper_state.theta)
    outflow2 = feature_calc(upper_state, lower_state.theta - theta_guess)
    return np.abs(outflow1.P - outflow2.P)
    # We will need to finish up this part. What wall is it from? And if so, how does it behave? 

def riemann_problem2d(upper_state, lower_state):
    theta_slipline = root_scalar(p_theta, x0 = np.mean([upper_state.theta, lower_state.theta]), args=(upper_state,lower_state)).root
    outflow_upper = feature_calc(upper_state, theta_slipline)
    outflow_lower = feature_calc(lower_state, theta_slipline)
    return outflow_upper, outflow_lower
    



        
    


delta_u = np.radians(-15)
delta_d = np.radians(10)

A = feature_calc(ambient, np.abs(delta_u))
D = feature_calc(ambient, np.abs(delta_d))

B, C= riemann_problem2d(A, D)

print(B.theta)
print(B.P)
print(C.theta)
print(C.P)
pass

































        