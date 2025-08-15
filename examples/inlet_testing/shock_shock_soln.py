import numpy as np
import src.moc_inlet.utils.comp_flow_fxns as cff
from scipy.optimize import root_scalar
from src.moc_inlet.utils.flowstate import FlowState

"""
This is a very crude 2-D Riemann problem; has yet to be implemented into the src.moc_inlet.utils
Goal here is just to figure out correct way to obtain theta/p function and find slipstream states
"""


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

































        