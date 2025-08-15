import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
import utils.comp_flow_fxns as cff




mech = "../Stanshock/data/mechanisms/N2O2HeAr.yaml"
M_A = 3.00
gas = ct.Solution(mech)
X_A = 'O2:0.21 N2:0.79'
gas.TPX = 300, 1e5, X_A
k = gas.cp / gas.cv
thetaA = 0
thetaB = np.radians(-12)

x_wall = [-1, 0, 1]
y_down = [0, 0, np.tan(thetaB)] 
y_up = [0.5, 0.5, 0.5]
x_wall_plot = np.array([-1, -1, 0, 1])
y_wall_plot = np.array([np.tan(thetaB), 0, 0, np.tan(thetaB)])
xy_fan = [0, 0]

#ANYTHING "A" corresponds to inflow FlowState
sigma_a = (thetaA + np.asin((1/M_A))) #theta_A is initial FlowState theta, 
nuA, nuB, M_B = cff.pm_mach_solver(k, M_A, thetaB)

sigma_b = (thetaB + np.asin((1/M_B)))

M_arr = [M_A]
T_arr = [gas.T]
P_arr = [gas.P]
N_wave = 5
psi_eval = np.linspace(sigma_a, sigma_b, N_wave)

for i in range(len(psi_eval)):
    M_out = M_expl(k, nuA, thetaA, psi_eval[i])
    T_rat = cff.H(k, M_arr[-1], M_out)
    P_rat =  cff.H(k, M_arr[-1], M_out)**(k / (k - 1))
    T_arr.append(T_rat)
    P_arr.append(P_rat)
    M_arr.append(M_out)






