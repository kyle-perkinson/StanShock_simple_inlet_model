from __future__ import annotations
import numpy as np
import cantera as ct
from scipy.optimize import root_scalar

"""
OBLIQUE SHOCK FUNCTIONS
"""

def shock_angle(k, M, d, a=1): #Returns: shock angle (rads) | assumes weak shock.
    #Inputs: gamma, Mach number, deflection angle (rads)
    lamb = np.sqrt((M**2 -1)**2 - 3*(1+((k-1)/2)*M**2)*(1+((k+1)/2)*M**2)*np.tan(d)**2)
    chi = (1/(lamb**3))*((M**2 - 1)**3 - 9*(1+((k-1)/2)*M**2)*(1+((k-1)/2)*M**2 + ((k+1)/4)*M**4)*np.tan(d)**2)
    chi = np.clip(chi, -1, 1)
    numerator = M**2 - 1 + 2*lamb*np.cos((4*np.pi*a + np.arccos(chi))/3)
    denominator = 3*(1 + ((k-1)/2)*M**2)*np.tan(d)
    return (np.arctan(numerator/denominator))

def F(k, M, h): #Returns: deflection angle (rads)
        #Inputs: gamma, Mach number, shock angle (rads)
    return (np.atan(((2 / np.tan(h)) * (M**2 * np.sin(h)**2 - 1)) / 
            (M**2 * (k + np.cos(2 * h)) + 2) ))

def G(k, M, h): #Returns: post-obl-shock Mach number
    #Inputs: gamma, Mach number, shock angle (rads)
    return (
        (np.sqrt(1 + (k - 1) * M**2 * np.sin(h)**2 + ((((k + 1)/2)**2 - k*np.sin(h)**2) * M**4 * np.sin(h)**2))) /
        (np.sqrt( (k*M**2*np.sin(h)**2) - ((k - 1) / 2)) * np.sqrt( ((k - 1)/2) * M**2 * np.sin(h)**2 + 1) )
        )

def W(k, M, h): #Returns: post-obl-shock pressure ratio
    #Inputs: gamma, Mach number, shock angle (rads)
    return ((2 * k * M**2 * np.sin(h)**2 - (k - 1)) / (k + 1))

def dW(h, k, M, P_rat):
    return (P_rat - W(k, M, h))

def A(k, M, h): #Returns: post-obl-shock temperature ratio
    #Inputs: gamma, Mach number, shock angle (rads)
    return (
        ((2*k*M**2 * np.sin(h)**2 - (k - 1)) * ((k - 1) * M**2 * np.sin(h)**2 + 2)) /
        ((k + 1) * M * np.sin(h))**2
    )

"""
PRANDTL-MEYER EXPANSION FUNCTIONS
"""

def pm_fxn(k, M):
    return (
        np.sqrt((k + 1) / (k - 1)) *
        np.arctan(np.sqrt((k - 1)*(M**2 - 1)/(k + 1))) -
        np.arctan(np.sqrt(M**2 - 1))
        )
def pm_diff(M, nu, k):
    return (nu - pm_fxn(k, M))

def pm_mach_solver(k, M, d):
    nu1 = pm_fxn(k, M)
    nu2 = nu1 + np.abs(d)
    M2 = root_scalar(pm_diff, x0=2.0, x1=3.0, args=(nu2, k)).root
    return M2

def H(k, M_in, M_out): #Returns: post-expansion temperature ratio
    #Inputs: gamma, incoming Mach number, outgoing Mach number
    return (
        (1 + ((k - 1) / 2) * M_in**2) / (1 + ((k - 1) / 2) * M_out**2)
    )

def M_expl(k, nu1, theta1, psi):
    return (
    np.sqrt(1 + ((k + 1) / (k - 1)) * (np.tan(np.sqrt((k - 1) / (k + 1)) 
    * (nu1 + theta1 - psi + np.pi/2))**2))
    )
    

"""
SOLVER WRAPPER FUNCTIONS
"""


def oblique_solver(k, M, d):
    """
    Wrapper function.
    Inputs: gas object, Mach number, relative deflection angle
    Outputs: post-shock gas object, post-shock Mach number, shock angle relative to horizontal
    """
    h = shock_angle(k, M, d)
    M2 = G(k, M, h)
    T2_T1 = A(k, M, h)
    P2_P1 = W(k, M, h)
    return T2_T1, P2_P1, M2, h


    
def pm_solver(k, M, d):
    """
    Wrapper function for Prandtl-Meyer fan.
    Inputs (from self): gas object, Mach number, relative turning angle
    Outputs: post-fan gas object, post-fan Mach number, fan angles v1 and v2 relative to horizontal
    """
    M2 = pm_mach_solver(k, M, np.abs(d))
    P2_P1 = (H(k, M, M2)) ** (k / (k - 1))
    T2_T1 = (H(k, M, M2))
    mu1 = np.asin((1/M))
    mu2 = np.asin((1/M2))
    return T2_T1, P2_P1, M2, mu1, mu2


def shock_fan_solver(k, M0, T0, P0, d1, d2):
    h1 = shock_angle(k, M0, d1)
    M1 = G(k, M0, h1)
    P1 = W(k, M0, h1) * P0
    T1 = A(k, M0, h1) * T0

    nu1, nu2, M2 = pm_solver(k, M0, d2)
    P2 = (H(k, M0, M2)) ** (k / (k - 1)) * P0
    T2 = H(k, M0, M2) * T0

    P4 = P1 * (H(k, M0, M2) ** (k / (k - 1)))
    P3_P2 = P4 / P2
    h3 = root_scalar(dW, x0 = 0.3, args=(k, M2, P3_P2)).root
    d3 = F(k, M2, h3)
    M3 = G(k, M2, h3)
    T3 = A(k, M2, h3) * T2
    P3 = P2 * W(k, M2, h3)
    d4 = d2 + d3 - d1
    nu3, nu4, M4 = pm_solver(k, M1, d4)

    T4 = T1 * (H(k, M1, M4))
    T_out = (T3 + T4) /2
    P_out = P4
    M_out = (M3 + M4) / 2
    return T_out, P_out, M_out


