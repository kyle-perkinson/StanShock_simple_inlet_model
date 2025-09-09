import numpy as np
import moc_inlet.utils.comp_flow_fxns as cff
from moc_inlet.utils.flowstate import FlowState
from moc_inlet.utils.geometry import Geometry, Inflections
from moc_inlet.utils.segment import Wave, WallSegment, Slipstream
from matplotlib import pyplot as plt
from scipy.optimize import root_scalar
def reflected_wave(x_next: float, wave: Wave, wall: WallSegment):
    inflow = wave.post_state
    proj, normal, delta = get_proj(inflow.theta, wall.sigma, wall.normal)
    y_next = wall.y_at(x_next)
    if proj > 0:
        return [generate_oblique_shock(inflow, delta, x_next, y_next, wall.sigma, normal)]
    elif proj < 0:
        return generate_pm_fan(inflow, delta, x_next, y_next, wall.sigma, normal, 1)

def incident_wave(x_next: float, event, inflow: FlowState, wave_res: float):
    proj, normal, delta = get_proj(inflow.theta, event.sigma, event.normal)    
    if proj > 0:
        return [generate_oblique_shock(inflow, delta, float(event.x_start), float(event.y_start), float(event.sigma), normal)]
    elif proj < 0:
        return generate_pm_fan(inflow, delta, float(event.x_start), float(event.y_start), float(event.sigma), normal, wave_res)

def riemann_problem(x_next: float, wave1: Wave, wave2: Wave, wave_res: float):
    y_next = wave1.y_at(x_next)
    state1 = wave1.post_state
    state2 = wave2.post_state
    
    theta_guess = np.mean([state1.theta, state2.theta])
    out_waves = []
    wave3 = None; wave4 = None
    sol = root_scalar(
        dP_theta,
        x0=theta_guess,
        args=(state1, state2, wave1, wave2),
        method='secant'
    )
    theta_slipstream = sol.root
    
    (delta_n1, n1, delta1), (delta_n2, n2, delta2) = wave_logic(theta_slipstream, state1, state2, wave1, wave2)
    if delta_n1 > 0:
        wave3 = [generate_oblique_shock(state1, delta1, x_next, y_next, theta_slipstream, n1)]

    elif delta_n1 < 0:
        wave3 = generate_pm_fan(state1, delta1, x_next, y_next, theta_slipstream, n1, wave_res)

    if delta_n2 > 0:
        wave4 = [generate_oblique_shock(state2, delta2, x_next, y_next, theta_slipstream, n2)]

    elif delta_n2 < 0:
        wave4 = generate_pm_fan(state2, delta2, x_next, y_next, theta_slipstream, n2, wave_res)
    
    if wave3:
        out_waves.extend(wave3)
        M3 = wave3[-1].post_state.M


    if wave4:
        out_waves.extend(wave4)
        M4 = wave4[-1].post_state.M
    
    if not np.isclose(M3, M4, atol=1e-12):
        slip3 = [Slipstream(x_next, y_next, theta_slipstream, n1)]
        out_waves.extend(slip3)
        slip4 = [Slipstream(x_next, y_next, theta_slipstream, n2)]
        out_waves.extend(slip4)
    return out_waves








def generate_oblique_shock(inflow: FlowState, delta, x_start: float, y_start: float, event_angle: float, event_normal=None):
    T_rat, P_rat, M_out, h = cff.oblique_solver(inflow.k, inflow.M, np.abs(delta))
    outflow = inflow.clone()
    outflow.set_state(
            T = inflow.T * T_rat,
            P = inflow.P * P_rat,
            M = M_out,
            theta = event_angle,
        )
    if event_normal is not None:
        sigma = get_sigma(inflow.theta, h, event_normal)
    else:
        sigma = h * np.sign(delta) + inflow.theta
    shock = Wave(x_start, y_start, sigma, inflow, outflow, wave_type="shock")
    return shock

def generate_pm_fan(inflow: FlowState, delta: float, x_start: float, y_start: float, event_angle: float, event_normal, wave_res: float):
    M2, nu1, nu2, mu1, mu2, _, _ = cff.pm_solver(inflow.k, inflow.M, np.abs(delta))

    theta_new_pm = -abs(event_angle - inflow.theta)

    sigma_eval_pm = np.linspace(mu1, theta_new_pm + mu2, wave_res)
    

    theta_eval_real = np.linspace(inflow.theta, event_angle, wave_res)
    mu_eval = np.linspace(mu1, mu2, wave_res)
    sigma_wave = get_sigma(theta_eval_real, mu_eval, event_normal)
    sigma_wave = np.atleast_1d(sigma_wave).reshape(-1)
    waves = []
    last_state = inflow.clone()

    for i in range(len(sigma_eval_pm)):
        M_out_i = cff.M_expl(last_state.k, nu1, 0.0, sigma_eval_pm[i])
        T_rat_i = (cff.H(last_state.k, last_state.M, M_out_i))
        P_rat_i = T_rat_i**(last_state.k / (last_state.k - 1))
        outflow = last_state.clone()
        outflow.set_state(
            T = last_state.T * T_rat_i,
            P = last_state.P * P_rat_i,
            M = M_out_i,
            theta = theta_eval_real[i],
        )
        wave_i = Wave(x_start,
                      y_start,
                      float(sigma_wave[i]),
                      last_state,
                      outflow,
                      wave_type="expansion")
        waves.append(wave_i)
        last_state = outflow

    return waves


def get_sigma(theta_inflow, phi, n_wall):
    theta_inflow = np.atleast_1d(theta_inflow)
    phi = np.atleast_1d(phi)
    theta_inflow, phi = np.broadcast_arrays(theta_inflow, phi)

    n_wall = np.asarray(n_wall).reshape(2)  # force (2,)
    candidates = np.stack([theta_inflow + phi, theta_inflow - phi], axis=-1)  # (..., 2)
    vecs = np.stack([np.cos(candidates), np.sin(candidates)], axis=-1)       # (..., 2, 2)
    dot_products = np.tensordot(vecs, n_wall, axes=([2], [0]))               # (..., 2)
    idx = np.argmax(dot_products > 0, axis=-1)                               # (...,)

    sigma_out = np.take_along_axis(candidates, idx[..., None], axis=-1)[..., 0]

    return np.squeeze(sigma_out)





def dP_theta(theta_slipstream, state1, state2, wave1: Wave, wave2: Wave):
    (delta_n1, n1, delta1), (delta_n2, n2, delta2) = wave_logic(theta_slipstream, state1, state2, wave1, wave2)
    p1 = get_p(state1, delta_n1, delta1)
    p2 = get_p(state2, delta_n2, delta2)
    return p1 - p2

def compute_normals(theta_slipstream):
    v = np.array([np.cos(theta_slipstream), np.sin(theta_slipstream)])
    n1 = [v[1], -v[0]]
    n2 = [-v[1], v[0]]
    return np.vstack((n1, n2))

def get_proj(inflow_angle, new_angle, normal):
    u1 = np.array([np.cos(inflow_angle), np.sin(inflow_angle)])
    u2 = np.array([np.cos(new_angle), np.sin(new_angle)])


    delta = new_angle - inflow_angle
    
    normal = normal.flatten()
    normal /= np.linalg.norm(normal)
    delta_n = np.dot(u2 - u1, normal)

    return delta_n, normal, delta

def get_p(inflow, delta_n, delta):
    if delta_n > 0:
        T_rat, P_rat, M2, h = cff.oblique_solver(inflow.k, inflow.M, np.abs(delta))
    else:
        _, _, _, mu1, mu2, T2_T1, P_rat, = cff.pm_solver(inflow.k, inflow.M, np.abs(delta))
    return inflow.P * P_rat

def wave_logic(theta_slipstream, state1, state2,  wave1: Wave, wave2: Wave):
    norms = compute_normals(theta_slipstream)
    mask = norms[:,1] > 0
    if np.sign(wave1.sigma) !=  np.sign(wave2.sigma) > 0:

        if wave1.sigma < 0:
            n1 = norms[mask]
            n2 = norms[~mask]
        else:
            n1 = norms[~mask]
            n2 = norms[mask]
    else:
        raise RuntimeError("You haven't done same family shock intersection yet!")
    delta_n1, n1, delta1 = get_proj(state1.theta, theta_slipstream, n1)
    delta_n2, n2, delta2 = get_proj(state2.theta, theta_slipstream, n2)

    return (delta_n1, n1, delta1), (delta_n2, n2, delta2)
