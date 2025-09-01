import numpy as np
import moc_inlet.utils.comp_flow_fxns as cff
from moc_inlet.utils.flowstate import FlowState
from moc_inlet.utils.geometry import Geometry, Inflections
from moc_inlet.utils.segment import Wave, WallSegment, Slipstream

def reflected_wave(x_next: float, wave: Wave, wall: WallSegment):
    inflow = wave.post_state
    proj, normal, delta = get_proj(inflow.theta, wall.sigma, wall.normal)
    y_next = wall.y_at(x_next)
    if proj < 0:
        return [generate_oblique_shock(inflow, delta, x_next, y_next, wall.sigma, normal)]
    elif proj > 0:
        return generate_pm_fan(inflow, delta, x_next, y_next, wall.sigma, normal, 1)

def incident_wave(x_next: float, event: Inflections, inflow: FlowState, N_wave: int):
    # delta = event.sigma - inflow.theta
    proj, normal, delta = get_proj(inflow.theta, event.sigma, event.normals)
    # v = np.array([np.cos(inflow.theta), np.sin(inflow.theta)])
    # normal = event.normals.flatten()
    # proj = delta * np.dot(v, normal)
    
    if proj < 0:
        return [generate_oblique_shock(inflow, delta, float(event.x), float(event.y), float(event.sigma), normal)]
    elif proj > 0:
        return generate_pm_fan(inflow, delta, float(event.x), float(event.y), float(event.sigma), normal, N_wave)

def riemann_problem(x_next: float, wave1: Wave, wave2: Wave):
    pass



def generate_oblique_shock(inflow: FlowState, delta, x_start: float, y_start: float, event_angle: float, event_normal):
    T_rat, P_rat, M_out, h = cff.oblique_solver(inflow.k, inflow.M, np.abs(delta))
    outflow = inflow.clone()
    outflow.set_state(
            T = inflow.T * T_rat,
            P = inflow.P * P_rat,
            M = M_out,
            theta = event_angle,
        )
    sigma = get_sigma(inflow.theta, h, event_normal)
    shock = Wave(x_start, y_start, sigma, inflow, outflow, wave_type="shock")
    return shock

def generate_pm_fan(inflow: FlowState, delta: float, x_start: float, y_start: float, event_angle: float, event_normal, N_wave: int):
    T_rat_total, P_rat_total, M_out, mu_in, mu_out = cff.pm_solver(inflow.k, inflow.M, np.abs(delta))
    if inflow.theta < event_angle: #flips flow vectors if from upper wall so inflow.theta > theta_new
        theta_in_pm =  -inflow.theta
        theta_new_pm = -event_angle
    else:
        theta_in_pm = inflow.theta
        theta_new_pm = event_angle

    sigma_eval_pm = np.linspace(theta_in_pm + mu_in, theta_new_pm + mu_out, N_wave)
    

    theta_eval_real = np.linspace(inflow.theta, event_angle, N_wave)
    mu_eval = np.linspace(mu_in, mu_out, N_wave)
    sigma_wave = get_sigma(theta_eval_real, mu_eval, event_normal)
    sigma_wave = np.atleast_1d(sigma_wave).reshape(-1)
    waves = []
    last_state = inflow.clone()

    for i in range(len(sigma_eval_pm)):
        M_out_i = cff.M_expl(last_state.k, mu_in, theta_in_pm, sigma_eval_pm[i])
        T_rat_i = cff.H(last_state.k, last_state.M, M_out)
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

def get_proj(inflow_angle, new_angle, normal):
    delta = new_angle - inflow_angle
    v = np.array([np.cos(inflow_angle), np.sin(inflow_angle)])
    
    normal = normal.flatten()
    
    # Flip only the y component if negative
    normal_for_proj = normal.copy()
    if normal_for_proj[1] < 0:
        normal_for_proj[1] *= -1

    proj = np.sign(delta) * np.dot(v, normal_for_proj)
    return proj, normal, delta