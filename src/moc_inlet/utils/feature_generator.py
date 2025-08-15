import numpy as np
import moc_inlet.utils.comp_flow_fxns as cff

from moc_inlet.utils.flowstate import FlowState
from moc_inlet.utils.feature import Feature

def feature_generator(pre_state: FlowState,
                       theta_new: float,
                       x_start: float,
                       y_start: float,
                       wall_type: str,
                       N_wave=10):
    delta = theta_new - pre_state.theta
    if wall_type == 'upper':
        feature_type = 'shock' if delta < 0 else 'pm_fan'
    elif wall_type == 'lower':  # lower wall
        feature_type = 'pm_fan' if delta < 0 else 'shock'

    if feature_type == 'shock':
        return generate_oblique_shock(pre_state, theta_new, x_start, y_start, wall_type)
    elif feature_type == 'pm_fan':
        return generate_pm_fan(pre_state, theta_new, x_start, y_start, wall_type, N_wave)
    
def generate_pm_fan(pre_state, theta_new, delta, x_start, y_start, wall_type, N_wave):
    T_rat_total, P_rat_total, M_out, mu_in, mu_out = cff.pm_solver(pre_state.k, pre_state.M, np.abs(delta))
    if pre_state.theta < theta_new: #flips flow vectors if from upper wall so pre_state.theta > theta_new
        theta_in_pm =  -pre_state.theta
        theta_new_pm = -theta_new #
        sigma_flipped = -1
    else:
        theta_in_pm = pre_state.theta
        theta_new_pm = theta_new
        sigma_flipped = 1

        
    
    sigma_in = theta_in_pm + mu_in
    sigma_out = theta_new_pm + mu_out
    sigma_eval_pm = np.linspace(sigma_in, sigma_out, N_wave)
    theta_eval_real = np.linspace(pre_state.theta, theta_new, N_wave)
    waves = []
    last_state = pre_state.clone()

    for i in range(len(sigma_eval_pm)):
        M_out_i = cff.M_expl(last_state.k, mu_in, theta_in_pm, sigma_eval_pm[i])
        T_rat_i = cff.H(last_state.k, last_state.M, M_out)
        P_rat_i = T_rat_i**(last_state.k / (last_state.k - 1))
        post_state = last_state.clone()
        post_state.set_state(
            T = last_state.T * T_rat_i,
            P = last_state.P * P_rat_i,
            M = M_out_i,
            theta = theta_eval_real[i],
        )

        wave_i = Feature(pre_state = last_state,
                         theta_new = theta_eval_real[i],
                         sigma = sigma_flipped * sigma_eval_pm[i],
                         x_start = x_start,
                         y_start = y_start,
                         wall_type = wall_type
        )

        wave_i.post_state = post_state
        waves.append(wave_i)
        last_state = post_state
    return waves


def generate_oblique_shock(pre_state, theta_new, delta, x_start, y_start, wall_type):
    if delta < 0:
        sigma_flipped = -1
        theta_in = -pre_state.theta
    else:
        sigma_flipped = 1
        theta_in = pre_state.theta

    T_rat, P_rat, M_out, h = cff.oblique_solver(pre_state.k, pre_state.M, np.abs(delta))

    post_state = pre_state.clone()
    post_state.set_state(
            T = pre_state.T * T_rat,
            P = pre_state.P * P_rat,
            M = M_out,
            theta = theta_new,
        )
    sigma = theta_in + (sigma_flipped * h)
    wave = Feature(
        pre_state = pre_state,
        theta_new = theta_new,
        sigma = sigma,
        x_start = x_start,
        y_start = y_start,
        wall_type = wall_type
    )

    wave.post_state = post_state

    return wave











                         



 