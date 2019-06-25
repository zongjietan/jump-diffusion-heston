import numpy as np

# log(cosh(x) + ysinh(x))
def log_cosh_sinh(x,y):
    return np.log(0.5) + x + np.log(1 + np.exp(-2*x) + y*(1 - np.exp(-2*x)))

# (y + tanh(x))/(1 + ytanh(x))
def tanh_frac(x,y):
    return (y*(1 + np.exp(-2*x)) + 1 - np.exp(-2*x))/(1 + np.exp(-2*x) + y*(1 - np.exp(-2*x)))

def truncate_curve(time_points_in, time_horizon):
    time_points_out = time_points_in
    if np.prod(time_horizon - time_points_out) != 0:
        time_points_out = np.append(time_points_out, time_horizon)
    time_points_out = np.sort(time_points_out)
    mask = (time_points_out <= time_horizon)
    time_points_out = time_points_out[mask]
    return time_points_out

def parameter_steps(params, time):
    truncated_times = truncate_curve(params[:,0], time)
    timesteps = np.ediff1d(truncated_times)
    param_steps = params[:len(timesteps),:].copy()
    param_steps[:,0] = timesteps
    return param_steps
