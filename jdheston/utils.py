import numpy as np

# log(cosh(x) + ysinh(x))
def log_cosh_sinh(x,y):
    return np.log(0.5) + x + np.log(1 + np.exp(-2*x) + y*(1 - np.exp(-2*x)))

# (y + tanh(x))/(1 + ytanh(x))
def tanh_frac(x,y):
    return (y*(1 + np.exp(-2*x)) + 1 - np.exp(-2*x))/(1 + np.exp(-2*x) + y*(1 - np.exp(-2*x)))
