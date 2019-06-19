import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

# Heston characteristic function
# for SDE dV = αβ(V)^0.5dW + β(γ - V)dt, V0 = v0
def char_func(u,t,θ):
    α,β,γ,v0,ρ = θ
    i = 1j
    #####################################################################
    # see . on p. of                                                    #
    # θ1 = 1-α*ρ*i*u                                                    #
    # θ2 = np.sqrt(θ1**2 + α**2*u*(i+u))                                #
    # θ3 = 0.5*β*θ2                                                     #
    # C = (θ1*t-2/β*np.log(np.cosh(θ3*t)+θ1/θ2*np.sinh(θ3*t)))*γ/α**2   #
    # D = (θ1-θ2*(θ1+θ2*np.tanh(θ3*t))/(θ2+θ1*np.tanh(θ3*t)))/β/α**2*v0 #
    #####################################################################
    # see ϕ2 on p.4 of 'the little heston trap'
    κ,η,λ = β,γ,α*β
    d = np.sqrt((ρ*λ*u*i-κ)**2+λ**2*(i*u+u**2))
    g1 = (κ-ρ*λ*i*u+d)/(κ-ρ*λ*i*u-d)
    g2 = 1/g1
    C = η*κ*λ**-2*((κ-ρ*λ*i*u-d)*t-2*np.log((1-g2*np.exp(-d*t))/(1-g2)))
    D = v0*λ**-2*(κ-ρ*λ*i*u-d)*(1-np.exp(-d*t))/(1-g2*np.exp(-d*t))
    return np.exp(C + D)

# THIS APPEARS GOOD!!!
# allow for float forward variance
# def char_func(u,t,params):
#     ξ0,ξ1,α,β,ρ = params
#     κ,λ = β,α*β
#     i=1j
#     d = np.sqrt((ρ*λ*u*i-κ)**2+λ**2*(i*u+u**2))
#     g1 = (κ-ρ*λ*i*u+d)/(κ-ρ*λ*i*u-d)
#     g2 = 1/g1
#
#     A  = ξ0*κ*λ**-2*((κ-ρ*λ*i*u-d)*(1/3)-2*np.log((1-g2*np.exp(-d*(1/3)))/(1-g2)))
#     A += ξ1*κ*λ**-2*((κ-ρ*λ*i*u-d)*(2/3)-2*np.log((1-g2*np.exp(-d*(2/3)))/(1-g2)))
#     B  = (ξ0 -  0)*λ**-2*(κ-ρ*λ*i*u-d)*(1-np.exp(-d*(3/3)))/(1-g2*np.exp(-d*(3/3)))
#     B += (ξ1 - ξ0)*λ**-2*(κ-ρ*λ*i*u-d)*(1-np.exp(-d*(2/3)))/(1-g2*np.exp(-d*(2/3)))
#
#     return np.exp(A + B)

# def char_func(u,t,params):
#     ξ,ρ,ν,H,ɛ = params
#     i = 1j
#     T = truncate_curve(ξ[:,0],t)
#     ξ = ξ[:len(T)-1,1]
#     Δξ = np.ediff1d(ξ,to_begin=ξ[0])
#     ΔTA = np.ediff1d(T)
#     ΔTB = t - T[:-1]
#     # first deal with NIG limit
#     if ɛ == 0:
#         C = 1/ν**2*(1 - ρ*ν*i*u - np.sqrt((1 - ρ*ν*i*u)**2 + ν**2*i*u*(1 - i*u)))
#         return np.exp(C*np.sum(ξ*ΔTA))
#     # otherwise change to Albrecher notation
#     κ,λ = ɛ**-1,ν*ɛ**(H - 0.5)
#     d = np.sqrt((ρ*λ*u*i-κ)**2+λ**2*(i*u+u**2))
#     g1 = (κ-ρ*λ*i*u+d)/(κ-ρ*λ*i*u-d)
#     g2 = 1/g1
#     # ΣA = np.sum((θ1*ξ[:,0] - θ2/θ3*np.log(np.cosh(θ3*ξ[:,0]) + θ1/θ2*np.sinh(θ3*ξ[:,0])))*ξ[:,1]/α**2)
#     # B = (θ1 - θ2*(θ1 + θ2*np.tanh(θ3*t))/(θ2 + θ1*np.tanh(θ3*t)))/β/α**2
#     B = λ**-2*(κ-ρ*λ*i*u-d)*(1-np.exp(-d*ΔTB))/(1-g2*np.exp(-d*ΔTB))
#     ΣA = np.sum(ξ*κ*λ**-2*((κ-ρ*λ*i*u-d)*ΔTA-2*np.log((1-g2*np.exp(-d*ΔTA))/(1-g2))))
#     return np.exp(ΣA + np.sum(B*Δξ))

def truncate_curve(time_points_in, time_horizon):
    time_points_out = time_points_in
    if np.prod(time_horizon - time_points_out) != 0:
        time_points_out = np.append(time_points_out, time_horizon)
    time_points_out = np.sort(time_points_out)
    mask = (time_points_out <= time_horizon)
    time_points_out = time_points_out[mask]
    return time_points_out

# Jump (Mechkov) Heston characteristic function
def jump_heston_cf(u, t, θ):
    σ, ρ, v = θ
    i = 1j
    a = σ*ρ*v*i*u
    return np.exp((1 - a - np.sqrt((1 - a)**2 + (σ*v)**2*u*(i + u)))/v**2*t)

# objective function for jump heston calibration
def rmse(x, expiry, logstrikes, vols):
    model_vols = jump_heston_vols(x, expiry, logstrikes)
    rmse = np.sqrt(np.mean((model_vols - vols)**2))
    return rmse

# Get variance swap from vols slice
def calibrate_jump_heston(expiry, logstrikes, vols):

    results =  minimize(rmse, (0.1, 0.0, 0.5),
                        method = 'L-BFGS-B',
                        args = (expiry, logstrikes, vols),
                        bounds = ((0,10), (-1,1), (0,10)),
                        options = {'maxiter': 100}
                        )
    return results








# Heston characteristic function
# def char_func(p,t,θ):
#     σ,ρ,v,κ = θ
#     i = 1j
#     θ0 = σ*ρ*v*i*p
#     if κ == np.inf:
#         return np.exp((1 - θ0 - np.sqrt((1 - θ0)**2 + σ**2*v**2*p*(i + p)))/v**2*t)
#     else:
#         # Mechkov
#         # θ1 = 1 - θ0
#         # θ2 = np.sqrt(θ1**2 + σ**2*v**2*p*(i + p))
#         # ϕ0 = (1 - θ0)*t - 2/κ*np.log(np.cosh(κ/2*θ2*t) + θ1/θ2*np.sinh(κ/2*θ2*t))
#         # ϕ1 = 1 - θ0 - θ2*(θ1 + θ2*np.tanh(κ/2*θ2*t))/(θ2 + θ1*np.tanh(κ/2*θ2*t))
#         # return np.exp(ϕ0/v**2 + ϕ1/v**2/κ)
#         # Gatheral / Albrecher et al
#         λ = κ*v*σ
#         η = σ**2
#         d = np.sqrt((ρ*λ*i*p - κ)**2 + λ**2*(i*p + p**2))
#         rm = κ - ρ*λ*i*p - d
#         rp = κ - ρ*λ*i*p + d
#         g = rm/rp
#         C = η*κ/λ**2*(rm*t - 2*np.log((1 - g*np.exp(-d*t))/(1-g)))
#         D = σ**2/λ**2*rm*(1 - np.exp(-d*t))/(1 - g*np.exp(-d*t))
#         return np.exp(C + D)

## log-price and integrated variance joint transform
def joint_mgf(u,t,params):
    u1,u2 = u
    V0,θ,ρ,ν,H,ɛ = params
    # first deal with levy limits
    if ɛ == 0:
        if   H >  -0.5: C = V0*(0.5*u1*(u1 - 1) + u2)
        elif H == -0.5: C = (V0 + θ)/ν**2*(1 - ρ*ν*u1 - np.sqrt((1 - ρ*ν*u1)**2 + ν**2*(u1*(1 - u1) - 2*u2)))
        elif H <  -0.5: C = -θ/ν*(ρ*u1 + np.sqrt(u1*(1 - (1 - ρ**2)*u1) - 2*u2))
        return np.exp(C*t)
    # otherwise change to convenient notation
    α = ν*ɛ**(H + 0.5)
    β = ɛ**-1
    γ = V0 + θ*ɛ**(H + 0.5)
    θ1 = 1 - α*ρ*u1
    θ2 = np.sqrt(θ1**2 + α**2*(u1*(1 - u1) - 2*u2))
    θ3 = 0.5*β*θ2
    S,C,T = np.sinh(θ3*t),np.cosh(θ3*t),np.tanh(θ3*t)
    A = (θ1*t - θ2/θ3*np.log(C + θ1/θ2*S))*γ/α**2
    B = (θ1 - θ2*(θ1 + θ2*T)/(θ2 + θ1*T))/β/α**2
    return np.exp(A + B*V0)

## Integrated CIR moment generating function
def integrated_cir_mgf(u,t,params):
    ɛ,H,ν,V0,θ = params
    # deal with levy limits
    if ɛ == 0:
        if   H >  -0.5: C = V0*u
        elif H == -0.5: C = (V0 + θ)/ν**2*(1 - np.sqrt(1 - 2*ν**2*u))
        elif H <  -0.5: C = -θ/ν*np.sqrt(-2*u)
        return np.exp(C*t)
    # otherwise change to convenient notation
    α = ɛ**(H + 0.5)*ν
    β = 1/ɛ
    γ = V0 + ɛ**(H + 0.5)*θ
    Θ2 = np.sqrt(1 - 2*α**2*u)
    Θ3 = 0.5*β*Θ2
    S,C,T = np.sinh(Θ3*t),np.cosh(Θ3*t),np.tanh(Θ3*t)
    A = (t - 2/β*np.log(C + S/Θ2))*γ/α**2
    B = (1 - Θ2*(1 + Θ2*T)/(Θ2 + T))/α**2/β
    return np.exp(A + B*V0)

## log-price moment generating function
def log_price_mgf(u,t,params):
    V0,θ,ρ,ν,H,ɛ = params
    # first deal with levy limits
    if ɛ == 0:
        if   H >  -0.5: C = 0.5*V0*u*(u - 1)
        elif H == -0.5: C = (V0 + θ)/ν**2*(1 - ρ*ν*u - np.sqrt((1 - ρ*ν*u)**2 + ν**2*u*(1 - u)))
        elif H <  -0.5: C = -θ/ν*(ρ*u + np.sqrt(u*(1 - (1 - ρ**2)*u)))
        # elif H <= -1.5: C = -np.inf
        return np.exp(C*t)
    # otherwise change to convenient notation
    α = ɛ**(H + 0.5)*ν
    β = 1/ɛ
    γ = V0 + ɛ**(H + 0.5)*θ
    θ1 = 1 - α*ρ*u
    θ2 = np.sqrt(θ1**2 + α**2*u*(1 - u))
    θ3 = 0.5*β*θ2
    S,C,T = np.sinh(θ3*t),np.cosh(θ3*t),np.tanh(θ3*t)
    A = (θ1*t - 2/β*np.log(C + θ1/θ2*S))*γ/α**2
    B = (θ1 - θ2*(θ1 + θ2*T)/(θ2 + θ1*T))/β/α**2
    return np.exp(A + B*V0)


## Integrated CIR characteristic function
def mgf(u,t,θ):
    σ,v,ɛ = θ
    ϑ = np.sqrt(1 - 2*(σ*v)**2*u)
    if ɛ == 0:
        return np.exp((1 - ϑ)/v**2*t)
    else:
        # RMc
        S = np.sinh(ϑ*t/2/ɛ)
        C = np.cosh(ϑ*t/2/ɛ)
        T = np.tanh(ϑ*t/2/ɛ)
        ϕ0 = t - 2*ɛ*np.log(C + 1/ϑ*S)
        ϕ1 = 1 - ϑ*(1 + ϑ*T)/(ϑ + T)
        return np.exp(ϕ0/v**2 + ɛ*ϕ1/v**2)

# Double Heston characteristic function
# Should generalise to multi
# def char_func(p,t,θ):
#     σ,ρ,v,κ = θ
#     i = 1j
#     σ1,σ2 = σ
#     ρ1,ρ2 = ρ
#     v1,v2 = v
#     κ1,κ2 = κ
#     θ01 = σ1*ρ1*v1*i*p
#     θ02 = σ2*ρ2*v2*i*p
#     # First factor
#     if κ1 == np.inf:
#         Φ1 = np.exp((1 - θ01 - np.sqrt((1 - θ01)**2 + σ1**2*v1**2*p*(i + p)))/v1**2*t)
#     else:
#         λ = κ1*v1*σ1
#         η = σ1**2
#         d = np.sqrt((ρ1*λ*i*p - κ1)**2 + λ**2*(i*p + p**2))
#         rm = κ1 - ρ1*λ*i*p - d
#         rp = κ1 - ρ1*λ*i*p + d
#         g = rm/rp
#         C = η*κ1/λ**2*(rm*t - 2*np.log((1 - g*np.exp(-d*t))/(1-g)))
#         D = σ1**2/λ**2*rm*(1 - np.exp(-d*t))/(1 - g*np.exp(-d*t))
#         Φ1 = np.exp(C + D)
#     # Second factor
#     if κ2 == np.inf:
#         Φ2 = np.exp((1 - θ02 - np.sqrt((1 - θ02)**2 + σ2**2*v2**2*p*(i + p)))/v2**2*t)
#     else:
#         λ = κ2*v2*σ2
#         η = σ2**2
#         d = np.sqrt((ρ2*λ*i*p - κ2)**2 + λ**2*(i*p + p**2))
#         rm = κ2 - ρ2*λ*i*p - d
#         rp = κ2 - ρ2*λ*i*p + d
#         g = rm/rp
#         C = η*κ2/λ**2*(rm*t - 2*np.log((1 - g*np.exp(-d*t))/(1-g)))
#         D = σ2**2/λ**2*rm*(1 - np.exp(-d*t))/(1 - g*np.exp(-d*t))
#         Φ2 = np.exp(C + D)
#     # Return product
#     return Φ1*Φ2

def ft_integrand(p,t,k,Θ):
    ρ = (np.exp(-1j*k*p)*char_func(p - .5j,t,Θ)/(p**2 + .25)).real
    return ρ

def ft_price(t,k,Θ,u=np.inf):
    # have changed to 2x
    p = 1 - 1/np.pi*np.exp(k/2)*quad(ft_integrand,0,u,args=(t,k,Θ))[0]
    return p


def pricer(T,k,Θ):
    m,n = k.shape
    p = [[ft_price(T[i],k[i,j],Θ) for j in range(n)] for i in range(m)]
    return np.array(p)

def bs_price(k,v):
    """
    k.shape = () : log-strike
    v.shape = () : total variance
    """
    σ = np.sqrt(v)
    d1 = -k/σ + 0.5*σ
    d2 = d1 - σ
    p = norm.cdf(d1) - np.exp(k)*norm.cdf(d2)
    return p
def obj_func(σ,k,t,p):
    """
    σ.shape = () : bs volatility
    k.shape = () : log-strike
    t.shape = () : years to maturity
    p.shape = () : option price
    """
    e = bs_price(k,σ**2*t) - p
    return e
def vol(k,t,p):
    """
    k.shape = () : log-strike
    t.shape = () : years to maturity
    p.shape = () : option price
    """
    p = np.maximum(p, np.maximum(1. - np.exp(k),0))
    σ = brentq(obj_func,1e-9,1e+9,args=(k,t,p))
    return σ
def surface(T,k,p):
    """
    k.shape = (m,n) : log-strikes
    t.shape = (m,) : years to maturity
    p.shape = (m,n) : option prices
    """
    m,n = k.shape
    σ = [[vol(k[i,j],T[i],p[i,j]) for j in range(n)] for i in range(m)]
    return np.array(σ)
