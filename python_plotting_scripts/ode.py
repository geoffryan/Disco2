import math
import numpy as np

def forward_euler(x, y, yprime, dx, *args, **kwargs):
    return y+dx*yprime(x, y, *args, **kwargs)

def rk2(x, y, yprime, dx, *args, **kwargs):
    k1 = yprime(x, y, *args, **kwargs)
    k2 = yprime(x + 0.5*dx, y + 0.5*dx*k1, *args, **kwargs)
    return y + dx*k2

def rk4(x, y, yprime, dx, *args, **kwargs):
    k1 = yprime(x, y, *args, **kwargs)
    k2 = yprime(x + 0.5*dx, y + 0.5*dx*k1, *args, **kwargs)
    k3 = yprime(x + 0.5*dx, y + 0.5*dx*k2, *args, **kwargs)
    k4 = yprime(x + dx, y + dx*k3, *args, **kwargs)
    return y + dx*(k1+2*k2+2*k3+k4)/6.0

def symp1(t, q, p, dTdP, dVdQ, dt, *args, **kwargs):
    dq = dTdP(t, q, p, *args, **kwargs)
    dp = -dVdQ(t, q+dt*dq, p, *args, **kwargs)
    return q + dt*dq, p + dt*dp

def symp2(t, q, p, dTdP, dVdQ, dt, *args, **kwargs):
    dq = dTdP(t, q, p, *args, **kwargs)
    q = q + 0.5*dt*dq
    dp = -dVdQ(t, q, p, *args, **kwargs)
    p = p + dt*dp
    dq = dTdP(t, q, p, *args, **kwargs)
    q = q + 0.5*dt*dq
    return q, p

def symp4(t, q, p, dTdP, dVdQ, dt, *args, **kwargs):
    c2 = math.pow(2.0,1.0/3.0)
    dq = dTdP(t, q, p, *args, **kwargs)
    q = q + dt*dq/(2.0*(2.0-c2))
    
    dp = -dVdQ(t, q, p, *args, **kwargs)
    p = p + dt*dp/(2.0-c2)    
    dq = dTdP(t, q, p, *args, **kwargs)
    q = q + dt*dq*(1.0-c2)/(2.0*(2.0-c2))
    
    dp = -dVdQ(t, q, p, *args, **kwargs)
    p = p - dt*dp*c2/(2.0-c2)    
    dq = dTdP(t, q, p, *args, **kwargs)
    q = q + dt*dq*(1.0-c2)/(2.0*(2.0-c2))
    
    dp = -dVdQ(t, q, p, *args, **kwargs)
    p = p + dt*dp/(2.0-c2)    
    dq = dTdP(t, q, p, *args, **kwargs)
    q = q + dt*dq/(2.0*(2.0-c2))
    
    return q, p

