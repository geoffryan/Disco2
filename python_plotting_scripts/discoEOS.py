import numpy as np

c = 2.99792458e10
G = 6.6738e-8
h = 6.62606957e-27
kb = 1.3806488e-16
sb = 1.56055371e59
mp = 1.672621777e-24
rg_solar = 1.4766250385e5
M_solar = 1.9884e33

def P_gas(rho, T, pars):
    eos_x1 = pars['EOSPar1']
    return eos_x1 * rho * T

def P_rad(rho, T, pars):
    eos_x2 = pars['EOSPar2']
    return eos_x2 * 4.0*sb/(3.0*c) * (T*mp*c*c)**4 / (c*c)

def P_deg(rho, T, pars):
    eos_x3 = pars['EOSPar3']
    return eos_x3 * 2*np.pi*h*c/3.0 * np.power(3*rho/(8*np.pi*mp),4.0/3.0) \
            / (c*c)

def e_gas(rho, T, pars):
    GAM = pars['Adiabatic_Index']
    eos_x1 = pars['EOSPar1']
    return eos_x1 * T / (GAM-1.0)

def e_rad(rho, T, pars):
    eos_x2 = pars['EOSPar2']
    return eos_x2 * 4.0*sb * (T*mp*c*c)**4 / (c*rho*c*c)

def e_deg(rho, T, pars):
    eos_x3 = pars['EOSPar3']
    return eos_x3 * h*c/(4.0*mp) * np.power(3*rho/(8*np.pi*mp),1.0/3.0) / (c*c)

def dPdr(rho, T, pars):
    eos_x1 = pars['EOSPar1']
    eos_x2 = pars['EOSPar2']
    eos_x3 = pars['EOSPar3']

    dPdr_gas = eos_x1 * T
    dPdr_rad = eos_x2 * 0.0
    dPdr_deg = eos_x3 * h*c/(3*mp)*np.power(3*rho/(8*np.pi*mp),1.0/3.0)/(c*c)

    if pars['EOSType'] == 0:
        der = dPdr_gas
    elif pars['EOSType'] == 1:
        der = dPdr_gas + dPdr_rad
    elif pars['EOSType'] == 2:
        der = dPdr_gas + dPdr_rad + dPdr_deg
    else:
        der = np.zeros(rho.shape)

    return der

def dPdT(rho, T, pars):
    eos_x1 = pars['EOSPar1']
    eos_x2 = pars['EOSPar2']
    eos_x3 = pars['EOSPar3']

    dPdT_gas = eos_x1 * rho
    dPdT_rad = eos_x2 * 16.0*sb/(3.0*c) * (T*mp*c*c)**3 * mp
    dPdT_deg = eos_x3 * 0.0

    if pars['EOSType'] == 0:
        der = dPdT_gas
    elif pars['EOSType'] == 1:
        der = dPdT_gas + dPdT_rad
    elif pars['EOSType'] == 2:
        der = dPdT_gas + dPdT_rad + dPdT_deg
    else:
        der = np.zeros(rho.shape)

    return der

def dedr(rho, T, pars):
    eos_x1 = pars['EOSPar1']
    eos_x2 = pars['EOSPar2']
    eos_x3 = pars['EOSPar3']

    dedr_gas = eos_x1 * 0.0
    dedr_rad = eos_x2 * -4.0*sb * (T*mp*c*c)**4 / (c*rho*rho*c*c)
    dedr_deg = eos_x3 * (3*h*c/(32*np.pi*mp*mp)
                    * np.power(3*rho/(8*np.pi*mp),-2.0/3.0) / (c*c))

    if pars['EOSType'] == 0:
        der = dedr_gas
    elif pars['EOSType'] == 1:
        der = dedr_gas + dedr_rad
    elif pars['EOSType'] == 2:
        der = dedr_gas + dedr_rad + dedr_deg
    else:
        der = np.zeros(rho.shape)

    return der

def dedT(rho, T, pars):
    eos_x1 = pars['EOSPar1']
    eos_x2 = pars['EOSPar2']
    eos_x3 = pars['EOSPar3']
    GAM = pars['Adiabatic_Index']

    dedT_gas = eos_x1 * c*c/(GAM-1.0)
    dedT_rad = eos_x2 * 16.0*sb*(T*mp*c*c)**3 * mp/(c*rho)
    dedT_deg = eos_x3 * 0.0

    if pars['EOSType'] == 0:
        der = dedT_gas
    elif pars['EOSType'] == 1:
        der = dedT_gas + dedT_rad
    elif pars['EOSType'] == 2:
        der = dedT_gas + dedT_rad + dedT_deg
    else:
        der = np.zeros(rho.shape)

    return der

def ppp(rho, T, pars):
    if pars['EOSType'] == 0:
        P = P_gas(rho,T,pars)
    elif pars['EOSType'] == 1:
        P = P_gas(rho,T,pars) + P_rad(rho,T,pars)
    elif pars['EOSType'] == 2:
        P = P_gas(rho,T,pars) + P_rad(rho,T,pars) + P_deg(rho,T,pars)
    else:
        P = np.zeros(rho.shape)
    return P

def eps(rho, T, pars):
    if pars['EOSType'] == 0:
        e = e_gas(rho,T,pars)
    elif pars['EOSType'] == 1:
        e = e_gas(rho,T,pars) + e_rad(rho,T,pars)
    elif pars['EOSType'] == 2:
        e = e_gas(rho,T,pars) + e_rad(rho,T,pars) + e_deg(rho,T,pars)
    else:
        e = np.zeros(rho.shape)
    return e

def cs2(rho, T, pars):
    P = ppp(rho, T, pars)
    e = eps(rho, T, pars)
    enth = 1.0 + e + P/rho
    return (dPdr(rho,T,pars)*dedT(rho,T,pars)
            - dedr(rho,T,pars)*dPdT(rho,T,pars)
            + P*dPdT(rho,T,pars)/(rho*rho)) / (dedT(rho,T,pars)*enth)

