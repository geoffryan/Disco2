import numpy as np

# 0: Flat, cyl coords
# 1: Schwarzschild, schwarzschild coords
# 2: Schwarzschild, Kerr-Schild coords
# 3: Flat, cartesian coords
# 4: Kerr, Kerr-Schild coords

def calc_u(r, vr, vp, pars):

    M = pars['GravM']
    a = pars['GravA']
    A = M*a

    u0 = np.zeros(r.shape)

    if pars['Metric'] == 0:
        u0 = 1.0 / np.sqrt(1.0 - vr*vr - r*r*vp*vp) 

    elif pars['Metric'] == 1:
        u0 = 1.0 / np.sqrt(1.0 - 2*M/r - vr*vr/(1-2*M/r) - r*r*vp*vp)
    
    elif pars['Metric'] == 2:
        u0 = 1.0 / np.sqrt(1.0 - 2*M/r - 4*M/r*vr - (1+2*M/r)*vr*vr
                            - r*r*vp*vp)

    elif pars['Metric'] == 3:
        u0 = 1.0 / np.sqrt(1.0 - vr*vr - vp*vp)

    elif pars['Metric'] == 4:
        u0 = 1.0 / np.sqrt(1.0 - 2*M/r - 4*M/r*vr + 4*M*A/r*vp
                            - (1+2*M/r)*vr*vr + 2*(1+2*M/r)*A*vr*vp
                            - (r*r + A*A + 2*M*A*A/r)*vp*vp)

    return u0, u0*vr, u0*vp

def lapse(r, pars):

    M = pars['GravM']
    a = pars['GravA']
    A = M*a

    al = np.zeros(r.shape)

    if pars['Metric'] == 0:
        al[:] = 1.0

    elif pars['Metric'] == 1:
        al = np.sqrt(1.0-2*M/r)
    
    elif pars['Metric'] == 2:
        al = 1.0 / np.sqrt(1.0+2.0*M/r)

    elif pars['Metric'] == 3:
        al[:] = 1.0

    elif pars['Metric'] == 4:
        al = 1.0 / np.sqrt(1.0+2.0*M/r)

    return al
