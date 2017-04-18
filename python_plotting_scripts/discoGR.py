import numpy as np
import math

# 0: Flat, cyl coords
# 1: Schwarzschild, schwarzschild coords
# 2: Schwarzschild, Kerr-Schild coords
# 3: Flat, cartesian coords
# 4: Kerr, Kerr-Schild coords
# 5: Flat, cyl coords, ADM
# 6: Schw, Kerr-Schild coords, ADM


def calc_g(r, pars):
    M = pars['GravM']
    a = pars['GravA']
    bw = pars['BinW']
    A = M*a

    g00 = np.zeros(r.shape)
    g0r = np.zeros(r.shape)
    g0p = np.zeros(r.shape)
    grr = np.zeros(r.shape)
    grp = np.zeros(r.shape)
    gpp = np.zeros(r.shape)

    if pars['Metric'] == 0:
        g00[:] = -1.0
        g0r[:] = 0.0
        g0p[:] = 0.0
        grr[:] = 1.0
        grp[:] = 0.0
        gpp[:] = r*r

    elif pars['Metric'] == 1:
        g00[:] = -1.0 + 2*M/r
        g0r[:] = 0.0
        g0p[:] = 0.0
        grr[:] = 1.0/(1.0 - 2*M/r)
        grp[:] = 0.0
        gpp[:] = r*r

    elif pars['Metric'] == 2:
        g00[:] = -1.0 + 2*M/r
        g0r[:] = 2*M/r
        g0p[:] = 0.0
        grr[:] = 1.0 + 2*M/r
        grp[:] = 0.0
        gpp[:] = r*r

    elif pars['Metric'] == 3:
        g00[:] = -1.0
        g0r[:] = 0.0
        g0p[:] = 0.0
        grr[:] = 1.0
        grp[:] = 0.0
        gpp[:] = 1.0

    elif pars['Metric'] == 6:
        g00[:] = -1.0 + 2*M/r + r*r*bw*bw
        g0r[:] = 2*M/r
        g0p[:] = r*r*bw
        grr[:] = 1.0 + 2*M/r
        grp[:] = 0.0
        gpp[:] = r*r

    return g00, g0r, g0p, grr, grp, gpp

def calc_shift(r, pars):
    M = pars['GravM']
    a = pars['GravA']
    bw = pars['BinW']
    A = M*a

    br = np.zeros(r.shape)
    bp = np.zeros(r.shape)

    if pars['Metric'] == 0:
        br = 0.0
        bp = 0.0

    elif pars['Metric'] == 1:
        br = 0.0
        bp = 0.0

    elif pars['Metric'] == 2:
        br = 2*M/(r + 2*M)
        bp = 0.0

    elif pars['Metric'] == 3:
        br = 0.0
        bp = 0.0

    elif pars['Metric'] == 6:
        br = 2*M/(r+2*M)
        bp = bw

    return br, bp

def calc_igam(r, pars):
    M = pars['GravM']
    a = pars['GravA']
    bw = pars['BinW']
    A = M*a

    igamrr = np.zeros(r.shape)
    igamrp = np.zeros(r.shape)
    igampp = np.zeros(r.shape)

    if pars['Metric'] == 0:
        igamrr = 1.0
        igamrp = 0.0
        igampp = 1.0/(r*r)

    elif pars['Metric'] == 1:
        igamrr = 1.0 - 2*M/r
        igamrp = 0.0
        igampp = 1.0/(r*r)


    elif pars['Metric'] == 2:
        igamrr = 1.0/(1.0+2*M/r)
        igamrp = 0.0
        igampp = 1.0/(r*r)

    elif pars['Metric'] == 3:
        igamrr = 1.0
        igamrp = 0.0
        igampp = 1.0

    elif pars['Metric'] == 6:
        igamrr = 1.0/(1.0+2*M/r)
        igamrp = 0.0
        igampp = 1.0/(r*r)

    return igamrr, igamrp, igampp

def calc_u(r, vr, vp, pars):

    M = pars['GravM']
    a = pars['GravA']
    bm = pars['BinM']
    ba = pars['BinA']
    bw = pars['BinW']
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
    elif pars['Metric'] == 5:
        if pars['BoostType'] == 0:
            u0 = 1.0 / np.sqrt(1.0 - vr*vr - r*r*vp*vp)
        elif pars['BoostType'] == 1:
            u0 = 1.0 / np.sqrt(1.0 - vr*vr - r*r*(vp+bw)*(vp+bw))
    elif pars['Metric'] == 6:
        if pars['BoostType'] == 0:
            u0 = 1.0 / np.sqrt(1.0 - 2*M/r - 4*M/r*vr - (1+2*M/r)*vr*vr
                            - r*r*vp*vp)
        elif pars['BoostType'] == 1:
            u0 = 1.0 / np.sqrt(1.0 - 2*M/r - 4*M/r*vr - (1+2*M/r)*vr*vr
                            - r*r*(vp+bw)*(vp+bw))

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

    elif pars['Metric'] == 5:
        al[:] = 1.0

    elif pars['Metric'] == 6:
        al = 1.0 / np.sqrt(1.0+2.0*M/r)

    return al

def isco(pars):
    
    M = pars['GravM']
    a = pars['GravA']
    A = M*a

    if pars['Metric'] == 0:
        Risco = None

    elif pars['Metric'] == 1:
        Risco = 6*M
    
    elif pars['Metric'] == 2:
        Risco = 6*M

    elif pars['Metric'] == 3:
        Risco = None

    elif pars['Metric'] == 4:
        Z1 = 1. + math.pow((1-a*a)*(1+a),1./3) + math.pow((1-a*a)*(1-a),1./3)
        Z2 = math.sqrt(3*a*a + Z1*Z1)
        if a > 0.0:
            Risco = M*(3.0 + Z2 - math.sqrt((3.0-Z1)*(3.0+Z1+2*Z2)))
        else:
            Risco = M*(3.0 + Z2 + math.sqrt((3.0-Z1)*(3.0+Z1+2*Z2)))

    elif pars['Metric'] == 5:
        Risco = None

    elif pars['Metric'] == 6:
        Risco = 6*M

    return Risco

def ergo(pars):
    
    M = pars['GravM']
    a = pars['GravA']
    A = M*a

    if pars['Metric'] == 0:
        Rergo = None

    elif pars['Metric'] == 1:
        Rergo = None
    
    elif pars['Metric'] == 2:
        Rergo = None

    elif pars['Metric'] == 3:
        Rergo = None

    elif pars['Metric'] == 4:
        Rergo = 2*M

    elif pars['Metric'] == 5:
        Rergo = None

    elif pars['Metric'] == 6:
        Rergo = None

    return Rergo

def horizon(pars):
    
    M = pars['GravM']
    a = pars['GravA']
    A = M*a

    if pars['Metric'] == 0:
        R = None

    elif pars['Metric'] == 1:
        Reh = 2*M
    
    elif pars['Metric'] == 2:
        Reh = 2*M

    elif pars['Metric'] == 3:
        Reh = None

    elif pars['Metric'] == 4:
        Reh = M*(1+math.sqrt((1-a)*(1+a)))

    elif pars['Metric'] == 5:
        Reh = None

    elif pars['Metric'] == 6:
        Reh = 2*M

    return Reh

def calc_geo_U(r, pars):
    
    M = pars['GravM']
    a = pars['GravA']
    A = M*a

    u0 = np.empty(r.shape)
    ur = np.empty(r.shape)
    up = np.empty(r.shape)

    Risco = isco(pars)
    inns = r < Risco
    outs = r > Risco
    ri = r[inns]
    ro = r[outs]

    if pars['Metric'] == 0:
        u0[:] = 1.0
        ur[:] = 0.0
        up[:] = 0.0
    elif pars['Metric'] == 3:
        u0[:] = 1.0
        ur[:] = 0.0
        up[:] = 0.0
    elif pars['Metric'] == 1:
        u0o = 1.0/np.sqrt(1-3*M/ro)
        uro = np.zeros(ro.shape)
        upo = u0o * np.sqrt(M/(ro*ro*ro))
        u0i = 2*math.sqrt(2.)/3. / (1 - 2*M/ri)
        x = Risco/ri - 1.0
        uri = -np.sqrt(x*x*x)/3.0
        upi = 2.*math.sqrt(3.0)*M / (ri*ri)
        u0[inns] = u0i
        ur[inns] = uri
        up[inns] = upi
        u0[outs] = u0o
        ur[outs] = uro
        up[outs] = upo
    elif pars['Metric'] == 2:
        u0o = 1.0/np.sqrt(1-3*M/ro)
        uro = np.zeros(ro.shape)
        upo = u0o * np.sqrt(M/(ro*ro*ro))
        x = Risco/ri - 1.0
        u0i = 2.*(math.sqrt(2.)*ri - M*np.sqrt(x*x*x)) / (3.*(ri - 2*M))
        uri = -np.sqrt(x*x*x)/3.0
        upi = 2.*math.sqrt(3.0)*M / (ri*ri)
        u0[inns] = u0i
        ur[inns] = uri
        up[inns] = upi
        u0[outs] = u0o
        ur[outs] = uro
        up[outs] = upo
    elif pars['Metric'] == 4:
        u0o = (ro + A*np.sqrt(M/ro)) / np.sqrt(ro*ro - 3*M*ro
                                                + 2*A*np.sqrt(M*ro))
        uro = np.zeros(ro.shape)
        omk = np.sqrt(M/(ro*ro*ro))
        upo = u0o * omk / (1+A*omk)

        OMK = math.sqrt(M/(Risco*Risco*Risco))
        U0 = (1. + A*OMK) / math.sqrt(1. - 3*M/Risco + 2*A*OMK)
        UP = U0 * OMK / (1. + A*OMK)
        eps = (-1. + 2*M/Risco) * U0 + (-2*M*A/Risco) * UP
        lll = (-2*M*A/Risco) * U0 + (Risco*Risco + A*A + 2*M*A*A/Risco) * UP

        x = Risco/ri - 1.0
        u0i = 2.*(math.sqrt(2.)*ri - M*np.sqrt(x*x*x)) / (3.*(ri - 2*M))
        uri = -np.sqrt(x*x*x)/3.0
        upi = 2.*math.sqrt(3.0)*M / (ri*ri)
        u0[inns] = u0i
        ur[inns] = uri
        up[inns] = upi
        u0[outs] = u0o
        ur[outs] = uro
        up[outs] = upo

    return u0, ur, up

