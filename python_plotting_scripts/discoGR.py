import numpy as np

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
