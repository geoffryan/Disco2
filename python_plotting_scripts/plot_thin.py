import sys
import math
import discopy as dp
import numpy as np
import matplotlib.pyplot as plt

# All constants in c.g.s.
sb = 1.56055371e59
c = 2.99792458e10
mp = 1.672621777e-24
h = 6.62606957e-27
ka_bbes = 0.2
rg_solar = 1.4766250385e5
r_scale = rg_solar
rho_scale = 1.0

nu = np.logspace(14,20,num=7,base=10.0)

blue = (31.0/255, 119.0/255, 180.0/255)
orange = (255.0/255, 127.0/255, 14.0/255)
green = (44.0/255, 160.0/255, 44.0/255)
red = (214.0/255, 39.0/255, 40.0/255)
purple = (148.0/255, 103.0/255, 189.0/255)

def P_gas(g, rho, T):
    return rho * T

def P_rad(g, rho, T):
    return 4.0*sb/(3.0*c) * (T*mp*c*c)**4 / (c*c)

def P_deg(g, rho, T):
    return 2*np.pi*h*c/3.0 * np.power(3*rho/(8*np.pi*mp),4.0/3.0) / (c*c)

def e_gas(g, rho, T):
    GAM = g._pars['Adiabatic_Index']
    return T / (GAM-1.0)

def e_rad(g, rho, T):
    return 4.0*sb * (T*mp*c*c)**4 / (c*rho*c*c)

def e_deg(g, rho, T):
    return h*c/(4.0*mp) * np.power(3*rho/(8*np.pi*mp),1.0/3.0) / (c*c)

def ppp(g, rho, T):
    EOSType = g._pars['EOSType']
    EOSPar1 = g._pars['EOSPar1']
    EOSPar2 = g._pars['EOSPar2']
    EOSPar3 = g._pars['EOSPar3']

    P = np.zeros(rho.shape)

    if EOSType == 0:
        P = P_gas(g, rho, T)
    elif EOSType == 1:
        P = EOSPar1 * P_gas(g, rho, T) + EOSPar2 * P_rad(g, rho, T)
    else:
        P = EOSPar1*P_gas(g,rho,T) + EOSPar2*P_rad(g,rho,T)\
            + EOSPar3*P_deg(g,rho,T)

    return P

def eps(g, rho, T):
    EOSType = g._pars['EOSType']
    EOSPar1 = g._pars['EOSPar1']
    EOSPar2 = g._pars['EOSPar2']
    EOSPar3 = g._pars['EOSPar3']

    e = np.zeros(rho.shape)

    if EOSType == 0:
        e = e_gas(g, rho, T)
    elif EOSType == 1:
        e = EOSPar1 * e_gas(g, rho, T) + EOSPar2 * e_rad(g, rho, T)
    else:
        e = EOSPar1*e_gas(g,rho,T) + EOSPar2*e_rad(g,rho,T)\
            + EOSPar3*e_deg(g,rho,T)

    return e

def gToArr1D(g):
    R = np.zeros(g.nr_tot)
    prim = np.zeros((g.nr_tot,5))

    ind = 0
    for i in xrange(g.nr_tot):
        r1 = g.rFaces[i]
        r2 = g.rFaces[i+1]
        r = 0.5*(r1+r2)

        R[i] = r
        prim[i,:] = g.prim[0][i][0,:5]

    return R, prim

def gToArr(g):
    R = np.zeros(g.np.sum())
    prim = np.zeros((g.np.sum(),5))

    ind = 0
    for k in xrange(g.nz_tot):
        for i in xrange(g.nr_tot):
            r1 = g.rFaces[i]
            r2 = g.rFaces[i+1]
            r = 0.5*(r1+r2)

            numP = g.np[k,i]

            R[ind:ind+numP] = r
            prim[ind:ind+numP,:] = g.prim[k][i][:,:5]

            ind += numP

    return R, prim

def ISCO(M, a):

    Z1 = 1.0 + math.pow((1-a*a)*(1+a),1.0/3.0) \
            + math.pow((1-a*a)*(1-a),1.0/3.0)
    Z2 = math.sqrt(3*a*a + Z1*Z1)
    if a > 0:
        Risco = M*(3.0 + Z2 - math.sqrt((3.0-Z1)*(3.0+Z1+2*Z2)))
    else:
        Risco = M*(3.0 + Z2 + math.sqrt((3.0-Z1)*(3.0+Z1+2*Z2)))

    return Risco

def EH(M, a):

    return M * (1.0 + math.sqrt(1-a*a))

def allTheThings(g, r, prim):

    M = g._pars['GravM']
    a = g._pars['GravA']
    BG = g._pars['Background']

    if BG == 3:
        rho = prim[:,0]
        T = prim[:,1]
        vr = prim[:,2]
        vp = prim[:,3]

        u0 = 1.0 / np.sqrt(1 - 2*M/r - 4*M/r*vr + 4*M*a/r*vp 
                - (1+2*M/r)*vr*vr + 2*(1+2*M/r)*a*vr*vp
                - (r*r+a*a+2*M*a*a/r)*vp*vp)
        p = ppp(g, rho, T)
        e = eps(g, rho, T)
        rhoh = rho + rho*e + p

        H = np.sqrt(r*r*r*p/(M*rhoh)) / u0
        sig = rho*H
        pi = p*H
        mdot = -2*np.pi*r*sig*u0*vr
        
        qdot = 8*sb * (T*mp*c*c)**4 / (3*ka_bbes*sig * c*c*c*rho_scale*r_scale)


    else:
        sig = prim[:,0]
        pi = prim[:,1]
        vr = prim[:,2]
        vp = prim[:,3]
        
        u0 = 1.0 / np.sqrt(1 - 2*M/r - 4*M/r*vr + 4*M*M*a/r*vp 
                - (1+2*M/r)*vr*vr + 2*(1+2*M/r)*M*a*vr*vp
                - (r*r+a*a*M*M+2*M*M*M*a*a/r)*vp*vp)
        mdot = -2*np.pi*r*sig*u0*vr

        H = np.sqrt(r*r*r*pi/(M*sig)) / u0
        rho = sig/H
        p = pi/H

        T = mp*c*c * pi/sig
        qdot = 8*sb * T*T*T*T / (3*ka_bbes*sig * c*c*c*rho_scale*r_scale)

    return rho, sig, p, pi, H, T, vr, vp, mdot, qdot

def calcNT(g, r, rs, Mdot):

    M = g._pars['GravM']
    a = g._pars['GravA']
    GAM = g._pars['Adiabatic_Index']
    AL = g._pars['AlphaVisc']

    EOSType = g._pars['EOSType']
    EOSPar2 = g._pars['EOSPar2']

    rad = False
    if EOSType == 1 and EOSPar2 == 1.0:
        rad = True

    A = M*a
    OMK = np.sqrt(M/(r*r*r))

    B = 1.0 + A*OMK
    C = 1.0 - 3.0*M/r + 2*A*OMK
    D = 1.0 - 2.0*M/r + A*A/(r*r)
    if A == 0:
        P = 1.0 - np.sqrt(rs/r) + np.sqrt(3*M/r)*(np.arctanh(np.sqrt(3*M/r))
                                            - np.arctanh(np.sqrt(3*M/rs)))
    else:
        th = math.acos(a) / 3.0
        ct = math.cos(th)
        st = math.sin(th)
        x1 = ct - math.sqrt(3.0)*st
        x2 = ct + math.sqrt(3.0)*st
        x3 = -2*ct
        c0 = -a*a/(x1*x2*x3)
        c1 = (x1-a)*(x1-a) / (x1*(x2-x1)*(x3-x1))
        c2 = (x2-a)*(x2-a) / (x2*(x1-x2)*(x3-x2))
        c3 = (x3-a)*(x3-a) / (x3*(x1-x3)*(x2-x3))

        x = np.sqrt(r/M)
        xs = math.sqrt(rs/M)

        #fig, ax = plt.subplots()
        P = 1.0 - xs/x - 3/x * (c0 * np.log(x/xs)
                                + c1 * np.log((x-x1)/(xs-x1))
                                + c2 * np.log((x-x2)/(xs-x2))
                                + c3 * np.log((x-x3)/(xs-x3)))
        #ax.plot(r, P)
        #P = 1.0 - np.sqrt(rs/r) + np.sqrt(3*M/r)*(np.arctanh(np.sqrt(3*M/r))
        #                                    - np.arctanh(np.sqrt(3*M/rs)))
        #ax.plot(r, P)
        #plt.show()


    U0 = B/np.sqrt(C)

    pi = Mdot / (3*np.pi*AL*math.sqrt(GAM)) * OMK
    PI = Mdot / (3*np.pi*AL*math.sqrt(GAM)) * OMK * np.sqrt(C) * P / (D*D)

    Qdot = 3*Mdot/(4*np.pi) * OMK*OMK
    QDOT = 3*Mdot/(4*np.pi) * OMK*OMK * P / C

    sig = np.power(8.0*sb/(3*ka_bbes*Qdot) * (pi*mp*c*c)**4
                    / (c*c*c*r_scale*rho_scale), 0.2)
    SIG = np.power(8.0*sb/(3*ka_bbes*QDOT) * (PI*mp*c*c)**4
                    / (c*c*c*r_scale*rho_scale), 0.2)
    vr = -Mdot / (2*np.pi*r*sig)
    VR = -Mdot / (2*np.pi*r*SIG * U0)

    vp = OMK
    VP = OMK / B

    if rad:
#        SIGrad = 8.0 * (PI*rho_scale*c*c*r_scale)\
#                / (ka_bbes * QDOT*rho_scale*c*c*c) / (rho_scale*r_scale)
        SIGrad = 8.0 * (PI) / (ka_bbes * rho_scale*r_scale * QDOT)
        VRrad = -Mdot / (2*np.pi*r*SIGrad * U0)
        
        return (sig, pi, vr, vp, Qdot), (SIG, PI, VR, VP, QDOT), \
                (SIGrad, PI, VRrad, VP, QDOT)

    return (sig, pi, vr, vp, Qdot), (SIG, PI, VR, VP, QDOT)

def calcGEO(g, r, Mdot, K):

    M = g._pars['GravM']
    A = g._pars['GravA']
    GAM = g._pars['Adiabatic_Index']

    if A == 0:
        ur = -np.power(6*M/r - 1.0, 1.5) / 3.0
        up = 2*math.sqrt(3.0)*M/(r*r)
        u0 = (-2*M/r*ur - np.sqrt(4*M*M/(r*r)*ur*ur - (-1+2.0*M/r) * (1.0+(1.0+2.0*M/r)*ur*ur + r*r*up*up))) / (-1+2*M/r)
    
    else:

        Risco = ISCO(M, A)

        OMK = math.sqrt(M/(Risco*Risco*Risco))
        U0 = (1.0 + A*M*OMK) / math.sqrt(1.0 - 3*M/Risco + 2*A*M*OMK)
        UP = OMK / math.sqrt(1.0 - 3*M/Risco + 2*A*M*OMK)

        EPS = -(1-2*M/Risco)*U0 - 2*M*M*A/Risco*UP
        LLL = -2*M*M*A/Risco*U0 + (Risco*Risco+M*M*A*A+2*M*M*M*A*A/Risco)*UP

        #Inverse Metric components
        g00 = -1-2*M/r
        g0r = 2*M/r
        g0p = np.zeros(r.shape)
        grr = 1 - 2*M/r + A*A*M*M/(r*r)
        grp = A*M/(r*r)
        gpp = 1.0/(r*r)

        u0d = EPS*np.ones(r.shape)
        upd = LLL*np.ones(r.shape)
        aa = grr
        bb = g0r*u0d + grp*upd
        cc = g00*u0d*u0d + 2*g0p*u0d*upd + gpp*upd*upd + 1.0
        urd = (-bb - np.sqrt(bb*bb - aa*cc)) / aa

        u0 = g00*u0d + g0r*urd + g0p*upd
        ur = g0r*u0d + grr*urd + grp*upd
        up = g0p*u0d + grp*urd + gpp*upd


    vr = ur/u0
    vp = up/u0

    sig = -Mdot / (2*math.pi*r*ur)

    pi = K * np.power(sig, GAM)

    T = mp*c*c * pi/sig
    qdot = 8*sb * T*T*T*T / (3*ka_bbes*sig * c*c*c*rho_scale*r_scale)

    return sig, pi, vr, vp, qdot

def plotSigNice(g):

    r, prim = gToArr1D(g)

    rho, sig, p, pi, H, T, vr, vp, mdot, qdot = allTheThings(g, r, prim)

    M = g._pars['GravM']
    a = g._pars['GravA']
    EOSType = g._pars['EOSType']
    EOSPar2 = g._pars['EOSPar2']
    rad = False
    if EOSType == 1 and EOSPar2 == 1.0:
        rad = True

    Risco = ISCO(M,a)
    Reh = EH(M,a)

    R = np.logspace(math.log10(Risco)+0.01, math.log10(r.max()), base=10.0, num=200)
    R2 = np.logspace(math.log10(r.min()), math.log10(Risco)-0.05, base=10.0, num=200)

    RS = Risco
    MDOT = 209
    K = 3.5e-5

    labelsize = 24
    ticksize = 18
    titlesize = 36
    legendsize = 24

    if rad:
        SSdat, NTdat, NTraddat = calcNT(g, R, RS, MDOT)
    else:
        SSdat, NTdat = calcNT(g, R, RS, MDOT)

    GEOdat = calcGEO(g, R2, MDOT, K)

    fig, ax = plt.subplots(1,1, figsize=(12,9))

    xlim = (1.0, 1.0e2)

    real_units = rho_scale*r_scale

    ax.axvspan(xlim[0], 2*M, color='lightgrey', alpha=0.5, 
                label='Event Horizon')
    ax.axvspan(xlim[0], Reh, color='grey', alpha=0.5, 
                label='Ergosphere')
    ax.axvline(Risco, ls='--', lw=5.0, color='lightgrey',
                label='ISCO')

    ax.plot(R2, GEOdat[0]*real_units, ls='-', lw=5.0, color=orange, 
            label='Plunging')
    ax.plot(R, NTdat[0]*real_units, ls='-', lw=5.0, color=blue, 
            label='Novikov-Thorne')
    ax.plot(r, sig * real_units, 'k+', ms=10, label='Disco')
    ax.set_xlabel(r"$r$ $(GM_\odot / c^2)$", fontsize=labelsize)
    ax.set_ylabel(r"$\Sigma$ $(g/cm^2)$", fontsize=labelsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(labelsize=ticksize)
    ax.set_xlim(xlim)
    plt.legend(loc="lower right", fontsize=legendsize)

    title = ax.set_title(r"$M=M_\odot$   $a=0.9$   $\mathcal{M}_{ISCO}=20$", 
            fontsize=titlesize)
    title.set_position((0.5,1.02))

    return fig, ax

def plotQdotNice(g):

    r, prim = gToArr1D(g)

    rho, sig, p, pi, H, T, vr, vp, mdot, qdot = allTheThings(g, r, prim)

    M = g._pars['GravM']
    a = g._pars['GravA']
    EOSType = g._pars['EOSType']
    EOSPar2 = g._pars['EOSPar2']
    rad = False
    if EOSType == 1 and EOSPar2 == 1.0:
        rad = True

    Risco = ISCO(M,a)
    Reh = EH(M,a)

    R = np.logspace(math.log10(Risco)+0.01, math.log10(r.max()), base=10.0, num=200)
    R2 = np.logspace(math.log10(r.min()), math.log10(Risco)-0.05, base=10.0, num=200)

    RS = Risco
    MDOT = 209
    K = 3.5e-5

    labelsize = 24
    ticksize = 18
    titlesize = 36
    legendsize = 18

    if rad:
        SSdat, NTdat, NTraddat = calcNT(g, R, RS, MDOT)
    else:
        SSdat, NTdat = calcNT(g, R, RS, MDOT)

    GEOdat = calcGEO(g, R2, MDOT, K)

    if a != 0.0:
        g._pars['GravA'] = 0.0
        R3 = np.logspace(math.log10(6*M)+0.01, math.log10(r.max()), base=10.0, num=200)
        SSdat2, NTdat2 = calcNT(g, R3, 6*M, MDOT)
        g._pars['GravA'] = a

    fig, ax = plt.subplots(1,1, figsize=(12,9))

    xlim = (1.0, 1.0e2)

    real_units = rho_scale*c*c*c

    ax.axvspan(xlim[0], 2*M, color='lightgrey', alpha=0.5, 
                label='Event Horizon')
    ax.axvspan(xlim[0], Reh, color='grey', alpha=0.5, 
                label='Ergosphere')
    ax.axvline(Risco, ls='--', lw=5.0, color='lightgrey',
                label='ISCO')

    ax.plot(R2, GEOdat[4]*real_units, ls='-', lw=5.0, color=orange, 
            label='Plunging')
    ax.plot(R, NTdat[4]*real_units, ls='-', lw=5.0, color=blue, 
            label='Novikov-Thorne')
    if a != 0.0:
        ax.plot(R3, NTdat2[4]*real_units, ls='-', lw=5.0, color=green, 
                label='Novikov-Thorne $a=0$')
    ax.plot(r, qdot * real_units, 'k+', ms=10, label='Disco')
    ax.set_xlabel(r"$r$ $(GM_\odot / c^2)$", fontsize=labelsize)
    ax.set_ylabel(r"$\dot{Q}$ $(erg/cm^2 s)$", fontsize=labelsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(labelsize=ticksize)
    ax.set_xlim(xlim)
    plt.legend(loc="lower right", fontsize=legendsize)

    title = ax.set_title(r"$M=M_\odot$   $a=0.9$   $\mathcal{M}_{ISCO}=20$", 
            fontsize=titlesize)
    title.set_position((0.5,1.02))

    return fig, ax

def plotHeightProfile(g):

    r, prim = gToArr1D(g)

    rho, sig, p, pi, H, T, vr, vp, mdot, qdot = allTheThings(g, r, prim)

    M = g._pars['GravM']
    a = g._pars['GravA']
    EOSType = g._pars['EOSType']
    EOSPar2 = g._pars['EOSPar2']
    rad = False
    if EOSType == 1 and EOSPar2 == 1.0:
        rad = True

    Risco = ISCO(M,a)
    Reh = EH(M,a)

    labelsize = 24
    ticksize = 18
    titlesize = 36
    legendsize = 18

    fig, ax = plt.subplots(1,1, figsize=(12,9))

    xlim = (1.0, 3)

    ax.axvspan(xlim[0], 2*M, color='lightgrey', alpha=0.5, 
                label='Event Horizon')
    ax.axvspan(xlim[0], Reh, color='grey', alpha=0.5, 
                label='Ergosphere')
    ax.axvline(Risco, ls='--', lw=5.0, color='lightgrey',
                label='ISCO')

    ax.fill_between(r, -H, H)


    ax.set_xlabel(r"$r$ $(GM_\odot / c^2)$", fontsize=labelsize)
    ax.set_ylabel(r"$H$ $(GM_\odot / c^2)$", fontsize=labelsize)
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.tick_params(labelsize=ticksize)
    ax.set_xlim(xlim)
    ax.set_ylim((-xlim[1],xlim[1]))
    ax.set_aspect("equal")
    plt.legend(loc="lower right", fontsize=legendsize)

    title = ax.set_title(r"$M=M_\odot$   $a=0.9$   $\mathcal{M}_{ISCO}=20$", 
            fontsize=titlesize)
    title.set_position((0.5,1.02))

    return fig, ax

def plot_shear(g):
    
    R, prim = gToArr1D(g)
    rho, sig, p, pi, H, T, VR, VP, mdot, qdot = allTheThings(g, R, prim)

    r = 0.5*(R[:-1]+R[1:])
    vr = 0.5*(VR[:-1]+VR[1:])
    vp = 0.5*(VP[:-1]+VP[1:])

    M = g._pars['GravM']
    a = g._pars['GravA']

    g00 = -1 + 2*M/r
    g0r = 2*M/r
    g0p = -2*M*M*a/r
    grr = 1+2*M/r
    grp = -a*M*(1+2*M/r)
    gpp = r*r + a*a*M*M + 2*M*M*M*a*a/r

    ig00 = -1 - 2*M/r
    ig0r = 2*M/r
    ig0p = 0
    igrr = 1-2*M/r+a*a*M*M/(r*r)
    igrp = a*M/(r*r)
    igpp = 1.0/(r*r)

    dg00 = -2*M/(r*r)
    dg0r = -2*M/(r*r)
    dg0p = 2*M*M*a/(r*r)
    dgrr = -2*M/(r*r)
    dgrp = 2*M*M*a/(r*r)
    dgpp = 2*r - 2*M*M*M*a*a/(r*r)

    G000 = 0.5*ig0r*(-dg00)
    G00r = 0.5*(ig00*(dg00) + ig0r*(0) + ig0p*(dg0p))
    G00p = 0.5*ig0r*(-dg0p)
    G0rr = 0.5*(ig00*(2*dg0r) + ig0r*(dgrr) + ig0p*(2*dgrp))
    G0rp = 0.5*(ig00*(dg0p) + ig0r*(0) + ig0p*(dgpp))
    G0pp = 0.5*(ig0r*(-dgpp))
    Gr00 = 0.5*(igrr*(-dg00))
    Gr0r = 0.5*(ig0r*(dg00) + igrr*(0) + igrp*(dg0p))
    Gr0p = 0.5*(igrr*(-dg0p))
    Grrr = 0.5*(ig0r*(2*dg0r) + igrr*(dgrr) + igrp*(2*dgrp))
    Grrp = 0.5*(ig0r*(dg0p) + igrr*(0) + igrp*(dgpp))
    Grpp = 0.5*(igrr*(-dgpp))
    Gp00 = 0.5*(igrp*(-dg00))
    Gp0r = 0.5*(ig0p*(dg00) + igrp*(0) + igpp*(dg0p))
    Gp0p = 0.5*(igrp*(-dg0p))
    Gprr = 0.5*(ig0p*(2*dg0r) + igrp*(dgrr) + igpp*(2*dgrp))
    Gprp = 0.5*(ig0p*(dg0p) + igrp*(0) + igpp*(dgpp))
    Gppp = 0.5*(igrp*(-dgpp))

    u0 = 1.0 / np.sqrt(-g00 - 2*g0r*vr - 2*g0p*vp - grr*vr*vr - 2*grp*vr*vp
                            - gpp*vp*vp)
    ur = u0*vr
    up = u0*vp

    dvr = (VR[1:]-VR[:-1]) / (R[1:]-R[:-1])
    dvp = (VP[1:]-VP[:-1]) / (R[1:]-R[:-1])

    du0 = 0.5*u0*u0*u0 * (dg00 + 2*dg0r*vr + 2*g0r*dvr + 2*dg0p*vp + 2*g0p*dvp
                        + dgrr*vr*vr + 2*grr*vr*dvr + 2*dgrp*vr*vp
                        + 2*grp*dvr*vp + 2*grp*vr*dvp + dgpp*vp*vp
                        + 2*gpp*vp*dvp)
    dur = du0*vr + u0*dvr
    dup = du0*vp + u0*dvp

    D0u0 = G000*u0 + G00r*ur + G00p*up
    D0ur = Gr00*u0 + Gr0r*ur + Gr0p*up
    D0up = Gp00*u0 + Gp0r*ur + Gp0p*up
    Dru0 = du0 + G00r*u0 + G0rr*ur + G0rp*up
    Drur = dur + Gr0r*u0 + Grrr*ur + Grrp*up
    Drup = dup + Gp0r*u0 + Gprr*ur + Gprp*up
    Dpu0 = G00p*u0 + G0rp*ur + G0pp*up
    Dpur = Gr0p*u0 + Grrp*ur + Grpp*up
    Dpup = Gp0p*u0 + Gprp*ur + Gppp*up

    h00 = ig00 + u0*u0
    h0r = ig0r + u0*ur
    h0p = ig0p + u0*up
    hrr = igrr + ur*ur
    hrp = igrp + ur*up
    hpp = igpp + up*up

    th = D0u0 + Drur + Dpup

    sig00 = h00*D0u0 + h0r*Dru0 + h0p*Dpu0 - h00*th
    sigrr = h0r*D0ur + hrr*Drur + hrp*Dpur - hrr*th
    sigpp = h0p*D0up + hrp*Drup + hpp*Dpup - hpp*th
    sig0r = 0.5*(h00*D0ur + h0r*Drur + h0p*Dpur
                + h0r*D0u0 + hrr*Dru0 + hrp*Dpu0) - h0r*th
    sig0p = 0.5*(h00*D0up + h0r*Drup + h0p*Dpup
                + h0p*D0u0 + hrp*Dru0 + hpp*Dpu0) - h0p*th
    sigrp = 0.5*(h0r*D0up + hrr*Drup + hrp*Dpup
                + h0p*D0ur + hrp*Drur + hpp*Dpur) - hrp*th

    xlim = (1, 100)
    fig, ax = plt.subplots(2,3, figsize=(14,9))
    ax[0,0].plot(r, sig00, 'k+')
    ax[0,0].set_xscale("log")
    ax[0,0].set_yscale("linear")
    ax[0,0].set_ylabel(r"$\sigma^{00}$")
    ax[0,0].set_xlim(xlim)
    ax[0,1].plot(r, sigrr, 'k+')
    ax[0,1].set_xscale("log")
    ax[0,1].set_yscale("linear")
    ax[0,1].set_ylabel(r"$\sigma^{rr}$")
    ax[0,1].set_xlim(xlim)
    ax[0,2].plot(r, sigpp, 'k+')
    ax[0,2].set_xscale("log")
    ax[0,2].set_yscale("linear")
    ax[0,2].set_ylabel(r"$\sigma^{\phi\phi}$")
    ax[0,2].set_xlim(xlim)
    ax[1,0].plot(r, sig0r, 'k+')
    ax[1,0].set_xscale("log")
    ax[1,0].set_yscale("linear")
    ax[1,0].set_ylabel(r"$\sigma^{0r}$")
    ax[1,0].set_xlim(xlim)
    ax[1,1].plot(r, sig0p, 'k+')
    ax[1,1].set_xscale("log")
    ax[1,1].set_yscale("linear")
    ax[1,1].set_ylabel(r"$\sigma^{0\phi}$")
    ax[1,1].set_xlim(xlim)

    ax[1,2].plot(r, sigrp, 'k+')
    ax[1,2].plot(r, -1.5 * np.sqrt(M/(r*r*r*r*r)), ls='-', color=blue)
    ax[1,2].set_xscale("log")
    ax[1,2].set_yscale("linear")
    ax[1,2].set_ylabel(r"$\sigma^{r\phi}$")
    ax[1,2].set_xlim(xlim)

    plt.tight_layout()

    return fig, ax


def plotNT(g):

    r, prim = gToArr1D(g)

    rho, sig, p, pi, H, T, vr, vp, mdot, qdot = allTheThings(g, r, prim)

    M = g._pars['GravM']
    a = g._pars['GravA']
    EOSType = g._pars['EOSType']
    EOSPar2 = g._pars['EOSPar2']
    rad = False
    if EOSType == 1 and EOSPar2 == 1.0:
        rad = True

    Risco = ISCO(M,a)

    R = np.logspace(math.log10(Risco)+0.01, math.log10(r.max()), base=10.0, num=200)
    R2 = np.logspace(math.log10(r.min()), math.log10(Risco)-0.05, base=10.0, num=200)

    RS = Risco
    MDOT = 209
    K = 3.5e-5

    if rad:
        SSdat, NTdat, NTraddat = calcNT(g, R, RS, MDOT)
    else:
        SSdat, NTdat = calcNT(g, R, RS, MDOT)

    GEOdat = calcGEO(g, R2, MDOT, K)

    if a != 0.0:
        g._pars['GravA'] = 0.0
        R3 = np.logspace(math.log10(6*M)+0.01, math.log10(r.max()), base=10.0, num=200)
        SSdat2, NTdat2 = calcNT(g, R3, 6*M, MDOT)
        g._pars['GravA'] = a

    fig, ax = plt.subplots(2,3, figsize=(14,9))

    xlim = (1.0, 1.0e3)

    ax[0,0].plot(r, sig, 'k+')
    ax[0,0].plot(R, SSdat[0], ls='-', lw=3.0, color=orange)
    ax[0,0].plot(R2, GEOdat[0], ls='-', lw=3.0, color=green)
    ax[0,0].plot(R, NTdat[0], ls='-', lw=3.0, color=blue)
    if a != 0.0:
        ax[0,0].plot(R3, NTdat2[0], ls='-', lw=3.0, color=red)
    if rad:
        ax[0,0].plot(R, NTraddat[0], ls='-', lw=3.0, color=purple)
    ax[0,0].set_xlabel(r"$r$")
    ax[0,0].set_ylabel(r"$\Sigma$")
    ax[0,0].set_xscale('log')
    ax[0,0].set_yscale('log')
    ax[0,0].set_xlim(xlim)
    
    ax[0,1].plot(r, pi, 'k+')
    ax[0,1].plot(R, SSdat[1], ls='-', lw=3.0, color=orange)
    ax[0,1].plot(R2, GEOdat[1], ls='-', lw=3.0, color=green)
    ax[0,1].plot(R, NTdat[1], ls='-', lw=3.0, color=blue)
    if a != 0.0:
        ax[0,1].plot(R3, NTdat2[1], ls='-', lw=3.0, color=red)
    if rad:
        ax[0,1].plot(R, NTraddat[1], ls='-', lw=3.0, color=purple)
    ax[0,1].set_xlabel(r"$r$")
    ax[0,1].set_ylabel(r"$\Pi$")
    ax[0,1].set_xscale('log')
    ax[0,1].set_yscale('log')
    ax[0,1].set_xlim(xlim)

    ax[1,0].plot(r, -vr, 'k+')
    ax[1,0].plot(R, -SSdat[2], ls='-', lw=3.0, color=orange)
    ax[1,0].plot(R2, -GEOdat[2], ls='-', lw=3.0, color=green)
    ax[1,0].plot(R, -NTdat[2], ls='-', lw=3.0, color=blue)
    if a != 0.0:
        ax[1,0].plot(R3, -NTdat2[2], ls='-', lw=3.0, color=red)
    if rad:
        ax[1,0].plot(R, -NTraddat[2], ls='-', lw=3.0, color=purple)
    ax[1,0].set_xlabel(r"$r$")
    ax[1,0].set_ylabel(r"$v^r$")
    ax[1,0].set_xscale('log')
    ax[1,0].set_yscale('log')
    ax[1,0].set_xlim(xlim)

    ax[1,1].plot(r, vp, 'k+')
    ax[1,1].plot(R, SSdat[3], ls='-', lw=3.0, color=orange)
    ax[1,1].plot(R2, GEOdat[3], ls='-', lw=3.0, color=green)
    ax[1,1].plot(R, NTdat[3], ls='-', lw=3.0, color=blue)
    if a != 0.0:
        ax[1,1].plot(R3, NTdat2[3], ls='-', lw=3.0, color=red)
    if rad:
        ax[1,1].plot(R, NTraddat[3], ls='-', lw=3.0, color=purple)
    ax[1,1].set_xlabel(r"$r$")
    ax[1,1].set_ylabel(r"$v^\phi$")
    ax[1,1].set_xscale('log')
    ax[1,1].set_yscale('log')
    ax[1,1].set_xlim(xlim)
    
    ax[0,2].plot(r, qdot, 'k+')
    ax[0,2].plot(R, SSdat[4], ls='-', lw=3.0, color=orange)
    ax[0,2].plot(R2, GEOdat[4], ls='-', lw=3.0, color=green)
    ax[0,2].plot(R, NTdat[4], ls='-', lw=3.0, color=blue)
    if a != 0.0:
        ax[0,2].plot(R3, NTdat2[4], ls='-', lw=3.0, color=red)
    if rad:
        ax[0,2].plot(R, NTraddat[4], ls='-', lw=3.0, color=purple)
    ax[0,2].set_xlabel(r"$r$")
    ax[0,2].set_ylabel(r"$\dot{Q}$")
    ax[0,2].set_xscale('log')
    ax[0,2].set_yscale('log')
    ax[0,2].set_xlim(xlim)

    ax[1,2].plot(r, mdot, 'k+')
    ax[1,2].plot(R, MDOT*np.ones(R.shape), ls='-', lw=3.0, color=orange)
    ax[1,2].plot(R, MDOT*np.ones(R.shape), ls='-', lw=3.0, color=blue)
    ax[1,2].set_xlabel(r"$r$")
    ax[1,2].set_ylabel(r"$\dot{M}$")
    ax[1,2].set_xscale('log')
    ax[1,2].set_yscale('log')
    ax[1,2].set_xlim(xlim)

    plt.tight_layout()

    return fig, ax

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Need a parfile and checkpoint, numbnuts!")

    print("Making grid.")
    par = dp.readParfile(sys.argv[1])
    g = dp.Grid(par)

    if len(sys.argv) == 3:

        checkpoint = sys.argv[2]
        
        print("Loading {0:s}".format(checkpoint))
        g.loadCheckpoint(sys.argv[2])
        num = int(sys.argv[2].split(".")[-2].split("_")[-1])
        name = "plot_thin_{0:04d}.png".format(num)
        nameSig = "plot_thin_Sig_{0:04d}.png".format(num)
        nameQdot = "plot_thin_Qdot_{0:04d}.png".format(num)
        nameShear = "plot_thin_shear_{0:04d}.png".format(num)

        print("   Plotting {0:s}".format(checkpoint))
        fig, ax = plotNT(g)
        fig2, ax2 = plotSigNice(g)
        fig3, ax3 = plotQdotNice(g)
        fig4, ax4 = plot_shear(g)
        
        print("   Saving {0:s}".format(name))
        fig.savefig(name)
        fig2.savefig(nameSig)
        fig3.savefig(nameQdot)
        fig4.savefig(nameShear)

        plt.show()

    else:

        for checkpoint in sys.argv[2:]:

            print("Loading {0:s}".format(checkpoint))
            g.loadCheckpoint(checkpoint)
            num = int(checkpoint.split(".")[-2].split("_")[-1])
            name = "plot_thin_{0:04d}.png".format(num)
            nameSig = "plot_thin_Sig_{0:04d}.png".format(num)
            
            print("   Plotting {0:s}".format(checkpoint))
            #fig, ax = plotNT(g)
            fig2, ax2 = plotSigNice(g)
            
            print("   Saving {0:s}".format(name))
            #fig.savefig(name)
            fig2.savefig(nameSig)
            
            plt.close()


