import math
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import sys
import numpy as np
import discopy as dp
import discoEOS as eos
import discoGR as gr

scale = 'log'
RMAX = 100.0
RMIN = 6.0

def allTheThings(filename, pars):

    dat = dp.readCheckpoint(filename)
    t = dat[0]
    r = dat[1]
    phi = dat[2]
    vr = dat[6]
    vp = dat[7]
    q = dat[10]
    dphi = dat[12]
    u0, ur, up = gr.calc_u(r, vr, vp, pars)
    M = pars['GravM']

    if pars['Background'] == 3:
        rho = dat[4]
        T = dat[5]
        P = eos.ppp(rho, T, pars)
        eps = eos.eps(rho, T, pars)
        rhoh = rho + rho*eps + P
        H = np.sqrt(r*r*r*P / (M*rhoh)) / u0
        sig = rho*H
        pi = P*H

    else:
        sig = dat[4]
        pi = dat[5]
        GAM = pars['Adiabatic_Index']
        sigh = sig + GAM/(GAM-1.0)*pi
        H = np.sqrt(r*r*r*pi / (M*sigh)) / u0
        rho = sig/H
        P = pi/H
        T = pi/sig

    return t, r, phi, rho, sig, T, P, pi, H, vr, vp, u0, q, dphi


def plot_data(ax, x, f, color='k', marker='+'):
    ax.plot(x, f, marker=marker, ls='', color=color)

def plot_line(ax, x, f, color='k', ls='-', lw=2.0, alpha=0.5):
    ax.plot(x, f, color=color, ls=ls, lw=lw, alpha=alpha)

def floor_sig(x, sig):
    if x == 0.0:
        return 0.0
    exp = int(math.floor(math.log10(math.fabs(x))))
    y = math.floor(x * 10.0**(-exp+sig))
    return y * 10.0**(exp-sig)
    #y = np.around(x, -exp+sig)
    #if y > x:
    #    y -= math.pow(10, sig-sig)
    #return y

def ceil_sig(x, sig):
    if x == 0.0:
        return 0.0
    exp = int(math.floor(math.log10(math.fabs(x))))
    y = math.ceil(x * 10.0**(-exp+sig))
    return y * 10.0**(exp-sig)
    #exp = int(math.floor(math.log10(math.fabs(x))))
    #y = np.around(x, -exp+sig)
    #if y < x:
    #    y += math.pow(10, sig-sig)
    #return y

def pretty_axis(ax, pars, xscale="linear", yscale="linear", 
                xlabel=None, ylabel=None, xlim=None, ylim=None, twin=False,
                horizon=False):

    if ylim is None:
        ylim0 = list(ax.get_ylim())
        ylim = [0.0,1.0]
        if yscale is "log":
            if ylim0[0] <= 0.0:
                ylim[0] = 0.0
            else:
                exp0 = math.floor(math.log10(ylim0[0]))
                ylim[0] = math.pow(10.0,exp0)
            if ylim0[1] <= 0.0:
                ylim[1] = 1.0
            else:
                exp1 = math.ceil(math.log10(ylim0[1]))
                ylim[1] = math.pow(10.0,exp1)
        else:
            ylim[0] = floor_sig(ylim0[0], 1)
            ylim[1] = ceil_sig(ylim0[1], 1)

    if ylim[0] > ylim[1]:
        print ylim
    ax.set_yscale(yscale)
    ax.set_ylim(ylim)
    if ylabel != None:
        ax.set_ylabel(ylabel)

    if xlim is None:
        xlim = list(ax.get_xlim())
        xlim[0] = floor_sig(xlim[0], 1)
        xlim[1] = ceil_sig(xlim[1], 1)
    if xlim[0] > xlim[1]:
        print xlim
    ax.set_xscale(xscale)
    ax.set_xlim(xlim)

    if not twin:
        M = pars['GravM']
        a = pars['GravA']

        Rsp = M*(1.0+math.sqrt(1.0-a*a))
        Rsm = M*(1.0-math.sqrt(1.0-a*a))
        Rer = 2*M

        if xlabel != None:
            ax.set_xlabel(xlabel)

        if horizon:
            #Horizon
            ax.axvspan(max(Rsm,xlim[0]), min(Rsp,xlim[1]), color='grey', 
                                            zorder=1, alpha=0.5)
            #Ergosphere
            ax.axvspan(max(0,xlim[0]), min(Rer,xlim[1]), color='lightgrey', 
                                            zorder=1, alpha=0.5)

def wavespeeds(r, u0, ur, up, cs, pars):
    g00, g0r, g0p, grr, grp, gpp = gr.calc_g(r, pars)
    igamrr, igamrp, igampp = gr.calc_igam(r, pars)
    br, bp = gr.calc_shift(r, pars)
    a = gr.lapse(r, pars)

    w = a*u0
    Ur = ur + br*u0
    Up = up + bp*u0
    Vr = Ur/w
    Vp = Up/w
    V2 = grr*Vr*Vr + 2*grp*Vr*Vp + gpp*Vp*Vp

    dvr = cs*np.sqrt(igamrr*(1-cs*cs*V2) - (1-cs*cs)*Vr*Vr) / w
    vrp = a * (Vr*(1-cs*cs) + dvr) / (1-cs*cs*V2) - br
    vrm = a * (Vr*(1-cs*cs) - dvr) / (1-cs*cs*V2) - br

    dvp = cs*np.sqrt(igampp*(1-cs*cs*V2) - (1-cs*cs)*Vp*Vp) / w
    vpp = a * (Vp*(1-cs*cs) + dvp) / (1-cs*cs*V2) - bp
    vpm = a * (Vp*(1-cs*cs) - dvp) / (1-cs*cs*V2) - bp

    return (vrp, vrm), (vpp, vpm)


def plot_r_profile(filename, pars, sca='linear', plot=True, bounds=None):

    print("Reading {0:s}".format(filename))
    
    M = pars['GravM']
    a = pars['GravA']
    gam = pars['Adiabatic_Index']
    bw = pars['BinW']
    A = a*M

    t, r, phi, rho, sig, T, P, pi, H, vr, vp, u0, q, dphi = allTheThings(
                                                                filename, pars)
    inds = (r < RMAX) * (r > RMIN)
    r = r[inds]
    phi = phi[inds]
    rho = rho[inds]
    sig = sig[inds]
    T = T[inds]
    P = P[inds]
    pi = pi[inds]
    H = H[inds]
    vr = vr[inds]
    vp = vp[inds]
    u0 = u0[inds]
    dphi = dphi[inds]
    inds = np.argsort(r)
    r = r[inds]
    phi = phi[inds]
    rho = rho[inds]
    sig = sig[inds]
    T = T[inds]
    P = P[inds]
    pi = pi[inds]
    H = H[inds]
    vr = vr[inds]
    vp = vp[inds]
    u0 = u0[inds]
    dphi = dphi[inds]

    R = np.logspace(np.log10(r.min()), np.log10(r.max()), 100)

    w = u0 * gr.lapse(r, pars)
    u = np.sqrt(w*w-1)

    g00, g0r, g0p, grr, grp, gpp = gr.calc_g(r, pars)
    ur = u0*vr
    up = u0*vp

    up_lab = up + bw*u0

    u0d = g00*u0 + g0r*ur + g0p*up
    urd = g0r*u0 + grr*ur + grp*up
    upd = g0p*u0 + grp*ur + gpp*up


    u0sc = np.sqrt((1.0 + ur*ur/(1-2*M/r) + up_lab*up_lab*r*r) / (1.0-2*M/r))
    u0dsc = -(1-2*M/r) * u0sc
    updsc = r*r*up_lab

    #Mdot = - r*sig*u0*vr * (eos.c * eos.rg_solar**2 * eos.year
    #                                    / eos.M_solar)
    #Mdot = - r*sig*ur

    cs2 = eos.cs2(rho, T, pars)
    cs = np.sqrt(cs2)
    ucs = np.sqrt(cs2 / (1-cs2))

    mach = u / ucs

    Rs = np.unique(r)

    avsig = np.zeros(Rs.shape)
    avD = np.zeros(Rs.shape)
    avL = np.zeros(Rs.shape)
    avE = np.zeros(Rs.shape)
    avvr = np.zeros(Rs.shape)
    avvp = np.zeros(Rs.shape)
    Mdot = np.zeros(Rs.shape)
    Ldot = np.zeros(Rs.shape)
    Edot = np.zeros(Rs.shape)
    Lindot = np.zeros(Rs.shape)
    avVflux = np.zeros(Rs.shape)
    avMach = np.zeros(Rs.shape)

    gam = pars['Adiabatic_Index']
    sigh = sig + gam/(gam-1.0)*pi
    S = np.log(pi * np.power(sig, -gam)) / (gam-1.0)

    nmodes = 16
    nm = np.arange(1, nmodes+1)
    A0 = np.zeros(Rs.shape[0])
    An = np.zeros((Rs.shape[0], nmodes))
    PhiN = np.zeros((Rs.shape[0], nmodes))

    deltaRho = np.zeros(Rs.shape[0])
    deltaSig = np.zeros(Rs.shape[0])
    deltaPi = np.zeros(Rs.shape[0])
    deltaS = np.zeros(Rs.shape[0])
    Rho0 = np.zeros(Rs.shape[0])
    Sig0 = np.zeros(Rs.shape[0])
    Pi0 = np.zeros(Rs.shape[0])
    S0 = np.zeros(Rs.shape[0])

    for i,R in enumerate(Rs):

        inds = r==R
        A = (R*dphi[inds]).sum()
        D = sig[inds]*u0[inds]
        sigtot = (sig[inds] * R*dphi[inds]).sum()
        Dtot = (D * R*dphi[inds]).sum()
        Ltot = (sigh[inds]*u0[inds]*upd[inds] * R*dphi[inds]).sum()
        Etot = -((sigh[inds]*u0[inds]*u0d[inds]+pi[inds]) * R*dphi[inds]).sum()

        Dflux = (sig[inds]*ur[inds] * R*dphi[inds]).sum()
        Lflux = (sigh[inds]*ur[inds]*upd[inds] * R*dphi[inds]).sum()
        Eflux = -(sigh[inds]*ur[inds]*u0d[inds] * R*dphi[inds]).sum()
        Dfluxp = (sig[inds]*up[inds] * R*dphi[inds]).sum() #this is weird
        Vflux = (sig[inds]*cs2[inds] * R*dphi[inds]).sum()

        avupd = (upd[inds] * R*dphi[inds]).sum() / A
        sighflux = (sigh[inds]*ur[inds] * R*dphi[inds]).sum()
        avsigflux = -(sig[inds]*ur[inds] * R*dphi[inds]).sum() / A

        deltaRho[i] = rho[inds].max() - rho[inds].min()
        deltaSig[i] = sig[inds].max() - sig[inds].min()
        deltaPi[i] = pi[inds].max() - pi[inds].min()
        deltaS[i] = S[inds].max() - S[inds].min()
        Pi0[i] = pi[inds].min()
        Sig0[i] = sig[inds].min()
        Rho0[i] = rho[inds].min()
        S0[i] = S[inds].min()

        avsig[i] = sigtot / A
        avD[i] = Dtot / A
        avL[i] = Ltot / A
        avE[i] = Etot / A
        avvr[i] = Dflux / Dtot
        avvp[i] = Dfluxp / Dtot
        avMach[i] = (sig[inds]*mach[inds] * R*dphi[inds]).sum() / sigtot

        Mdot[i] = -Dflux
        Ldot[i] = -Lflux
        Edot[i] = -Eflux
        Lindot[i] = sighflux * avupd
        avVflux[i] = Vflux * R  #Not actually average, just go with it.

        ph = phi[inds]
        a0 = (D*dphi[inds]).sum()/(2*np.pi)
        an = (D[None,:] * np.cos(nm[:,None]*ph[None,:]) * dphi[inds][None,:]
                ).sum(axis=1) / np.pi
        bn = (D[None,:] * np.sin(nm[:,None]*ph[None,:]) * dphi[inds][None,:]
                ).sum(axis=1) / np.pi
        A0[i] = a0
        An[i,:] = np.sqrt(an*an + bn*bn)
        PhiN[i,:] = np.arctan2(bn,an) / nm

    j = Ldot/Mdot
    e = Edot/Mdot
    jin = Lindot/Mdot
    jout = j - jin
    
    alpha = -2.0/3.0 * Lindot / avVflux

    #Rafikov-Analysis
    djdr = (j[2:]-j[:-2]) / (Rs[2:]-Rs[:-2])
    pi_raf_sig = (2+(gam+1)*deltaSig/Sig0) / (2-(gam-1)*deltaSig/Sig0)
    pi_raf_rho = (2+(gam+1)*deltaRho/Rho0) / (2-(gam-1)*deltaRho/Rho0)

    psi_Q_sig = (pi_raf_sig * np.power((gam+1+(gam-1)*pi_raf_sig)
                    /(gam-1+(gam+1)*pi_raf_sig), gam) - 1) / (gam-1)
    psi_Q_rho = (pi_raf_rho * np.power((gam+1+(gam-1)*pi_raf_rho)
                    /(gam-1+(gam+1)*pi_raf_rho), gam) - 1) / (gam-1)

    Mdot_raf_sig = 2 * avD[1:-1] * Rs[1:-1] * Pi0[1:-1]/Sig0[1:-1] \
                        * psi_Q_sig[1:-1] / djdr
    Mdot_raf_rho = 2 * avD[1:-1] * Rs[1:-1] * Pi0[1:-1]/Sig0[1:-1] \
                        * psi_Q_rho[1:-1] / djdr

    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.plot(Rs, Mdot, 'k+')
    ax.plot(Rs[1:-1], Mdot_raf_sig, 'b+')
    #ax.plot(Rs[1:-1], Mdot_raf_rho, 'r+')
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\dot{M}$")
    ax.set_xlim(Rs.min(), Rs.max())
    ax.set_xscale("log")
    ax.set_yscale("log")

    outpath = filename.split("/")[:-1]
    chckname = filename.split("/")[-1]
    outname = "plot_minidisc_rafikov_{0}.png".format(
                "_".join(chckname.split(".")[0].split("_")[1:]))
    print("Saving {0:s}...".format(outname))
    fig.savefig(outname)
    plt.close(fig)

    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.plot(Rs*np.cos(PhiN[:,0]), Rs*np.sin(PhiN[:,0]), 'k+')

    """
    ax.plot(Rs*np.cos(PhiN[:,1]), Rs*np.sin(PhiN[:,1]), 'b+')
    ax.plot(Rs*np.cos(PhiN[:,1]+np.pi), Rs*np.sin(PhiN[:,1]+np.pi), 'b+')
    
    ax.plot(Rs*np.cos(PhiN[:,2]), Rs*np.sin(PhiN[:,2]), 'g+')
    ax.plot(Rs*np.cos(PhiN[:,2]+2*np.pi/3), Rs*np.sin(PhiN[:,2]+2*np.pi/3), 
            'g+')
    ax.plot(Rs*np.cos(PhiN[:,2]+4*np.pi/3), Rs*np.sin(PhiN[:,2]+4*np.pi/3), 
            'g+')

    ax.plot(Rs*np.cos(PhiN[:,3]), Rs*np.sin(PhiN[:,3]), 'r+')
    ax.plot(Rs*np.cos(PhiN[:,3]+np.pi/2), Rs*np.sin(PhiN[:,3]+np.pi/2), 'r+')
    ax.plot(Rs*np.cos(PhiN[:,3]+np.pi), Rs*np.sin(PhiN[:,3]+np.pi), 'r+')
    ax.plot(Rs*np.cos(PhiN[:,3]+3*np.pi/2), Rs*np.sin(PhiN[:,3]+3*np.pi/2), 
            'r+')
    """
    ax.set_xlim(-RMAX, RMAX)
    ax.set_ylim(-RMAX, RMAX)

    outname = "plot_minidisc_phases_{0}.png".format(
                "_".join(chckname.split(".")[0].split("_")[1:]))
    print("Saving {0:s}...".format(outname))
    fig.savefig(outname)
    plt.close(fig)

    UPDSC_circ = np.sqrt(M*Rs/(1-3*M/Rs))
    U0DSC_circ = -(1-2*M/Rs) / np.sqrt(1-3*M/Rs)
    
    fig, ax = plt.subplots(2,2, figsize=(12,9))
    plot_data(ax[0,0], updsc, u0dsc)
    plot_data(ax[0,0], UPDSC_circ, U0DSC_circ, 'r')
    pretty_axis(ax[0,0], pars, xlabel=r"$u_\phi$", ylabel=r"$u_0$")
    plot_data(ax[0,1], Rs, alpha)
    pretty_axis(ax[0,1], pars, xlabel=r"$R$", ylabel=r"$\alpha$", yscale='log')
    plot_data(ax[1,0], avMach, alpha)
    pretty_axis(ax[1,0], pars, xlabel=r"$\mathcal{M}$", ylabel=r"$\alpha$", 
            xscale='log', yscale='log')
    plot_data(ax[1,1], Rs, avMach)
    pretty_axis(ax[1,1], pars, xlabel=r"$R$", ylabel=r"$\mathcal{M}$", 
            yscale='log')
    outname = "plot_minidisc_orbit_{0}.png".format(
                "_".join(chckname.split(".")[0].split("_")[1:]))
    print("Saving {0:s}...".format(outname))
    fig.savefig(outname)
    plt.close(fig)

    fig, ax = plt.subplots(3, 4, figsize=(14,9))
    plot_data(ax[0,0], Rs, An[:,0]/A0)
    pretty_axis(ax[0,0], pars, xlabel=r"$R$ ($M$)", ylabel=r"$m=1$")
    plot_data(ax[0,1], Rs, An[:,1]/A0)
    pretty_axis(ax[0,1], pars, xlabel=r"$R$ ($M$)", ylabel=r"$m=2$")
    plot_data(ax[0,2], Rs, An[:,2]/A0)
    pretty_axis(ax[0,2], pars, xlabel=r"$R$ ($M$)", ylabel=r"$m=3$")
    plot_data(ax[0,3], Rs, An[:,3]/A0)
    pretty_axis(ax[0,3], pars, xlabel=r"$R$ ($M$)", ylabel=r"$m=4$")
    plot_data(ax[1,0], Rs, An[:,4]/A0)
    pretty_axis(ax[1,0], pars, xlabel=r"$R$ ($M$)", ylabel=r"$m=5$")
    plot_data(ax[1,1], Rs, An[:,5]/A0)
    pretty_axis(ax[1,1], pars, xlabel=r"$R$ ($M$)", ylabel=r"$m=6$")
    plot_data(ax[1,2], Rs, An[:,6]/A0)
    pretty_axis(ax[1,2], pars, xlabel=r"$R$ ($M$)", ylabel=r"$m=7$")
    plot_data(ax[1,3], Rs, An[:,7]/A0)
    pretty_axis(ax[1,3], pars, xlabel=r"$R$ ($M$)", ylabel=r"$m=8$")
   
    i = len(Rs)/4
    plot_data(ax[2,0], nm, An[i,:]/A0[i])
    pretty_axis(ax[2,0], pars, xlabel=r"$m$", 
                    ylabel=r"$A_m$ ($r={0:.2g}$)".format(Rs[i]))
    i = len(Rs)/2
    plot_data(ax[2,1], nm, An[i,:]/A0[i])
    pretty_axis(ax[2,1], pars, xlabel=r"$m$", 
                    ylabel=r"$A_m$ ($r={0:.2g}$)".format(Rs[i]))
    i = 3*len(Rs)/4
    plot_data(ax[2,2], nm, An[i,:]/A0[i])
    pretty_axis(ax[2,2], pars, xlabel=r"$m$", 
                    ylabel=r"$A_m$ ($r={0:.2g}$)".format(Rs[i]))
    i = len(Rs)-1
    plot_data(ax[2,3], nm, An[i,:]/A0[i])
    pretty_axis(ax[2,3], pars, xlabel=r"$m$", 
                    ylabel=r"$A_m$ ($r={0:.2g}$)".format(Rs[i]))
    plt.tight_layout()
    
    outname = "plot_minidisc_amplitudes_{0}.png".format(
                "_".join(chckname.split(".")[0].split("_")[1:]))
    print("Saving {0:s}...".format(outname))
    fig.savefig(outname)
    plt.close(fig)


    fig, ax = plt.subplots(5, 5, figsize=(14,10))

    i = len(Rs)/5
    plot_data(ax[0,0], phi[r==Rs[i]], sig[r==Rs[i]])
    pretty_axis(ax[0,0], pars, xlabel=r"$\phi$", 
            ylabel=r"$\Sigma_0$ ($r={0:.2g}$)".format(Rs[i]))
    plot_data(ax[1,0], phi[r==Rs[i]], pi[r==Rs[i]])
    pretty_axis(ax[1,0], pars, xlabel=r"$\phi$", ylabel=r"$\Pi_0$")
    plot_data(ax[2,0], phi[r==Rs[i]], vr[r==Rs[i]])
    pretty_axis(ax[2,0], pars, xlabel=r"$\phi$", ylabel=r"$v^r$")
    plot_data(ax[3,0], phi[r==Rs[i]], vp[r==Rs[i]])
    plot_data(ax[3,0], phi[r==Rs[i]], np.sqrt(M/(r[r==Rs[i]]**3))-bw,
                color='r')
    pretty_axis(ax[3,0], pars, xlabel=r"$\phi$", ylabel=r"$v^\phi$")
    plot_data(ax[4,0], phi[r==Rs[i]], S[r==Rs[i]])
    pretty_axis(ax[4,0], pars, xlabel=r"$\phi$", ylabel=r"$S$")

    i = 2*len(Rs)/5
    plot_data(ax[0,1], phi[r==Rs[i]], sig[r==Rs[i]])
    pretty_axis(ax[0,1], pars, xlabel=r"$\phi$", 
            ylabel=r"$\Sigma_0$ ($r={0:.2g}$)".format(Rs[i]))
    plot_data(ax[1,1], phi[r==Rs[i]], pi[r==Rs[i]])
    pretty_axis(ax[1,1], pars, xlabel=r"$\phi$", ylabel=r"$\Pi_0$")
    plot_data(ax[2,1], phi[r==Rs[i]], vr[r==Rs[i]])
    pretty_axis(ax[2,1], pars, xlabel=r"$\phi$", ylabel=r"$v^r$")
    plot_data(ax[3,1], phi[r==Rs[i]], vp[r==Rs[i]])
    plot_data(ax[3,1], phi[r==Rs[i]], np.sqrt(M/(r[r==Rs[i]]**3))-bw, 
                color='r')
    pretty_axis(ax[3,1], pars, xlabel=r"$\phi$", ylabel=r"$v^\phi$")
    plot_data(ax[4,1], phi[r==Rs[i]], S[r==Rs[i]])
    pretty_axis(ax[4,1], pars, xlabel=r"$\phi$", ylabel=r"$S$")

    i = 3*len(Rs)/5
    plot_data(ax[0,2], phi[r==Rs[i]], sig[r==Rs[i]])
    pretty_axis(ax[0,2], pars, xlabel=r"$\phi$", 
            ylabel=r"$\Sigma_0$ ($r={0:.2g}$)".format(Rs[i]))
    plot_data(ax[1,2], phi[r==Rs[i]], pi[r==Rs[i]])
    pretty_axis(ax[1,2], pars, xlabel=r"$\phi$", ylabel=r"$\Pi_0$")
    plot_data(ax[2,2], phi[r==Rs[i]], vr[r==Rs[i]])
    pretty_axis(ax[2,2], pars, xlabel=r"$\phi$", ylabel=r"$v^r$")
    plot_data(ax[3,2], phi[r==Rs[i]], vp[r==Rs[i]])
    plot_data(ax[3,2], phi[r==Rs[i]], np.sqrt(M/(r[r==Rs[i]]**3))-bw, 
                color='r')
    pretty_axis(ax[3,2], pars, xlabel=r"$\phi$", ylabel=r"$v^\phi$")
    plot_data(ax[4,2], phi[r==Rs[i]], S[r==Rs[i]])
    pretty_axis(ax[4,2], pars, xlabel=r"$\phi$", ylabel=r"$S$")

    i = 4*len(Rs)/5
    plot_data(ax[0,3], phi[r==Rs[i]], sig[r==Rs[i]])
    pretty_axis(ax[0,3], pars, xlabel=r"$\phi$", 
            ylabel=r"$\Sigma_0$ ($r={0:.2g}$)".format(Rs[i]))
    plot_data(ax[1,3], phi[r==Rs[i]], pi[r==Rs[i]])
    pretty_axis(ax[1,3], pars, xlabel=r"$\phi$", ylabel=r"$\Pi_0$")
    plot_data(ax[2,3], phi[r==Rs[i]], vr[r==Rs[i]])
    pretty_axis(ax[2,3], pars, xlabel=r"$\phi$", ylabel=r"$v^r$")
    plot_data(ax[3,3], phi[r==Rs[i]], vp[r==Rs[i]])
    plot_data(ax[3,3], phi[r==Rs[i]], np.sqrt(M/(r[r==Rs[i]]**3))-bw, 
                color='r')
    pretty_axis(ax[3,3], pars, xlabel=r"$\phi$", ylabel=r"$v^\phi$")
    plot_data(ax[4,3], phi[r==Rs[i]], S[r==Rs[i]])
    pretty_axis(ax[4,3], pars, xlabel=r"$\phi$", ylabel=r"$S$")

    i = len(Rs)-1
    plot_data(ax[0,4], phi[r==Rs[i]], sig[r==Rs[i]])
    pretty_axis(ax[0,4], pars, xlabel=r"$\phi$", 
            ylabel=r"$\Sigma_0$ ($r={0:.2g}$)".format(Rs[i]))
    plot_data(ax[1,4], phi[r==Rs[i]], pi[r==Rs[i]])
    pretty_axis(ax[1,4], pars, xlabel=r"$\phi$", ylabel=r"$\Pi_0$")
    plot_data(ax[2,4], phi[r==Rs[i]], vr[r==Rs[i]])
    pretty_axis(ax[2,4], pars, xlabel=r"$\phi$", ylabel=r"$v^r$")
    plot_data(ax[3,4], phi[r==Rs[i]], vp[r==Rs[i]])
    plot_data(ax[3,4], phi[r==Rs[i]], np.sqrt(M/(r[r==Rs[i]]**3))-bw,
                color='r')
    pretty_axis(ax[3,4], pars, xlabel=r"$\phi$", ylabel=r"$v^\phi$")
    plot_data(ax[4,4], phi[r==Rs[i]], S[r==Rs[i]])
    pretty_axis(ax[4,4], pars, xlabel=r"$\phi$", ylabel=r"$S$")
    plt.tight_layout()

    outname = "plot_minidisc_azimuth_{0}.png".format(
                "_".join(chckname.split(".")[0].split("_")[1:]))
    print("Saving {0:s}...".format(outname))
    fig.savefig(outname)
    plt.close(fig)

    fig, ax = plt.subplots(3, 4, figsize=(14,9))

    plot_data(ax[0,0], Rs, avsig)
    pretty_axis(ax[0,0], pars, xlabel=r"$R$ ($M$)", 
                    ylabel=r"$\langle \Sigma_0 \rangle$")

    plot_data(ax[0,1], Rs, avvr)
    pretty_axis(ax[0,1], pars, xlabel=r"$R$ ($M$)", 
                    ylabel=r"$\langle v^r \rangle$")

    plot_data(ax[0,2], Rs, avvp)
    pretty_axis(ax[0,2], pars, xlabel=r"$R$ ($M$)", 
                    ylabel=r"$\langle v^\phi \rangle$")

    plot_data(ax[1,0], Rs, avD)
    pretty_axis(ax[1,0], pars, xlabel=r"$R$ ($M$)", 
                    ylabel=r"$\langle D \rangle$")

    plot_data(ax[1,1], Rs, avL)
    pretty_axis(ax[1,1], pars, xlabel=r"$R$ ($M$)", 
                    ylabel=r"$\langle T^0_\phi \rangle$")

    plot_data(ax[1,2], Rs, avE)
    pretty_axis(ax[1,2], pars, xlabel=r"$R$ ($M$)", 
                    ylabel=r"$\langle -T^0_0 \rangle$")

    plot_data(ax[2,0], Rs, Mdot)
    pretty_axis(ax[2,0], pars, xlabel=r"$R$ ($M$)", 
                    ylabel=r"$\langle \dot{M} \rangle$")

    plot_data(ax[2,1], Rs, Ldot)
    pretty_axis(ax[2,1], pars, xlabel=r"$R$ ($M$)", 
                    ylabel=r"$\langle T^r_\phi \rangle$")

    plot_data(ax[2,2], Rs, Edot)
    pretty_axis(ax[2,2], pars, xlabel=r"$R$ ($M$)", 
                    ylabel=r"$\langle -T^r_0 \rangle$")

    plot_data(ax[1,3], Rs, j)
    plot_data(ax[1,3], Rs, jin, color='b')
    plot_data(ax[1,3], Rs, jout, color='r')
    pretty_axis(ax[1,3], pars, xlabel=r"$R$ ($M$)", 
                    ylabel=r"$j$", ylim=[-10,10])

    plot_data(ax[2,3], Rs, e)
    pretty_axis(ax[2,3], pars, xlabel=r"$R$ ($M$)", 
                    ylabel=r"$e$", ylim=[-10,10])


    plt.tight_layout()

    outname = "plot_minidisc_analysis_{0}.png".format("_".join(chckname.split(".")[0].split("_")[1:]))

    print("Saving {0:s}...".format(outname))
    fig.savefig(outname)

    return fig, None

"""
    #binW = pars['BinW']#\

    #ur = u0*vr
    #up = u0*(vp+binW)

    #u0sc = np.sqrt((1 + ur*ur/(1-2*M/r) + r*r*up*up) / (1-2*M/r))
    #u0d = -(1-2*M/r)*u0sc
    #upd = r*r*up

    #en = 0.5*(u0d*u0d-1.0)
    #j = upd

    #slr = j*j/M
    #ecc = np.sqrt(1 + 2*en*j*j/(M*M))
    #phip = phi - np.arccos((slr/r-1)/ecc)
    #print(r[(slr/r-1)/ecc <= -1.0])
    #print(r[(slr/r-1)/ecc >= 1.0])
    #print(phi[(slr/r-1)/ecc >= 1.0])
    phip[phip>np.pi] -= 2*np.pi
    phip[phip<-np.pi] += 2*np.pi

    Rs = np.unique(r)
    alphaSS = np.zeros(Rs.shape[0])
    MDOT = np.zeros(Rs.shape[0])
    MDOTRRR = np.zeros(Rs.shape[0])
    sigJump = np.zeros(Rs.shape[0])
    piJump = np.zeros(Rs.shape[0])
    sJump = np.zeros(Rs.shape[0])
    
    l = np.zeros(Rs.shape[0])

    for i,RR in enumerate(Rs):
        inds = (r==RR)
        N = r[inds].shape[0]
        l[i] = RR*RR*(u0[inds]*(vp[inds]+binW)).sum()/N
    dl = np.gradient(l)
    for i,RR in enumerate(Rs):
        inds = (r==RR)
        N = r[inds].shape[0]
        MDOT[i] = Mdot[inds].sum() * 2*np.pi/N
        DISS = (1.5*cs[inds]*H[inds]*sig[inds]).sum() * 2*np.pi/N
        alphaSS[i] = MDOT[i]/DISS

        dsig = np.zeros(N)
        dsig[:-1] = np.diff(sig[inds])
        dsig[-1] = sig[inds][0]-sig[inds][-1]
        dpi = np.zeros(N)
        dpi[:-1] = np.diff(pi[inds])
        dpi[-1] = pi[inds][0]-pi[inds][-1]
        s = np.log(np.power(pi[inds]/sig[inds],1/(gam-1))/sig[inds])
        ds = np.zeros(N)
        ds[:-1] = np.diff(s)
        ds[-1] = s[0] - s[-1]

        MDOTRRR[i] = ((sig[inds]*pi[inds]/(vp[inds]+binW)).sum() 
                        /sig[inds].sum()) / (dl[i]*RR/l[i])

        jmax = np.argmax(np.fabs(dsig))
        #print jmax,
        sigJump[i] = dsig[jmax]/sig[inds][jmax]
        jmax = np.argmax(np.fabs(dpi))
        #print jmax,
        piJump[i] = dpi[jmax]/pi[inds][jmax]
        jmax = np.argmax(np.fabs(ds))
        sJump[i] = ds[jmax]
        #print jmax


    if plot:

        print("Plotting t = {0:g}".format(t))

        #Plot.
        fig, ax = plt.subplots(3,3,figsize=(12,9))

        plot_data(ax[0,0], slr, ecc)
        pretty_axis(ax[0,0], pars, xlabel=r"$\ell$ ($M$)", ylabel=r"$\epsilon$")
        
        plot_data(ax[1,0], slr, phip)
        pretty_axis(ax[1,0], pars, xlabel=r"$\ell$ ($M$)", ylabel=r"$\phi_p$")
        
        plot_data(ax[0,1], phip, ecc)
        pretty_axis(ax[0,1], pars, xlabel=r"$\phi_p$", ylabel=r"$\epsilon$")

        plot_data(ax[0,2], Rs, MDOT)
        plot_data(ax[0,2], Rs, MDOTRRR, color='b')
        pretty_axis(ax[0,2], pars, xlabel=r"$r$ ($M$)", ylabel=r"$\dot{M}$",
                    xscale='log', yscale='log')

        plot_data(ax[1,1], r, vp)
        pretty_axis(ax[1,1], pars, xlabel=r"$r$ ($M$)", ylabel=r"$v^\phi$",
                    xscale='log', yscale='linear')

        plot_data(ax[1,2], Rs, alphaSS)
        pretty_axis(ax[1,2], pars, xlabel=r"$r$ ($M$)", ylabel=r"$\alpha$",
                    xscale='log', yscale='log')

        plot_data(ax[2,0], Rs, sigJump)
        pretty_axis(ax[2,0], pars, xlabel=r"$r$ ($M$)", 
                    ylabel=r"$\Delta \Sigma_s / \Sigma_0$", 
                    xscale='log', yscale='log')

        plot_data(ax[2,1], Rs, piJump)
        pretty_axis(ax[2,1], pars, xlabel=r"$r$ ($M$)", 
                    ylabel=r"$\Delta \Pi_s / \Pi_0$", 
                    xscale='log', yscale='log')

        plot_data(ax[2,2], Rs, sJump)
        pretty_axis(ax[2,2], pars, xlabel=r"$r$ ($M$)", 
                    ylabel=r"$\Delta S_s / S_0$", 
                    xscale='log', yscale='log')


        fig.suptitle(r"DISCO $M = {0:.1g}M_\odot$ $a_* = {1:.2f}$".format(M,a))

        plt.tight_layout()

        outpath = filename.split("/")[:-1]
        chckname = filename.split("/")[-1]
        outname = "plot_minidisc_analysis_{0}.png".format("_".join(chckname.split(".")[0].split("_")[1:]))

        print("Saving {0:s}...".format(outname))
        plt.savefig(outname)

    else:
        fig = None

    return fig, bounds
"""

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("\nGive me a parfile and checkpoint (.h5) file(s).\n")
        sys.exit()

    elif len(sys.argv) == 3:
        parname = sys.argv[1]
        filename = sys.argv[2]
        pars = dp.readParfile(parname)
        fig = plot_r_profile(filename, pars, sca=scale)
        plt.show()

    else:
        all_bounds = np.zeros((9,2))
        parname = sys.argv[1]
        pars = dp.readParfile(parname)
        #all_bounds[:,0] = np.inf
        #all_bounds[:,1] = -np.inf
        #for filename in sys.argv[1:]:
            #fig, bounds = plot_r_profile(filename, pars, sca=scale,
            #                                plot=False)
            #fig, bounds = plot_r_profile(filename, pars, sca=scale,
            #                                plot=False)
            #all_bounds[:,0] = np.minimum(all_bounds[:,0], bounds[:,0])
            #all_bounds[:,1] = np.maximum(all_bounds[:,1], bounds[:,1])

        for filename in sys.argv[2:]:
            fig, bounds = plot_r_profile(filename, pars, sca=scale, plot=True,
                                        bounds=all_bounds)
            plt.close(fig)
