import math
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.signal as signal
import sys
import numpy as np
import discopy as dp
import discoEOS as eos
import discoGR as gr

scale = 'log'
RMAX = 70.0
RMIN = 6.0

blue = (31.0/255, 119.0/255, 180.0/255)
orange = (255.0/255, 127.0/255, 14.0/255)
green = (44.0/255, 160.0/255, 44.0/255)
red = (214.0/255, 39.0/255, 40.0/255)
purple = (148.0/255, 103.0/255, 189.0/255)

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

def dissipation_plot(t, r, phi, sig, pi, vr, vp, u0, dphi, name, pars):

    Rs = np.unique(r)
    M = pars['GravM']
    gam = pars['Adiabatic_Index']

    S = np.log(pi * np.power(sig, -gam)) / (gam-1.0)

    S -= S.min()

    up = u0*vp
    N = Rs.shape[0]

    dS = np.zeros((N,3))
    psiQ = np.zeros((N,3))
    dQdm = np.zeros((N,3))
    dQdr = np.zeros((N,3))

    for i,R in enumerate(Rs):
        ind = r==R
        s = S[ind]
        sl = np.roll(s,1)
        sll = np.roll(s,2)
        sr = np.roll(s,-1)
        srr = np.roll(s,-2)
        d2s = sr - 2*s + sl
        d2s = -srr + 16*sr - 30*s + 16*sl - sll

        maxinds = signal.argrelmax(d2s, order=10, mode='wrap')[0]
        mininds = signal.argrelmin(d2s, order=10, mode='wrap')[0]
        maxinds = maxinds[np.argsort(d2s[maxinds])[::-1]]
        mininds = mininds[np.argsort(d2s[mininds])]

        if len(maxinds) == 0:
            maxinds = [0,0]
            mininds = [0,0]
        if len(maxinds) == 1:
            maxinds = [maxinds[0], 0]
            mininds = [mininds[0], 0]

        isa1 = maxinds[0]
        isb1 = maxinds[1]

        diffa = phi[mininds[0]] - phi[isa1]
        diffb = phi[mininds[1]] - phi[isa1]

        if diffa < -np.pi:
            diffa += 2*np.pi
        elif diffa > np.pi:
            diffa -= 2*np.pi
        if diffb < -np.pi:
            diffb += 2*np.pi
        elif diffb > np.pi:
            diffb -= 2*np.pi

        if np.fabs(diffa) < np.fabs(diffb):
            isa2 = mininds[0]
            isb2 = mininds[1]
        else:
            isa2 = mininds[1]
            isb2 = mininds[0]

        dsa = s[isa2] - s[isa1]
        dsb = s[isb2] - s[isb1]
        siga = sig[ind][isa1]
        sigb = sig[ind][isb1]
        Ta = pi[ind][isa1] / siga
        Tb = pi[ind][isb1] / sigb

        upa = u0[ind][isa1]*vp[ind][isa1]
        upb = u0[ind][isb1]*vp[ind][isb1]

        dS[i,0] = dsa
        dS[i,1] = dsb
        dS[i,2] = dsa+dsb
        psiQ[i,0] = (math.exp((gam-1)*dsa)-1) / (gam-1.0)
        psiQ[i,1] = (math.exp((gam-1)*dsb)-1) / (gam-1.0)
        psiQ[i,2] = psiQ[i,0] + psiQ[i,1]
        dQdm[i,0] = Ta * psiQ[i,0]
        dQdm[i,1] = Ta * psiQ[i,1]
        dQdm[i,2] = dQdm[i,0] + dQdm[i,1]
        dQdr[i,0] = siga * upa * dQdm[i,0]
        dQdr[i,1] = sigb * upb * dQdm[i,1]
        dQdr[i,2] = dQdr[i,0] + dQdr[i,1]

    fig, ax = plt.subplots(4, 1, figsize=(12,9))
    ax[0].plot(Rs, dS[:,0], 'r+')
    ax[0].plot(Rs, dS[:,1], 'b+')
    ax[0].plot(Rs, dS[:,2], 'k+')
    ax[0].set_ylabel(r'$\Delta s$')
    ax[0].set_xscale('log')
    if (dS > 0).any():
        ax[0].set_yscale('log')
    ax[1].plot(Rs, psiQ[:,0], 'r+')
    ax[1].plot(Rs, psiQ[:,1], 'b+')
    ax[1].plot(Rs, psiQ[:,2], 'k+')
    ax[1].set_ylabel(r'$\psi_Q$')
    ax[1].set_xscale('log')
    if (psiQ > 0).any():
        ax[1].set_yscale('log')
    ax[2].plot(Rs, dQdm[:,0], 'r+')
    ax[2].plot(Rs, dQdm[:,1], 'b+')
    ax[2].plot(Rs, dQdm[:,2], 'k+')
    ax[2].set_ylabel(r'$\frac{dQ}{dm}$')
    ax[2].set_xscale('log')
    if (dQdm > 0).any():
        ax[2].set_yscale('log')
    ax[3].plot(Rs, dQdr[:,0], 'r+')
    ax[3].plot(Rs, dQdr[:,1], 'b+')
    ax[3].plot(Rs, dQdr[:,2], 'k+')
    ax[3].set_xlabel(r'$r$')
    ax[3].set_ylabel(r'$\frac{dQ}{dr}$')
    ax[3].set_xscale('log')
    if (dQdr > 0).any():
        ax[3].set_yscale('log')

    fig.savefig("plot_minidisc_psi_{0}.png".format(name))
    plt.close(fig)



    R1 = Rs[0]
    R2 = Rs[N/4]
    R3 = Rs[N/2]
    R4 = Rs[3*N/4]
    R5 = Rs[N-1]
    
    RR = np.array([R1, R2, R3, R4, R5])

    fig1, ax1 = plt.subplots(7, RR.shape[0], figsize=(50,40))

    for i,R in enumerate(RR):
        ind = r==R
        s = S[ind]
        sl = np.roll(s,1)
        sll = np.roll(s,2)
        sr = np.roll(s,-1)
        srr = np.roll(s,-2)
        d2s = sr - 2*s + sl
        d2s = -srr + 16*sr - 30*s + 16*sl - sll

        ax2 = ax1[0,i].twinx()
        maxinds = signal.argrelmax(d2s, order=10, mode='wrap')[0]
        mininds = signal.argrelmin(d2s, order=10, mode='wrap')[0]
        maxinds = maxinds[np.argsort(d2s[maxinds])[::-1]]
        mininds = mininds[np.argsort(d2s[mininds])]
        for j in xrange(ax.shape[0]):
            if maxinds.shape[0] > 1:
                ax1[j,i].axvline(phi[ind][maxinds[0]], color='b')
            if maxinds.shape[0] > 2:
                ax1[j,i].axvline(phi[ind][maxinds[1]], color='b')
            if mininds.shape[0] > 1:
                ax1[j,i].axvline(phi[ind][mininds[0]], color='g')
            if mininds.shape[0] > 2:
                ax1[j,i].axvline(phi[ind][mininds[1]], color='g')
        ax1[0,i].plot(phi[ind], d2s, 'r+')
        ax2.plot(phi[ind], S[ind], 'k+')

        ax1[1,i].plot(phi[ind], sig[ind]*S[ind], 'k+')
        ax1[2,i].plot(phi[ind], sig[ind]*u0[ind]*S[ind], 'k+')
        ax1[3,i].plot(phi[ind], sig[ind]*up[ind]*S[ind], 'k+')
        ax1[4,i].plot(phi[ind], sig[ind], 'k+')
        ax1[5,i].plot(phi[ind], pi[ind], 'k+')
        ax1[6,i].plot(phi[ind], pi[ind]/sig[ind], 'k+')
        ax1[6,i].set_xlabel(r"$\phi$")

    ax1[0,0].set_ylabel(r"$s$")
    ax1[1,0].set_ylabel(r"$\Sigma_0 s$")
    ax1[2,0].set_ylabel(r"$\Sigma_0 u^0 s$")
    ax1[3,0].set_ylabel(r"$\Sigma_0 u^\phi s$")
    ax1[4,0].set_ylabel(r"$\Sigma_0$")
    ax1[5,0].set_ylabel(r"$\Pi$")
    ax1[6,0].set_ylabel(r"$\Pi / \Sigma$")

    fig.savefig("plot_minidisc_diss_{0}.png".format(name))
    plt.close(fig)

    return psiQ


def plot_r_profile(filename, pars, sca='linear', plot=True, bounds=None):

    print("Reading {0:s}".format(filename))
    
    M = pars['GravM']
    a = pars['GravA']
    gam = pars['Adiabatic_Index']
    bw = pars['BinW']
    A = a*M

    t, r, phi, rho, sig, T, P, pi, H, vr, vp, u0, q, dphi = allTheThings(
                                                                filename, pars)
    
    inds = (r < RMAX*M) * (r > RMIN*M)
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
    
    outpath = filename.split("/")[:-1]
    chckname = filename.split("/")[-1]
    chcknum = "_".join(chckname.split(".")[0].split("_")[1:])

    psiQ = dissipation_plot(t, r, phi, sig, pi, vr, vp, u0, dphi, 
                            chcknum, pars)

    RR = np.logspace(np.log10(r.min()), np.log10(r.max()), 100)

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
    vrsc = ur/u0sc
    vpsc = up_lab/u0sc

    #Mdot = - r*sig*u0*vr * (eos.c * eos.rg_solar**2 * eos.year
    #                                    / eos.M_solar)
    #Mdot = - r*sig*ur

    #cs2 = eos.cs2(rho, T, pars)
    gam = pars['Adiabatic_Index']
    sigh = sig + gam/(gam-1.0)*pi
    cs2 = gam*pi/sigh
    cs = np.sqrt(cs2)
    ucs = np.sqrt(cs2 / (1-cs2))
    S = np.log(pi * np.power(sig, -gam)) / (gam-1.0)

    mach = u / ucs
    machNorb = r*vpsc / cs

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
    avMachNorb = np.zeros(Rs.shape)

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

    shock0Phi = np.zeros(Rs.shape[0])
    shock1Phi = np.zeros(Rs.shape[0])

    for i,R in enumerate(Rs):

        inds = (r==R)
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
        avMachNorb[i] = (sig[inds]*machNorb[inds] * R*dphi[inds]).sum()/sigtot

        Mdot[i] = -Dflux
        Ldot[i] = -Lflux
        Edot[i] = -Eflux
        Lindot[i] = sighflux * avupd
        avVflux[i] = Vflux  #Not actually average, just go with it.

        ph = phi[inds]
        a0 = (D*dphi[inds]).sum()/(2*np.pi)
        an = (D[None,:] * np.cos(nm[:,None]*ph[None,:]) * dphi[inds][None,:]
                ).sum(axis=1) / np.pi
        bn = (D[None,:] * np.sin(nm[:,None]*ph[None,:]) * dphi[inds][None,:]
                ).sum(axis=1) / np.pi
        A0[i] = a0
        An[i,:] = np.sqrt(an*an + bn*bn)
        PhiN[i,:] = np.arctan2(bn,an) / nm

        #sortinds = np.argsort(phi[inds])
        #sigi = sig[inds][sortinds]
        #phii = phi[inds][sortinds]
        #pii = pi[inds][sortinds]
        sigi = sig[inds]
        phii = phi[inds]
        pii = pi[inds]
        maxinds = signal.argrelmax(pii, order=20, mode='wrap')[0]
        #print len(maxinds), phii[maxinds], pii[maxinds]
        phimaxima = phii[maxinds]

        if len(maxinds) == 0:
            shock0Phi[i] = np.inf
            shock1Phi[i] = np.inf
        elif len(maxinds) == 1:
            shock0Phi[i] = phimaxima
            shock1Phi[i] = np.inf
        elif i > 0:
            last0 = shock0Phi[i-1]
            last1 = shock1Phi[i-1]
            if last0 == np.inf:
                shock0Phi[i] = phimaxima[0]
                shock1Phi[i] = phimaxima[1]
            elif last1 == np.inf:
                diff0 = phimaxima - last0
                diff0[diff0>np.pi] -= 2*np.pi
                diff0[diff0<-np.pi] += 2*np.pi
                i0 = np.argmin(np.fabs(diff0))
                if i0 == 1:
                    i1 = 0
                else:
                    i1 = 1
                shock0Phi[i] = phimaxima[i0]
                shock1Phi[i] = phimaxima[i1]
            else:
                diff0 = phimaxima - last0
                diff0[diff0>np.pi] -= 2*np.pi
                diff0[diff0<-np.pi] += 2*np.pi
                diff1 = phimaxima - last1
                diff1[diff1>np.pi] -= 2*np.pi
                diff1[diff1<-np.pi] += 2*np.pi

                i0 = np.argmin(np.fabs(diff0))
                i1 = np.argmin(np.fabs(diff1))

                shock0Phi[i] = phimaxima[i0]
                shock1Phi[i] = phimaxima[i1]
        else:
            shock0Phi[i] = phimaxima[0]
            shock1Phi[i] = phimaxima[1]


    j = Ldot/Mdot
    e = Edot/Mdot
    jin = Lindot/Mdot
    jout = j - jin
    
    alpha = -2.0/(3.0*Rs) *  Lindot / avVflux
    alpha *= (1-3*M/Rs) / ((1-2*M/Rs)*(1-2*M/Rs)) 

    dphi0dr = (shock0Phi[2:] - shock0Phi[:-2]) / (Rs[2:] - Rs[:-2])
    dphi1dr = (shock1Phi[2:] - shock1Phi[:-2]) / (Rs[2:] - Rs[:-2])
    tanPitch0 = -1.0 / (dphi0dr * Rs[1:-1])
    tanPitch1 = -1.0 / (dphi1dr * Rs[1:-1])

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
    Mdot_raf = 2 * avD[1:-1] * Rs[1:-1] * Pi0[1:-1]/Sig0[1:-1] \
            * psiQ[1:-1][:,2] / djdr

    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.plot(Rs, Mdot, 'k+')
    ax.plot(Rs[1:-1], Mdot_raf, 'b+')
    #ax.plot(Rs[1:-1], Mdot_raf_sig, 'g+')
    #ax.plot(Rs[1:-1], Mdot_raf_rho, 'r+')
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\dot{M}$")
    ax.set_xlim(Rs.min(), Rs.max())
    ax.set_xscale("log")
    ax.set_yscale("log")

    outname = "plot_minidisc_rafikov_{0}.png".format(chcknum)
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
    
    fig, ax = plt.subplots(3,3, figsize=(12,9))
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
            yscale='log', xscale='log')
    plot_data(ax[2,0], Rs, shock0Phi)
    plot_data(ax[2,0], Rs, shock1Phi, 'b')
    pretty_axis(ax[2,0], pars, xlabel=r"$R$", ylabel=r"$\phi_{S}$", 
            yscale='linear')
    if (tanPitch0 < np.inf).any() or (tanPitch1 < np.inf).any():
        real0 = (tanPitch0 > 0) * (tanPitch0 < np.inf)
        real1 = (tanPitch1 > 0) * (tanPitch1 < np.inf)
        TPmin0 = 1.0e300
        TPmax0 = -1.0
        TPmin1 = 1.0e300
        TPmax1 = -1.0
        coeff0 = None
        coeff1 = None
        if len(tanPitch0[real0]) > 0:
            TPmin0 = tanPitch0[real0].min()
            TPmax0 = tanPitch0[real0].max()
            coeff0 = np.polyfit(np.log10(avMach[1:-1][real0]), 
                    np.log10(tanPitch0[real0]), 1)
            print coeff0
        if len(tanPitch1[real1]) > 0:
            TPmin1 = tanPitch1[real1].min()
            TPmax1 = tanPitch1[real1].max()
            coeff1 = np.polyfit(np.log10(avMach[1:-1][real1]), 
                    np.log10(tanPitch1[real1]), 1)
            print coeff1
        TP = np.linspace(min(TPmin0,TPmin1), max(TPmax0,TPmax1), 100)
        plot_data(ax[2,1], Rs[1:-1], tanPitch0)
        plot_data(ax[2,1], Rs[1:-1], tanPitch1, 'b')
        pretty_axis(ax[2,1], pars, xlabel=r"$R$", ylabel=r"$\tan \theta_S$", 
                yscale='log', xscale='log')
        plot_data(ax[2,2], tanPitch0, alpha[1:-1])
        plot_data(ax[2,2], tanPitch1, alpha[1:-1], 'b')
        pretty_axis(ax[2,2], pars, xlabel=r"$\tan \theta_S$",
                        ylabel=r"$\alpha$", yscale='log', xscale='log')

        plot_data(ax[1,2], avMach[1:-1], tanPitch0)
        plot_data(ax[1,2], avMach[1:-1], tanPitch1, 'b')
        #plot_line(ax[1,2], avMach[1]*np.power(TP/TP[1], -1.5), TP, color='r')
        #plot_line(ax[1,2], avMach[1]*np.power(TP/TP[1], -1.0), TP, color='r')
        #plot_line(ax[1,2], avMach[1]*np.power(TP/TP[1], -2.0), TP, color='r')
        #if coeff0 is not None:
        #    plot_line(ax[1,2], math.pow(10.0,coeff0[1])*np.power(TP,coeff0[0]),
        #                    TP, color='g')
        #if coeff1 is not None:
        #    plot_line(ax[1,2], math.pow(10.0,coeff1[1])*np.power(TP,coeff1[0]),
        #                    TP, color='g')
        omk = np.sqrt(M/(Rs[1:-1]*Rs[1:-1]*Rs[1:-1]))
        tpWKB = 1.0 / (avMach[1:-1] * np.sqrt((1-bw/omk)*(1-bw/omk)-0.25))
        tpWKB1 = 1.0 / (avMach[1:-1] * np.sqrt((1-bw/omk)*(1-bw/omk)-0.6))
        tpWKB2 = 1.0 / (avMach[1:-1] * np.sqrt((1-bw/omk)*(1-bw/omk)-0.7))
        tpWKBrel = 1.0 / avMachNorb[1:-1] * np.sqrt(
                (1-2*M/Rs[1:-1])*(1-3*M/Rs[1:-1])
            / ((1-bw/omk)*(1-bw/omk)-0.25*(1-6*M/Rs[1:-1])/(1-2*M/Rs[1:-1])))
        tpWKBrel1 = 1.0 / avMachNorb[1:-1] * np.sqrt(
                (1-2*M/Rs[1:-1])*(1-3*M/Rs[1:-1])
            / ((1-bw/omk)*(1-bw/omk)-0.6*(1-6*M/Rs[1:-1])/(1-2*M/Rs[1:-1])))
        tpWKBrel2 = 1.0 / avMachNorb[1:-1] * np.sqrt(
                (1-2*M/Rs[1:-1])*(1-3*M/Rs[1:-1])
            / ((1-bw/omk)*(1-bw/omk)-0.7*(1-6*M/Rs[1:-1])/(1-2*M/Rs[1:-1])))
        plot_line(ax[1,2], avMach[1:-1], tpWKB, color='r')
        #plot_line(ax[1,2], avMach[1:-1], tpWKB1, color='r', ls='--')
        #plot_line(ax[1,2], avMach[1:-1], tpWKB2, color='r', ls=':')
        plot_line(ax[1,2], avMach[1:-1], tpWKBrel, color='g')
        #plot_line(ax[1,2], avMach[1:-1], tpWKBrel1, color='g', ls='--')
        plot_line(ax[1,2], avMach[1:-1], tpWKBrel2, color='g', ls='--')

        pretty_axis(ax[1,2], pars, xlabel=r"$\mathcal{M}$",
                        ylabel=r"$\tan \theta_S$", yscale='log', xscale='log')

        figNice, axNice = plt.subplots(1,1,figsize=(12,9))
        axNice.plot(avMachNorb[1:-1], tpWKBrel, '-', lw=4, color="grey", 
                        label="WKB")
        axNice.plot(avMachNorb[1:-1][tanPitch0>0], tanPitch0[tanPitch0>0], 
                        '+', color=blue, ms=10, mew=2, label="Shock 1")
        axNice.plot(avMachNorb[1:-1][tanPitch1>0], tanPitch1[tanPitch1>0], 
                        '+', color=orange, ms=10, mew=2, label="Shock 2")
        axNice.set_xlabel(r"$\mathcal{M}_N = \langle r v^\phi / c_s \rangle$",
                            fontsize=24)
        axNice.set_ylabel(r"$\tan \theta_S$", fontsize=24)
        #axNice.set_xscale('log')
        #axNice.set_yscale('log')
        axNice.tick_params(labelsize=18)
        plt.legend(loc="upper right", fontsize=24)
        axNice.set_title("Dispersion Relation", fontsize=36)
        outname = "plot_minidisc_tanq_mach_{0}.png".format(
                    "_".join(chckname.split(".")[0].split("_")[1:]))
        print("Saving {0:s}...".format(outname))
        figNice.savefig(outname)
        plt.close(figNice)


    figNice, axNice = plt.subplots(1,1,figsize=(12,9))
    axNice.plot(Rs, alpha, '+', ms=10, mew=2, color=blue, label=r"$\dot{M}$")
    axNice.plot(Rs, psiQ[:,2]/(1.5*np.pi), '+', ms=10, mew=2, color=orange, 
                    label=r"\delta s")
    axNice.set_xlabel(r"$r$", fontsize=24)
    axNice.set_ylabel(r"$\alpha$", fontsize=24)
    axNice.set_xscale('log')
    axNice.set_yscale('log')
    axNice.tick_params(labelsize=18)
    axNice.set_title(r"$\alpha$ Parameter", fontsize=36)
    outname = "plot_minidisc_alpha_r_{0}.png".format(
                "_".join(chckname.split(".")[0].split("_")[1:]))
    print("Saving {0:s}...".format(outname))
    figNice.savefig(outname)
    plt.close(figNice)


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
