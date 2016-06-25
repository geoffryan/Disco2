import math
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.signal as signal
import scipy.integrate as integrate
import sys
import numpy as np
import discopy as dp
import discoEOS as eos
import discoGR as gr

scale = 'log'
RMAX = 100.0
RMIN = 6.0

shockDetPlot = True
dissPlot = True
allTheOtherPlots = True

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

def shockPlot(r, phi, dphi, sig, pi, u0, vr, vp, name, pars, slopelimit=False,
                TH=2.0):

    v12RS, v21RS, vRS, vSS, v12 = shockVal(r, phi, sig, pi, u0, vr, vp, pars,
                                            slopelimit=slopelimit, TH=TH)

    gam = pars['Adiabatic_Index']
    S = np.log(pi * np.power(sig, -gam)) / (gam-1.0)
    dSdr, dSdp = calcGrad(r, phi, phi+dphi, S, pars, slopelimit=False)


    fig, ax = plt.subplots(2,4, figsize=(40,20))

    Rs = np.unique(r)
    Nr = Rs.shape[0]

    chi = (v12-vRS) / (vSS-vRS)
    dvRSmax = (v12-vRS).max()
    dvRSmin = (v12-vRS).min()
    dvSSmax = (v12-vSS).max()
    dvSSmin = (v12-vSS).min()
    dvSSRSmin = (vSS-vRS).min()
    dvSSRSmax = (vSS-vRS).max()

    chi[chi<0] = chi[chi>0].min()
    lchi = np.log10(chi)

    for i, R in enumerate(Rs):
        ind = r==R
        N = r[ind].shape[0]

        sind = np.argmax(dSdp[ind])

        if i>0:
            rm = 0.5*(R+Rs[i-1])
        else:
            rm = R - 0.5*(Rs[1]-R)
        if i<Nr-1:
            rp = 0.5*(R+Rs[i+1])
        else:
            rp = R + 0.5*(R-Rs[i-1])

        X = np.zeros((2, N+1))
        Y = np.zeros((2, N+1))
        X[:,1:] = (phi[ind] + 0.5*dphi[ind])[None,:]
        X[:,0] = phi[ind][0] - 0.5*dphi[ind][0]
        Y[0,:] = rm
        Y[1,:] = rp

        print("Plotting shocks: {0:d} of {1:d}".format(i+1,Rs.shape[0]))
        #print(X.shape)
        #print(Y.shape)
        #print(v12RS.shape)
        #print(v21RS.shape)
        #print(v12.shape)


        C0 = ax[0,0].pcolormesh(X, Y, np.atleast_2d(vRS[ind]), 
                        edgecolors='none', vmin=vRS.min(), vmax=vRS.max(),
                        cmap=dp.viridis)
        C1 = ax[0,1].pcolormesh(X, Y, np.atleast_2d(vSS[ind]), 
                        edgecolors='none', vmin=vSS.min(), vmax=vSS.max(),
                        cmap=dp.viridis)
        C2 = ax[0,2].pcolormesh(X, Y, np.atleast_2d(v12[ind]), 
                        edgecolors='none', vmin=v12.min(), vmax=v12.max(),
                        cmap=dp.viridis)
        C3 = ax[1,0].pcolormesh(X, Y, np.atleast_2d(v12[ind]-vRS[ind]), 
                        edgecolors='none', vmin=0.0, vmax=dvRSmax,
                        cmap=dp.viridis)
        C4 = ax[1,1].pcolormesh(X, Y, np.atleast_2d(v12[ind]-vSS[ind]), 
                        edgecolors='none', vmin=dvSSmin, vmax=dvSSmax,
                        cmap=dp.viridis)
        C5 = ax[1,2].pcolormesh(X, Y, np.atleast_2d(vSS[ind]-vRS[ind]), 
                        edgecolors='none', vmin=dvSSRSmin, vmax=dvSSRSmax,
                        cmap=dp.viridis)
        C6 = ax[1,3].pcolormesh(X, Y, np.atleast_2d(lchi[ind]), 
                        edgecolors='none', 
                        vmin=lchi.min(),
                        vmax=lchi.max(),
                        #vmin=math.log10(chi.min()), 
                        #vmax=math.log10(chi.max()),
                        cmap=dp.viridis)
        
        ax[0,0].plot(phi[ind][sind], r[ind][sind], 'r+')
        ax[0,1].plot(phi[ind][sind], r[ind][sind], 'r+')
        ax[0,2].plot(phi[ind][sind], r[ind][sind], 'r+')
        ax[1,0].plot(phi[ind][sind], r[ind][sind], 'r+')
        ax[1,1].plot(phi[ind][sind], r[ind][sind], 'r+')
        ax[1,2].plot(phi[ind][sind], r[ind][sind], 'r+')
        ax[1,3].plot(phi[ind][sind], r[ind][sind], 'r+')

    fig.colorbar(C0, ax=ax[0,0])
    fig.colorbar(C1, ax=ax[0,1])
    fig.colorbar(C2, ax=ax[0,2])
    fig.colorbar(C3, ax=ax[1,0])
    fig.colorbar(C4, ax=ax[1,1])
    fig.colorbar(C5, ax=ax[1,2])
    fig.colorbar(C6, ax=ax[1,3])

    ax[0,0].set_xlabel(r'$\phi$')
    ax[0,1].set_xlabel(r'$\phi$')
    ax[0,2].set_xlabel(r'$\phi$')
    ax[1,0].set_xlabel(r'$\phi$')
    ax[1,1].set_xlabel(r'$\phi$')
    ax[1,2].set_xlabel(r'$\phi$')
    ax[1,3].set_xlabel(r'$\phi$')
    ax[0,0].set_ylabel(r'$r$')
    ax[0,1].set_ylabel(r'$r$')
    ax[0,2].set_ylabel(r'$r$')
    ax[1,0].set_ylabel(r'$r$')
    ax[1,1].set_ylabel(r'$r$')
    ax[1,2].set_ylabel(r'$r$')
    ax[1,3].set_ylabel(r'$r$')
    ax[0,0].set_title(r'$(v_{12})_\mathcal{RS}$')
    ax[0,1].set_title(r'$(v_{12})_\mathcal{2S}$')
    ax[0,2].set_title(r'$v_{12}$')
    ax[1,0].set_title(r'$v_{12} - (v_{12})_\mathcal{RS}$')
    ax[1,1].set_title(r'$v_{12} - (v_{12})_\mathcal{2S}$')
    ax[1,2].set_title(r'$(v_{12})_\mathcal{2S} - (v_{12})_\mathcal{RS}$')
    ax[1,3].set_title(r'$(v_{12} - (v_{12})_\mathcal{RS}) / ((v_{12})_\mathcal{2S} - (v_{12})_\mathcal{RS})$')
    ax[0,0].set_ylim(r.min(), r.max())
    ax[0,1].set_ylim(r.min(), r.max())
    ax[0,2].set_ylim(r.min(), r.max())
    ax[1,0].set_ylim(r.min(), r.max())
    ax[1,1].set_ylim(r.min(), r.max())
    ax[1,2].set_ylim(r.min(), r.max())
    ax[1,3].set_ylim(r.min(), r.max())
    ax[0,0].set_yscale('log')
    ax[0,1].set_yscale('log')
    ax[0,2].set_yscale('log')
    ax[1,0].set_yscale('log')
    ax[1,1].set_yscale('log')
    ax[1,2].set_yscale('log')
    ax[1,3].set_yscale('log')

    figname = "plot_minidisc_shockDet_{0}.png".format(name)
    print("Saving {0:s}...".format(figname))
    fig.savefig(figname)
    plt.close(fig)

    return vRS, vSS, v12


def shockVal(r, phi, sig, pi, u0, vr, vp, pars, slopelimit=True, TH=2.0):

    Rs = np.unique(r)

    v12 = np.zeros(r.shape)
    v12RS = np.zeros(r.shape)
    v21RS = np.zeros(r.shape)
    vRS = np.zeros(r.shape)
    vSS = np.zeros(r.shape)

    M = pars['GravM']
    gam = pars['Adiabatic_Index']
    bw = pars['BinW']

    lapse = 1.0/np.sqrt(1+2*M/r)
    shiftr = 2*M / (r*(1+2*M/r))
    shiftp = bw

    W = lapse * u0
    VR = np.sqrt(1.0+2*M/r) * (vr + shiftr)/lapse
    VP = r * (vp + shiftp)/lapse

    print np.fabs(W - 1.0/np.sqrt(1-VR*VR-VP*VP)).sum() / r.shape[0]

    for i,R in enumerate(Rs):
        ind = r==R

        #print W-1.0/np.sqrt(1.0-VR*VR-VP*VP)

        print("Shock analysis: {0:d} of {1:d}".format(i+1,Rs.shape[0]))

        N = r[ind].shape[0]

        v12_s = np.zeros(N)
        v12RS_s = np.zeros(N)
        v21RS_s = np.zeros(N)
        vRS_s = np.zeros(N)
        vSS_s = np.zeros(N)

        dsig = np.zeros(N)
        dpi = np.zeros(N)
        dVR = np.zeros(N)
        dVP = np.zeros(N)

        phiC = phi[ind]
        phiR = np.roll(phiC, -1)
        phiL = np.roll(phiC, 1)
        phiR[phiR<phiC] += 2*np.pi
        phiL[phiL>phiC] -= 2*np.pi
        dsigC = (np.roll(sig[ind],-1) - np.roll(sig[ind],1)) / (phiR-phiL)
        dpiC = (np.roll(pi[ind],-1) - np.roll(pi[ind],1)) / (phiR-phiL)
        dVRC = (np.roll(VR[ind],-1) - np.roll(VR[ind],1)) / (phiR-phiL)
        dVPC = (np.roll(VP[ind],-1) - np.roll(VP[ind],1)) / (phiR-phiL)
        dsigL = (sig[ind] - np.roll(sig[ind],1)) / (phiC-phiL)
        dpiL = (pi[ind] - np.roll(pi[ind],1)) / (phiC-phiL)
        dVRL = (VR[ind] - np.roll(VR[ind],1)) / (phiC-phiL)
        dVPL = (VP[ind] - np.roll(VP[ind],1)) / (phiC-phiL)
        dsigR = (np.roll(sig[ind],-1) - sig[ind]) / (phiR-phiC)
        dpiR = (np.roll(pi[ind],-1) - pi[ind]) / (phiR-phiC)
        dVRR = (np.roll(VR[ind],-1) - VR[ind]) / (phiR-phiC)
        dVPR = (np.roll(VP[ind],-1) - VP[ind]) / (phiR-phiC)

        if slopelimit:
            dsig = minmod(TH*dsigL, dsigC, TH*dsigR)
            dpi = minmod(TH*dpiL, dpiC, TH*dpiR)
            dVR = minmod(TH*dVRL, dVRC, TH*dVRR)
            dVP = minmod(TH*dVPL, dVPC, TH*dVPR)
        else:
            dsig = dsigC
            dpi = dpiC
            dVR = dVRC
            dVP = dVPC

        for j in xrange(N):
            iL = j
            iR = j+1
            if iR >= N:
                iR = 0
            dphiL = 0.5*(phiR[iL]-phiC[iL])
            dphiR = 0.5*(phiL[iR]-phiC[iR])
            sigL = sig[ind][iL] + dsig[iL]*dphiL
            piL = pi[ind][iL] + dpi[iL]*dphiL
            VRL = VR[ind][iL] + dVR[iL]*dphiL
            VPL = VP[ind][iL] + dVP[iL]*dphiL
            sigR = sig[ind][iR] + dsig[iR]*dphiR
            piR = pi[ind][iR] + dpi[iR]*dphiR
            VRR = VR[ind][iR] + dVR[iR]*dphiR
            VPR = VP[ind][iR] + dVP[iR]*dphiR

            v12RS_s[j] = relvRS(sigL, piL, VPL, VRL, sigR, piR, VPR, VRR, gam)
            v21RS_s[j] = relvRS(sigR, piR, VPR, VRR, sigL, piL, VPL, VRL, gam)

            if piR > piL:
                vRS_s[j] = v21RS_s[j]
                vSS_s[j] = relvSS(sigR, piR, -VPR, VRR, sigL, piL, -VPL, VRL, gam)
            else:
                vRS_s[j] = v12RS_s[j]
                vSS_s[j] = relvSS(sigL, piL, VPL, VRL, sigR, piR, VPR, VRR, gam)

            v12_s[j] = (VPL - VPR) / (1.0 - VPL*VPR)

        v12[ind] = v12_s[:]
        v12RS[ind] = v12RS_s[:]
        v21RS[ind] = v21RS_s[:]
        vRS[ind] = vRS_s[:]
        vSS[ind] = vSS_s[:]

    return v12RS, v21RS, vRS, vSS, v12


def relvRS(sig1, pi1, v1x, v1t, sig2, pi2, v2x, v2t, gam):

# One-shock limiting relative velocity
    s1 = math.log(pi1 * np.power(sig1, -gam)) / (gam-1.0)
    W1 = 1.0 / math.sqrt(1 - v1x*v1x - v1t*v1t)
    A1 = -(1.0 + gam*pi1/((gam-1)*sig1)) * W1 * v1t

    def func(p):
        sig =  sig1 * math.pow(p/pi1, 1.0/gam)
        h = 1 + gam*p/((gam-1)*sig)
        cs = math.sqrt(gam*p/(sig*h))
        return math.sqrt(h*h + A1*A1*(1-cs*cs)) / ((h*h+A1*A1)*sig*cs)

    res = integrate.quad(func, pi1, pi2)

#for i,r in enumerate(res):
#    print i,r

    return math.tanh(res[0])

def relvSS(sig1, pi1, v1x, v1t, sig2, pi2, v2x, v2t, gam):

# Two-shock limiting relative velocity
# These formula are pulled straight from Rezzolla, Zannotti, and Pons 2003.

    h1 = 1.0 + gam*pi1/((gam-1.0)*sig1)
    h2 = 1.0 + gam*pi2/((gam-1.0)*sig2)
    W22 = 1.0 / (1.0 - v2x*v2x - v2t*v2t)

#eq B.11
    D = 1 - 4*gam*pi1*((gam-1.0)*pi2+pi1)*(h2*(pi2-pi1)/sig2-h2*h2) / (
                (gam-1.0)*(gam-1.0)*(pi1-pi2)*(pi1-pi2))
#eq B.10
    h3p = (math.sqrt(D)-1.0)*(gam-1.0)*(pi1-pi2) / ( 2*((gam-1.0)*pi2+pi1))
#eq B.9
    J23p2 = -gam*(pi1-pi2) / ((gam-1.0)*(h3p*(h3p-1.0)/pi1-h2*(h2-1.0)/pi2))
    if J23p2 < 0:
        print("Whoa: J23p2 = {0:f}".format(J23p2))
        print sig1, pi1, v1x, v1t, sig2, pi2, v2x, v2t
        J23p2 = 0.0
#eq B.8
    Vs = (sig2*sig2*W22*v2x*v2x + math.sqrt(J23p2*(J23p2 + sig2*sig2*W22
                * (1-v2x*v2x)))) / (sig2*sig2*W22 + J23p2)
#eq 4.5
    return (pi1-pi2) * (1-v2x*Vs) / (
                (Vs-v2x) * (h2*sig2*W22*(1-v2x*v2x)+pi1-pi2))

def minmod(a, b, c):
    d = a.copy()
    bltd = np.fabs(b) < np.fabs(d)
    d[bltd] = b[bltd]
    cltd = np.fabs(c) < np.fabs(d)
    d[cltd] = c[cltd]
    d[a*b < 0] = 0.0
    d[a*c < 0] = 0.0
    return d

def calcGrad(r, phi, piph, f, pars, slopelimit=True, TH=2.0):

    g = dp.Grid(pars)
    rf = (g.rFaces).copy()

    Rs = np.unique(r)

    rf = rf[(rf>Rs.min()) * (rf < Rs.max())]
    
    dfdr = np.zeros(f.shape)
    dfdrC = np.zeros(f.shape)
    dA = np.zeros(f.shape)
    dfdp = np.zeros(f.shape)
    dfdpL = np.zeros(f.shape)
    dfdpR = np.zeros(f.shape)
    dfdpC = np.zeros(f.shape)
    
    for i,R in enumerate(Rs):
        ind = r==R
        phiC = phi[ind]
        phiR = np.roll(phiC, -1)
        phiL = np.roll(phiC, 1)
        phiR[phiR<phiC] += 2*np.pi
        phiL[phiL>phiC] -= 2*np.pi
        fC = f[ind]
        fR = np.roll(fC, -1)
        fL = np.roll(fC, 1)

        dfdpR[ind] = (fR - fC) / (phiR - phiC)
        dfdpC[ind] = (fR - fL) / (phiR - phiL)
        dfdpL[ind] = (fC - fL) / (phiC - phiL)

    if slopelimit:
        dfdp = minmod(TH*dfdpL, dfdpC, TH*dfdpR)
    else:
        dfdp = dfdpC

    for i in xrange(Rs.shape[0]-1):
        R1 = Rs[i]
        R2 = Rs[i+1]
        R = rf[i]
        if R > R2 or R < R1:
            print "Grid Wrong! {0} {1} {2}".format(R1, R, R2)

        ind1 = r==R1
        ind2 = r==R2

        phi1 = phi[ind1]
        phi2 = phi[ind2]
        piph1 = piph[ind1]
        piph2 = piph[ind2]
        f1 = f[ind1]
        f2 = f[ind2]
        dfdp1 = dfdp[ind1]
        dfdp2 = dfdp[ind2]

        dfdrC1_s = np.zeros(f1.shape)
        dfdrC2_s = np.zeros(f2.shape)
        dA1_s = np.zeros(f1.shape)
        dA2_s = np.zeros(f2.shape)

        j2 = 0
        phi1p = piph1[0]
        phi1m = piph1[-1]
        before = False
        after = False
        j2_start = -1
        for j2 in xrange(phi2.shape[0]):
            diff = piph2[j2]-phi1m
            if diff > np.pi:
                diff -= 2*np.pi
            elif diff < -np.pi:
                diff += 2*np.pi
            if diff < 0:
                before = True
            if before and diff > 0:
                j2_start = j2
        if j2_start == -1:
            j2_start = 0

        j1 = 0
        j2 = j2_start
        #j2_start is the first cell in i2 that shares a face with j1=0
        while j1 < phi1.shape[0]:
            phi1p = piph1[j1]
            phi1m = piph1[j1-1]
            phi2p = piph2[j2]
            phi2m = piph2[j2-1]
            if phi1m-phi1p > np.pi:
                phi1m -= 2*np.pi
            elif phi1m-phi1p < -np.pi:
                phi1m += 2*np.pi
            if phi2p-phi1p > np.pi:
                phi2p -= 2*np.pi
            elif phi2p-phi1p < -np.pi:
                phi2p += 2*np.pi
            if phi2m-phi1p > np.pi:
                phi2m -= 2*np.pi
            elif phi2m-phi1p < -np.pi:
                phi2m += 2*np.pi

            phiF = min(phi1p, phi2p)
            phiB = max(phi1m, phi2m)
            phiC = 0.5*(phiF + phiB)
            dphi = phiF - phiB
            dphi1 = phiF - phi1[j1]
            dphi2 = phiF - phi2[j2]
            if dphi1 > np.pi:
                dphi1 -= 2*np.pi
            elif dphi1 < -np.pi:
                dphi1 += 2*np.pi
            if dphi2 > np.pi:
                dphi2 -= 2*np.pi
            elif dphi2 < -np.pi:
                dphi2 += 2*np.pi

            fL = f1[j1] + dphi1*dfdp1[j1]
            fR = f2[j2] + dphi2*dfdp2[j2]

            Af = dphi*R

            dfdrC1_s[j1] += Af * (fR - fL) / (R2 - R1)
            dfdrC2_s[j2] += Af * (fR - fL) / (R2 - R1)
            dA1_s[j1] += Af
            dA2_s[j2] += Af

            if phi1p < phi2p:
                j1 += 1
            else:
                j2 += 1

            if j2 == phi2.shape[0]:
                j2 = 0
        dfdrC[ind1] += dfdrC1_s[:]
        dfdrC[ind2] += dfdrC2_s[:]
        dA[ind1] += dA1_s[:]
        dA[ind2] += dA2_s[:]

    dfdrC = dfdrC / dA

    if slopelimit:
        for i in xrange(Rs.shape[0]-1):
            R1 = Rs[i]
            R2 = Rs[i+1]
            R = rf[i]
            if R > R2 or R < R1:
                print "Grid Wrong! {0} {1} {2}".format(R1, R, R2)

            ind1 = r==R1
            ind2 = r==R2

            phi1 = phi[ind1]
            phi2 = phi[ind2]
            piph1 = piph[ind1]
            piph2 = piph[ind2]
            f1 = f[ind1]
            f2 = f[ind2]
            dfdp1 = dfdp[ind1]
            dfdp2 = dfdp[ind2]
            
            dfdr1_s = dfdrC[ind1].copy()
            dfdr2_s = dfdrC[ind2].copy()

            j2 = 0
            phi1p = piph1[0]
            phi1m = piph1[-1]
            before = False
            after = False
            j2_start = -1
            for j2 in xrange(phi2.shape[0]):
                diff = piph2[j2]-phi1m
                if diff > np.pi:
                    diff -= 2*np.pi
                elif diff < -np.pi:
                    diff += 2*np.pi
                if diff < 0:
                    before = True
                if before and diff > 0:
                    j2_start = j2
            if j2_start == -1:
                j2_start = 0

            j1 = 0
            j2 = j2_start
            #j2_start is the first cell in i2 that shares a face with j1=0
            while j1 < phi1.shape[0]:
                phi1p = piph1[j1]
                phi1m = piph1[j1-1]
                phi2p = piph2[j2]
                phi2m = piph2[j2-1]
                if phi1m-phi1p > np.pi:
                    phi1m -= 2*np.pi
                elif phi1m-phi1p < -np.pi:
                    phi1m += 2*np.pi
                if phi2p-phi1p > np.pi:
                    phi2p -= 2*np.pi
                elif phi2p-phi1p < -np.pi:
                    phi2p += 2*np.pi
                if phi2m-phi1p > np.pi:
                    phi2m -= 2*np.pi
                elif phi2m-phi1p < -np.pi:
                    phi2m += 2*np.pi

                phiF = min(phi1p, phi2p)
                phiB = max(phi1m, phi2m)
                phiC = 0.5*(phiF + phiB)
                dphi = phiF - phiB
                dphi1 = phiF - phi1[j1]
                dphi2 = phiF - phi2[j2]
                if dphi1 > np.pi:
                    dphi1 -= 2*np.pi
                elif dphi1 < -np.pi:
                    dphi1 += 2*np.pi
                if dphi2 > np.pi:
                    dphi2 -= 2*np.pi
                elif dphi2 < -np.pi:
                    dphi2 += 2*np.pi

                fL = f1[j1] + dphi1*dfdp1[j1]
                fR = f2[j2] + dphi2*dfdp2[j2]

                df = (fR - fL) / (R2 - R1)

                dfL = dfdr1_s[j1]
                dfR = dfdr2_s[j2]

                if df*dfL < 0:
                    dfdr1_s[j1] = 0.0
                elif np.fabs(TH*df) < np.fabs(dfL):
                    dfdr1_s[j1] = TH*df
                if df*dfR < 0:
                    dfdr2_s[j2] = 0.0
                elif np.fabs(TH*df) < np.fabs(dfR):
                    dfdr2_s[j2] = TH*df

                if phi1p < phi2p:
                    j1 += 1
                else:
                    j2 += 1

                if j2 == phi2.shape[0]:
                    j2 = 0

            dfdr[ind1] = dfdr1_s[:]
            dfdr[ind2] = dfdr2_s[:]

    else:
        dfdr = dfdrC

    return dfdr, dfdp


def find_shocks_d2s(r, phi, S):

#Find shocks by looking at derivatives of S.

    Rs = np.unique(r)
    N = Rs.shape[0]

    phiSa = np.zeros((N,2))
    phiSb = np.zeros((N,2))

    iSa = np.zeros((N,2),dtype=np.int)
    iSb = np.zeros((N,2),dtype=np.int)

    for i,R in enumerate(Rs):
        ind = r==R

        phiC = phi[ind]
        phiR = np.roll(phiC, -1)
        phiL = np.roll(phiC, 1)
        phiR[phiR<phiC] += 2*np.pi
        phiL[phiL>phiC] -= 2*np.pi
        dphiAve = (phiR-phiC).mean()

        s = S[ind]
        sl = np.roll(s,1)
        sll = np.roll(s,2)
        sr = np.roll(s,-1)
        srr = np.roll(s,-2)
        d2s = (sr - 2*s + sl) / (dphiAve*dphiAve)
        d2s = (-srr + 16*sr - 30*s + 16*sl - sll) / (dphiAve*dphiAve)

        smax = s.max()
        smin = s.min()

        d2sB = 2 * 2*(smax-smin)

        maxinds = signal.argrelmax(d2s, order=10, mode='wrap')[0]
        mininds = signal.argrelmin(d2s, order=10, mode='wrap')[0]
        if mininds.shape[0] > 0:
            mininds = mininds[np.argsort(d2s[mininds])]

        maxinds = maxinds[d2s[maxinds] >  d2sB]
        mininds = mininds[d2s[mininds] < -d2sB]

        #print d2s.min(), d2s.max(), d2sB

        iS = []

        used = []
        for j in mininds:
            is2 = j
            phi2 = phi[ind][j]
            diff = np.inf
            is1 = -1

            for k in maxinds:
                if k in used:
                    continue
                jump = phi2 - phi[ind][k]
                if jump > np.pi:
                    jump -= 2*np.pi
                if jump < -np.pi:
                    jump += 2*np.pi
                if jump > 0 and jump < diff:
                    is1 = k
                    diff = jump

            if diff > np.pi/4:
                continue

            iS.append([is1, is2])
            used.append(is1)

        #print iS

        if len(iS) == 0:
            iSa[i,:] = -1
            iSb[i,:] = -1
            phiSa[i,:] = np.inf
            phiSb[i,:] = np.inf
            continue

        if i == 0:
            iSa[i,:] = iS[0]
            phiSa[i,:] = phi[ind][iS[0]]
            if len(iS) > 1:
                iSb[i,:] = iS[1]
                phiSb[i,:] = phi[ind][iS[1]]
            else:
                iSb[i,:] = -1
                phiSb[i,:] = np.inf

        else:
            sa = -1
            sb = -1
            diffa = np.inf
            diffb = np.inf
            for j,sh in enumerate(iS):
                jumpa = phi[ind][sh[1]] - phiSa[i-1,1]
                jumpb = phi[ind][sh[1]] - phiSb[i-1,1]
                if jumpa > np.pi:
                    jumpa -= 2*np.pi
                elif jumpa < -np.pi:
                    jumpa += 2*np.pi
                if jumpb > np.pi:
                    jumpb -= 2*np.pi
                elif jumpb < -np.pi:
                    jumpb += 2*np.pi
                if math.fabs(jumpa) < math.fabs(diffa):
                    sa = j
                    diffa = jumpa
                if math.fabs(jumpb) < math.fabs(diffb):
                    sb = j
                    diffb = jumpb
            if iSa[i-1,0] < 0 and iSb[i-1,0] < 0:
                iSa[i,:] = iS[0]
                iSb[i,:] = iS[1]
                phiSa[i,:] = phi[ind][iS[0]]
                phiSb[i,:] = phi[ind][iS[1]]
            elif iSb[i-1,0] < 0:
                iSa[i,:] = iS[sa]
                phiSa[i,:] = phi[ind][iS[sa]]
                if len(iS) == 1:
                    iSb[i,:] = -1
                    phiSb[i,:] = np.inf
                elif sa == 0:
                    iSb[i,:] = 1
                    phiSb[i,:] = phi[ind][iS[1]]
                else:
                    iSb[i,:] = 0
                    phiSb[i,:] = phi[ind][iS[0]]
            elif iSa[i-1,0] < 0:
                iSb[i,:] = iS[sb]
                phiSb[i,:] = phi[ind][iS[sb]]
                if len(iS) == 1:
                    iSa[i,:] = -1
                    phiSa[i,:] = np.inf
                elif sb == 0:
                    iSa[i,:] = 1
                    phiSa[i,:] = phi[ind][iS[1]]
                else:
                    iSa[i,:] = 0
                    phiSa[i,:] = phi[ind][iS[0]]
            elif sa != sb and sa > -1 and sb > -1:
                iSa[i,:] = iS[sa]
                iSb[i,:] = iS[sb]
                phiSa[i,:] = phi[ind][iS[sa]]
                phiSb[i,:] = phi[ind][iS[sb]]
            else:
                if math.fabs(diffa) < math.fabs(diffb):
                    iSa[i,:] = iS[sa]
                    iSb[i,:] = -1
                    phiSa[i,:] = phi[ind][iS[sa]]
                    phiSb[i,:] = np.inf
                else:
                    iSa[i,:] = -1
                    iSb[i,:] = iS[sb]
                    phiSa[i,:] = np.inf
                    phiSb[i,:] = phi[ind][iS[sb]]

    for i in xrange(N):
        if iSa[i,0] > -1:
            dphia = phiSa[i,1] - phiSa[i,0]
            if dphia > np.pi:
                phiSa[i,0] += 2*np.pi
            elif dphia < -np.pi:
                phiSa[i,0] -= 2*np.pi
        if iSb[i,0] > -1:
            dphib = phiSb[i,1] - phiSb[i,0]
            if dphib > np.pi:
                phiSb[i,0] += 2*np.pi
            elif dphib < -np.pi:
                phiSb[i,0] -= 2*np.pi

    return iSa, iSb, phiSa, phiSb

def find_shocks_d2sDet(r, phi, dphi, S, dV12):

#Find shocks by looking at derivatives of S AND the Rezzolla&Zanotti quantity.

    Rs = np.unique(r)
    N = Rs.shape[0]

    phiSa = np.zeros((N,2))
    phiSb = np.zeros((N,2))

    iSa = np.zeros((N,2),dtype=np.int)
    iSb = np.zeros((N,2),dtype=np.int)

    RsI = [Rs[0], Rs[N/5], Rs[(2*N)/5], Rs[(3*N)/5], Rs[(4*N)/5], Rs[-20]]
    fig, ax = plt.subplots(2, 3, figsize=(24,9))

    for i, ax in enumerate(ax.flat):
        R = RsI[i]
        ind = r==R

        phiC = phi[ind]
        phiR = np.roll(phiC, -1)
        phiL = np.roll(phiC, 1)
        phiR[phiR<phiC] += 2*np.pi
        phiL[phiL>phiC] -= 2*np.pi
        dphiAve = (phiR-phiC).mean()

        s = S[ind]
        sl = np.roll(s,1)
        sll = np.roll(s,2)
        sr = np.roll(s,-1)
        srr = np.roll(s,-2)
        d2s = (-srr + 16*sr - 30*s + 16*sl - sll) / (dphiAve*dphiAve)
        d1s = (sr - sl) / (phiR - phiL)

        dv12 = dV12[ind]

        maxinds = signal.argrelmax(dv12, order=10, mode='wrap')[0]
        print maxinds
        if maxinds.shape[0] > 0:
            maxinds = maxinds[np.argsort(dv12[maxinds])[::-1]]
        print maxinds
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax4 = ax.twinx()
        ax.plot(phiC, s, 'k+')
        ax2.plot(phiC, d1s, 'b+')
        ax3.plot(phiC, d2s, 'g+')
        ax4.plot(0.5*(phiR+phiC), dv12, 'r+')
        if maxinds.shape[0] >= 1:
            ax.axvline(phiC[maxinds[0]], color='k')
        if maxinds.shape[0] >= 2:
            ax.axvline(phiC[maxinds[1]], color='b')
        if maxinds.shape[0] >= 3:
            ax.axvline(phiC[maxinds[2]], color='g')
        if maxinds.shape[0] >= 4:
            ax.axvline(phiC[maxinds[3]], color='r')
        ax.set_title(r"$R = {0:f}$".format(R))
    plt.tight_layout()
    fig.savefig("shock_comp.png")
    plt.close()

    fig, ax = plt.subplots(1, 1)
    for i,R in enumerate(Rs):
        ind = r==R
        numphi = r[ind].shape[0]

        if i>0:
            rm = 0.5*(R+Rs[i-1])
        else:
            rm = R - 0.5*(Rs[1]-R)
        if i<N-1:
            rp = 0.5*(R+Rs[i+1])
        else:
            rp = R + 0.5*(R-Rs[i-1])

        X = np.zeros((2, numphi+1))
        Y = np.zeros((2, numphi+1))
        X[:,1:] = (phi[ind] + 0.5*dphi[ind])[None,:]
        X[:,0] = phi[ind][0] - 0.5*dphi[ind][0]
        Y[0,:] = rm
        Y[1,:] = rp

        dat = np.zeros(r[ind].shape)
        phiC = phi[ind]
        phiR = np.roll(phiC, -1)
        phiR[phiR<phiC] += 2*np.pi

        s = S[ind]
        sr = np.roll(s,-1)
        ds = (sr-s) / (phiR - phiC)
        
        dv12 = dV12[ind]
        maxinds = signal.argrelmax(dv12, order=10, mode='wrap')[0]
        if maxinds.shape[0] > 0:
            maxinds = maxinds[np.argsort(dv12[maxinds])[::-1]]
        if maxinds.shape[0] > 2:
            thresh = dv12[maxinds[2]]
            indsss = (dv12 > thresh)*(ds > 0)
            dat[indsss] = 1.0

        C = ax.pcolormesh(X, Y, np.atleast_2d(dat), 
                        edgecolors='none', vmin=0.0, vmax=1.0,
                        cmap=dp.viridis)
    ax.set_yscale('log')
    plt.tight_layout()
    fig.savefig("shock_det.png")
    plt.close()

    for i,R in enumerate(Rs):
        ind = r==R

        phiC = phi[ind]
        phiR = np.roll(phiC, -1)
        phiL = np.roll(phiC, 1)
        phiR[phiR<phiC] += 2*np.pi
        phiL[phiL>phiC] -= 2*np.pi

        s = S[ind]
        sl = np.roll(s,1)
        sr = np.roll(s,-1)

        d1sR = (sr - s) / (phiR -phiC)

        smax = s.max()
        smin = s.min()

        dv12 = dV12[ind]
        maxinds = signal.argrelmax(dv12, order=10, mode='wrap')[0]
        if maxinds.shape[0] > 0:
            maxinds = maxinds[np.argsort(dv12[maxinds])[::-1]]
        if maxinds.shape[0] >= 4:
            thresh = dv12[maxinds[3]]
        elif maxinds.shape[0] > 1:
            thresh = dv12[maxinds[-1]]
        else:
            thresh = dv12.mean()
        if thresh < 0.0:
            thresh = 0.0

        inds = (d1sR > 0) * (dv12 > thresh)

        iS = []

        sortInds = np.argsort(d1sR)[::-1]
        shock1 = 0
        shock2 = 0
        for j in sortInds:
            if not inds[j]:
                continue
            if shock1 == 0:
                shock1 = 1
            elif shock1 == 2 and shock2 == 0:
                used = False
                for pair in iS:
                    if pair[0] < pair[1]:
                        if j >= pair[0] and j < pair[1]:
                            used = True
                            break
                    elif pair[0] > pair[1]:
                        if (j >= pair[0] or j < pair[1]):
                            used = True
                            break
                if not used:
                    shock2 = 1
            if shock1 == 1 or shock2 == 1:
                jL = j
                jR = j+1
                if jR == s.shape[0]:
                    jR = 0
                while inds[jR]:
                    jR += 1
                    if jR == s.shape[0]:
                        jR = 0
                while inds[jL]:
                    jL -= 1
                    if jL == -1:
                        jL = s.shape[0]-1
                iS.append([jL, jR])
                if shock1 == 1:
                    shock1 = 2
                elif shock2 == 1:
                    break

        print iS, len(s[inds])
        if len(iS) == 1:
            print np.arange(s.shape[0])[inds]

        if len(iS) == 0:
            iSa[i,:] = -1
            iSb[i,:] = -1
            phiSa[i,:] = np.inf
            phiSb[i,:] = np.inf
            continue

        if i == 0:
            iSa[i,:] = iS[0]
            phiSa[i,:] = phi[ind][iS[0]]
            if len(iS) > 1:
                iSb[i,:] = iS[1]
                phiSb[i,:] = phi[ind][iS[1]]
            else:
                iSb[i,:] = -1
                phiSb[i,:] = np.inf

        else:
            sa = -1
            sb = -1
            diffa = np.inf
            diffb = np.inf
            for j,sh in enumerate(iS):
                jumpa = phi[ind][sh[1]] - phiSa[i-1,1]
                jumpb = phi[ind][sh[1]] - phiSb[i-1,1]
                if jumpa > np.pi:
                    jumpa -= 2*np.pi
                elif jumpa < -np.pi:
                    jumpa += 2*np.pi
                if jumpb > np.pi:
                    jumpb -= 2*np.pi
                elif jumpb < -np.pi:
                    jumpb += 2*np.pi
                if math.fabs(jumpa) < math.fabs(diffa):
                    sa = j
                    diffa = jumpa
                if math.fabs(jumpb) < math.fabs(diffb):
                    sb = j
                    diffb = jumpb
            if iSa[i-1,0] < 0 and iSb[i-1,0] < 0:
                iSa[i,:] = iS[0]
                iSb[i,:] = iS[1]
                phiSa[i,:] = phi[ind][iS[0]]
                phiSb[i,:] = phi[ind][iS[1]]
            elif iSb[i-1,0] < 0:
                iSa[i,:] = iS[sa]
                phiSa[i,:] = phi[ind][iS[sa]]
                if len(iS) == 1:
                    iSb[i,:] = -1
                    phiSb[i,:] = np.inf
                elif sa == 0:
                    iSb[i,:] = 1
                    phiSb[i,:] = phi[ind][iS[1]]
                else:
                    iSb[i,:] = 0
                    phiSb[i,:] = phi[ind][iS[0]]
            elif iSa[i-1,0] < 0:
                iSb[i,:] = iS[sb]
                phiSb[i,:] = phi[ind][iS[sb]]
                if len(iS) == 1:
                    iSa[i,:] = -1
                    phiSa[i,:] = np.inf
                elif sb == 0:
                    iSa[i,:] = 1
                    phiSa[i,:] = phi[ind][iS[1]]
                else:
                    iSa[i,:] = 0
                    phiSa[i,:] = phi[ind][iS[0]]
            elif sa != sb and sa > -1 and sb > -1:
                iSa[i,:] = iS[sa]
                iSb[i,:] = iS[sb]
                phiSa[i,:] = phi[ind][iS[sa]]
                phiSb[i,:] = phi[ind][iS[sb]]
            else:
                if math.fabs(diffa) < math.fabs(diffb):
                    iSa[i,:] = iS[sa]
                    iSb[i,:] = -1
                    phiSa[i,:] = phi[ind][iS[sa]]
                    phiSb[i,:] = np.inf
                else:
                    iSa[i,:] = -1
                    iSb[i,:] = iS[sb]
                    phiSa[i,:] = np.inf
                    phiSb[i,:] = phi[ind][iS[sb]]

    for i in xrange(N):
        if iSa[i,0] > -1:
            dphia = phiSa[i,1] - phiSa[i,0]
            if dphia > np.pi:
                phiSa[i,0] += 2*np.pi
            elif dphia < -np.pi:
                phiSa[i,0] -= 2*np.pi
        if iSb[i,0] > -1:
            dphib = phiSb[i,1] - phiSb[i,0]
            if dphib > np.pi:
                phiSb[i,0] += 2*np.pi
            elif dphib < -np.pi:
                phiSb[i,0] -= 2*np.pi

    return iSa, iSb, phiSa, phiSb

def dissipation_plot(t, r, phi, sig, pi, vr, vp, u0, dphi, shockDat, 
                        name, pars):

    Rs = np.unique(r)
    M = pars['GravM']
    gam = pars['Adiabatic_Index']

    if shockDat != None:
        velRS = shockDat[0]
        velSS = shockDat[1]
        vel12 = shockDat[2]
        dV12 = vel12 - velRS
    else:
        dV12 = None

    S = np.log(pi * np.power(sig, -gam)) / (gam-1.0)
    S -= S.min()

    #iSa, iSb, phiSa, phiSb = find_shocks_d2s(r, phi, S)
    iSa, iSb, phiSa, phiSb = find_shocks_d2sDet(r, phi, dphi, S, dV12)

    up = u0*vp
    N = Rs.shape[0]
    dS = np.zeros((N,3))
    psiQ = np.zeros((N,3))
    dQdm = np.zeros((N,3))
    dQdr = np.zeros((N,3))
    phiS = np.zeros((N,2))

    phiS[:,0] = 0.5*(phiSa[:,0] + phiSa[:,1])
    phiS[:,1] = 0.5*(phiSb[:,0] + phiSb[:,1])

    fig, ax = plt.subplots(1,1,figsize=(12,9))
    ax.plot(Rs*np.cos(phiS[:,0]), Rs*np.sin(phiS[:,0]), 'k+')
    ax.plot(Rs*np.cos(phiSa[:,0]), Rs*np.sin(phiSa[:,0]), 'r+')
    ax.plot(Rs*np.cos(phiSa[:,1]), Rs*np.sin(phiSa[:,1]), 'b+')
    ax.plot(Rs*np.cos(phiS[:,1]), Rs*np.sin(phiS[:,1]), 'k+')
    ax.plot(Rs*np.cos(phiSb[:,0]), Rs*np.sin(phiSb[:,0]), 'r+')
    ax.plot(Rs*np.cos(phiSb[:,1]), Rs*np.sin(phiSb[:,1]), 'b+')
    figname = "plot_minidisc_shockPlot_{0}.png".format(name)
    print("Saving {0:s}...".format(figname))
    fig.savefig(figname)
    plt.close(fig)

    temp = pi/sig
    ka_bbes = 0.4 # cm^2/g
    cool_cgs = 8*eos.sb * (temp*eos.mp*eos.c*eos.c)**4 / (
                3*ka_bbes*sig * eos.rho_scale*eos.rg_solar)
    cool = cool_cgs / (eos.c*eos.c*eos.c*eos.rho_scale)
    dQcool = np.zeros(Rs.shape)

    for i,R in enumerate(Rs):

        ind = (r==R)
        s = S[ind]

        #print iSa[i], iSb[i]
        
        if iSa[i,0] > -1:
            dsa = s[iSa[i,1]] - s[iSa[i,0]]
            siga = sig[ind][iSa[i,0]]
            Ta = pi[ind][iSa[i,0]] / siga
            upa = up[ind][iSa[i,0]]
        else:
            dsa = 0.0
            siga = 0.0
            Ta = 0.0
            upa = 0.0
        if iSb[i,0] > -1:
            dsb = s[iSb[i,1]] - s[iSb[i,0]]
            sigb = sig[ind][iSb[i,0]]
            Tb = pi[ind][iSb[i,0]] / sigb
            upb = up[ind][iSb[i,0]]
        else:
            dsb = 0.0
            sigb = 0.0
            Tb = 0.0
            upb = 0.0

        dS[i,0] = dsa
        dS[i,1] = dsb
        dS[i,2] = dsa+dsb
        psiQ[i,0] = (math.exp((gam-1)*dsa)-1) / (gam-1.0)
        psiQ[i,1] = (math.exp((gam-1)*dsb)-1) / (gam-1.0)
        psiQ[i,2] = psiQ[i,0] + psiQ[i,1]
        dQdm[i,0] = Ta * psiQ[i,0]
        dQdm[i,1] = Tb * psiQ[i,1]
        dQdm[i,2] = dQdm[i,0] + dQdm[i,1]
        dQdr[i,0] = R * siga * upa * dQdm[i,0]
        dQdr[i,1] = R * sigb * upb * dQdm[i,1]
        dQdr[i,2] = dQdr[i,0] + dQdr[i,1]
        dQcool[i] = (cool[ind] * R * dphi[ind]).sum()

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

    figname = "plot_minidisc_psi_{0}.png".format(name)
    print("Saving {0:s}...".format(figname))
    fig.savefig(figname)
    plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(12,9))
    ax[0].plot(Rs, dS[:,0], ls='', marker='+', ms=10, mew=2, color=orange)
    ax[0].plot(Rs, dS[:,1], ls='', marker='+', ms=10, mew=2, color=green)
    ax[0].plot(Rs, dS[:,2], ls='', marker='+', ms=10, mew=2, color=blue)
    ax[0].set_ylabel(r'$\mu \Delta s$')
    ax[0].set_xscale('log')
    ax[0].set_xlim(Rs.min(), Rs.max())
    if (dS > 0).any():
        ax[0].set_yscale('log')
    ax[1].plot(Rs, psiQ[:,0], ls='', marker='+', ms=10, mew=2, color=orange)
    ax[1].plot(Rs, psiQ[:,1], ls='', marker='+', ms=10, mew=2, color=green)
    ax[1].plot(Rs, psiQ[:,2], ls='', marker='+', ms=10, mew=2, color=blue)
    ax[1].set_ylabel(r'$\psi_Q$')
    ax[1].set_xscale('log')
    ax[1].set_xlim(Rs.min(), Rs.max())
    if (psiQ > 0).any():
        ax[1].set_yscale('log')
    ax[2].plot(Rs, dQdr[:,0], ls='', marker='+', ms=10, mew=2, color=orange)
    ax[2].plot(Rs, dQdr[:,1], ls='', marker='+', ms=10, mew=2, color=green)
    ax[2].plot(Rs, dQdr[:,2], ls='', marker='+', ms=10, mew=2, color=blue)
    ax[2].plot(Rs, dQcool, ls='', marker='+', ms=10, mew=2, color=red)
    ax[2].set_ylabel(r'$\dot{Q}$')
    ax[2].set_xscale('log')
    ax[2].set_xlim(Rs.min(), Rs.max())
    if (dQdr > 0).any() or (dQcool > 0).any():
        ax[2].set_yscale('log')
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    ax[2].set_xlabel(r'$r$ ($M$)')

    figname = "plot_minidisc_psiNice_{0}.pdf".format(name)
    print("Saving {0:s}...".format(figname))
    fig.savefig(figname)
    plt.close(fig)

    Is = np.array([0,N/4,N/2,3*N/4,N-1])

    fig1, ax1 = plt.subplots(8, Is.shape[0], figsize=(50,40))

    for i,I in enumerate(Is):
        R = Rs[I]
        ind = r==R
        s = S[ind]
        sl = np.roll(s,1)
        sll = np.roll(s,2)
        sr = np.roll(s,-1)
        srr = np.roll(s,-2)
        d2s = sr - 2*s + sl
        d2s = -srr + 16*sr - 30*s + 16*sl - sll

        phir = np.roll(phi[ind],-1)

        NP = s.shape[0]

        ax2 = ax1[0,i].twinx()
        if shockDetPlot:
            ax3 = ax1[0,i].twinx()

        for j in xrange(ax1.shape[0]):
            if iSa[I,0] > -1:
                ax1[j,i].axvline(phiSa[I,0], color='b')
                ax1[j,i].axvline(phiSa[I,1], color='g')
            if iSb[I,0] > -1:
                ax1[j,i].axvline(phiSb[I,0], color='b')
                ax1[j,i].axvline(phiSb[I,1], color='g')
        if shockDetPlot:
            ax1[0,i].plot(phi[ind], vel12[ind]-velRS[ind], 'b+')
            ax2.plot(phi[ind], d2s, 'r+')
            ax3.plot(phi[ind], s, 'k+')
        else:
            ax2.plot(phi[ind], d2s, 'r+')
            ax1[0,i].plot(phi[ind], s, 'k+')

        ax1[1,i].plot(phi[ind] + 0.5*dphi[ind], (sr-s)/(phir-phi[ind]), 'k+')
        ax1[2,i].plot(phi[ind], sig[ind]*s, 'k+')
        ax1[3,i].plot(phi[ind], sig[ind]*u0[ind]*s, 'k+')
        ax1[4,i].plot(phi[ind], sig[ind]*up[ind]*s, 'k+')
        ax1[5,i].plot(phi[ind], sig[ind], 'k+')
        ax1[6,i].plot(phi[ind], pi[ind], 'k+')
        ax1[7,i].plot(phi[ind], pi[ind]/sig[ind], 'k+')
        ax1[7,i].set_xlabel(r"$\phi$")

    ax1[0,0].set_ylabel(r"$s$")
    ax1[1,0].set_ylabel(r"$\partial s / \partial \phi$")
    ax1[2,0].set_ylabel(r"$\Sigma_0 s$")
    ax1[3,0].set_ylabel(r"$\Sigma_0 u^0 s$")
    ax1[4,0].set_ylabel(r"$\Sigma_0 u^\phi s$")
    ax1[5,0].set_ylabel(r"$\Sigma_0$")
    ax1[6,0].set_ylabel(r"$\Pi$")
    ax1[7,0].set_ylabel(r"$\Pi / \Sigma$")

    figname = "plot_minidisc_diss_{0}.png".format(name)
    print("Saving {0:s}...".format(figname))
    fig1.savefig(figname)
    plt.close(fig1)

    dSdr, dSdp = calcGrad(r, phi, phi+dphi, S, pars, slopelimit=True)

    dS = vr*dSdr+vp*dSdp
    fig, ax = plt.subplots(1,3, figsize=(16,9))

    Nr = Rs.shape[0]
    for i,R in enumerate(Rs):
        ind = r==R
        N = r[ind].shape[0]

        if i>0:
            rm = 0.5*(R+Rs[i-1])
        else:
            rm = R - 0.5*(Rs[1]-R)
        if i<Nr-1:
            rp = 0.5*(R+Rs[i+1])
        else:
            rp = R + 0.5*(R-Rs[i-1])

        X = np.zeros((2, N+1))
        Y = np.zeros((2, N+1))
        X[:,1:] = (phi[ind] + 0.5*dphi[ind])[None,:]
        X[:,0] = phi[ind][0] - 0.5*dphi[ind][0]
        Y[0,:] = rm
        Y[1,:] = rp

        C0 = ax[0].pcolormesh(X, Y, np.atleast_2d(dSdr[ind]), 
                        edgecolors='none', vmin=dSdr.min(), vmax=dSdr.max(),
                        cmap=dp.viridis)
        C1 = ax[1].pcolormesh(X, Y, np.atleast_2d(dSdp[ind]), 
                        edgecolors='none', vmin=dSdp.min(), vmax=dSdp.max(),
                        cmap=dp.viridis)
        C2 = ax[2].pcolormesh(X, Y, np.atleast_2d(dS[ind]-dS[ind].min()), 
                        edgecolors='none',
                        cmap=dp.viridis)
    fig.colorbar(C0, ax=ax[0])
    fig.colorbar(C1, ax=ax[1])
    fig.colorbar(C2, ax=ax[2])
    figname = "plot_minidisc_ds_{0}.png".format(name)
    print("Saving {0:s}...".format(figname))
    fig.savefig(figname)
    plt.close(fig)

    return phiS, phiSa, phiSb, psiQ, dQdm, dQdr

def angular_momentum_flux_plot(r, sig, pi, u0, vr, vp, phi, dphi, 
                                shockDissDat, pars, name):

    gam = pars['Adiabatic_Index']
    a = pars['BinA']
    M1 = pars['BinM']

    g00, g0r, g0p, grr, grp, gpp = gr.calc_g(r, pars)
    ur = u0*vr
    up = u0*vp

    u0d = g00*u0 + g0r*ur + g0p*up
    urd = g0r*u0 + grr*ur + grp*up
    upd = g0p*u0 + grp*ur + gpp*up

    h = 1 + gam*pi/((gam-1)*sig)

    d = r*sig*u0
    j = r*sig*u0*h*upd
    l = h*upd
    mdot = r*sig*ur
    f = r*sig*ur*h*upd

    temp = pi/sig
    ka_bbes = 0.4 # cm^2/g
    cool_cgs = 8*eos.sb * (temp*eos.mp*eos.c*eos.c)**4 / (
                3*ka_bbes*sig * eos.rho_scale*eos.rg_solar)
    cool = cool_cgs / (eos.c*eos.c*eos.c*eos.rho_scale)
    qdotc = -r*cool*upd

    x = r*np.cos(phi)
    y = r*np.sin(phi)
    X = -a
    Y = 0.0
    A = math.sqrt(X*X+Y*Y)
    r2 = np.sqrt((x-X)*(x-X)+(y-Y)*(y-Y))
    fx = M1*( -(x-X)/(r2*r2*r2) - X/(A*A*A) ) * sig*h*u0*u0
    fy = M1*( -(y-Y)/(r2*r2*r2) - Y/(A*A*A) ) * sig*h*u0*u0
    t = r*(-y*fx + x*fy)

    Rs = np.unique(r)
    L = np.zeros(Rs.shape)
    J = np.zeros(Rs.shape)
    Mdot = np.zeros(Rs.shape)
    D = np.zeros(Rs.shape)
    F = np.zeros(Rs.shape)
    Qdotc = np.zeros(Rs.shape)
    Cool = np.zeros(Rs.shape)
    T = np.zeros(Rs.shape)
    Om = np.zeros(Rs.shape)

    for i,R in enumerate(Rs):
        ind = (r==R)
        L[i] = (l[ind]*dphi[ind]).sum() / (2*math.pi)
        Om[i] = (up[ind]*dphi[ind]).sum() / (2*math.pi)
        J[i] = (j[ind]*dphi[ind]).sum()
        F[i] = (f[ind]*dphi[ind]).sum()
        D[i] = (d[ind]*dphi[ind]).sum()
        Mdot[i] = (mdot[ind]*dphi[ind]).sum()
        Qdotc[i] = (qdotc[ind]*dphi[ind]).sum()
        T[i] = (t[ind]*dphi[ind]).sum()
        Cool[i] = (R*cool[ind]*dphi[ind]).sum()

    fig, ax = plt.subplots(2,3, figsize=(12,9))
    ax[0,0].plot(Rs, L, 'k+')
    ax[0,0].set_ylabel(r'$\langle h u_\phi \rangle$')
    ax[0,0].set_xscale('log')
    ax[0,1].plot(Rs, J, 'k+')
    ax[0,1].set_ylabel(r'$[ r \rho u^t h u_\phi ]$')
    ax[0,1].set_xscale('log')
    ax[0,2].plot(Rs, F, 'k+')
    ax[0,2].set_ylabel(r'$[ r \rho u^r h u_\phi ]$')
    ax[0,2].set_xscale('log')
    ax[1,0].plot(Rs, Mdot, 'k+')
    ax[1,0].set_ylabel(r'$[ r \rho u^r ]$')
    ax[1,0].set_xscale('log')
    ax[1,1].plot(Rs, Qdotc, 'k+')
    ax[1,1].set_ylabel(r'$[ r \dot{Q} u_\phi ]$')
    ax[1,1].set_xscale('log')
    ax[1,2].plot(Rs, T, 'k+')
    ax[1,2].set_ylabel(r'$[ r f_\phi ]$')
    ax[1,2].set_xscale('log')
    ax[1,0].set_xlabel(r'$R$')
    ax[1,1].set_xlabel(r'$R$')
    ax[1,2].set_xlabel(r'$R$')

    plt.tight_layout()

    figname = "plot_minidisc_fluxRaw_{0}.png".format(name)
    print("Saving {0:s}...".format(figname))
    fig.savefig(figname)
    plt.close(fig)

    if shockDissDat is not None:
        dQdr = shockDissDat[5][:,2]

    fig, ax = plt.subplots(2,2, figsize=(12,9))
    ax[0,0].plot(Rs, J, 'b+')
    ax[0,0].plot(Rs, D*L, 'r+')
    ax[0,0].plot(Rs, J - D*L, 'k+')
    ax[0,1].plot(Rs, F, 'b+')
    ax[0,1].plot(Rs, Mdot*L, 'r+')
    ax[0,1].plot(Rs, F - Mdot*L, 'k+')
    
    AMmdot = np.zeros(Rs.shape)
    AMdiss = np.zeros(Rs.shape)
    AMcool = np.zeros(Rs.shape)
    AMtorq = np.zeros(Rs.shape)

    dLdR = (L[2:]-L[:-2]) / (Rs[2:]-Rs[:-2])
    AMmdot[1:-1] = Mdot[1:-1] * dLdR
    AMdiss[1:-1] = ((F-Mdot*L)[2:] - (F-Mdot*L)[:-2]) / (Rs[2:]-Rs[:-2])
    AMcool[:] = Qdotc[:]
    AMtorq[:] = T[:]
    if shockDissDat is not None:
        AMdqdr = np.zeros(Rs.shape)
        dOm = (Om[2:] - Om[:-2]) / (Rs[2:] - Rs[:-2])
        #dOm = -1.5 * Om/Rs
        tau = dQdr[1:-1] / dOm
        tauS = signal.savgol_filter(tau, 9, 1)
        AMdqdr = -(tauS[2:]-tauS[:-2]) / (Rs[3:-1] - Rs[1:-3])

    ax[1,0].plot(Rs, AMmdot, 'b+')
    ax[1,0].plot(Rs, AMdiss, 'r+')
    ax[1,0].plot(Rs, AMcool, 'bx')
    ax[1,0].plot(Rs, AMtorq, 'rx')
    ax[1,0].plot(Rs, -AMdiss+AMcool+AMtorq, 'g+')
    ax[1,0].plot(Rs, AMmdot+AMdiss-AMcool-AMtorq, 'k+')

    ax[1,1].plot(Rs,-AMmdot, 'b+')
    ax[1,1].plot(Rs, AMdiss-AMcool-AMtorq, 'g+')
    if shockDissDat is not None:
        ax[1,1].plot(Rs[2:-2], AMdqdr, 'r+')
        #ax[1,1].plot(Rs[1:-1], tau, 'k+')
        #ax[1,1].plot(Rs[1:-1], tauS, 'b+')
        #ax[1,1].plot(Rs[1:-1], tauS2, 'g+')
        #ax[1,1].plot(Rs[1:-1], tauS3, 'y+')
        #ax[1,1].plot(Rs[1:-1], tauS4, 'r+')
        #ax[1,1].plot(Rs, Rs * dQdr / Om, 'rx')
    #ax[1,1].plot(Rs, Riph * dQdr / Om + AMmdot, 'k+')
   
    """
    Riph = 0.5*(Rs[:-1]+Rs[1:])
    AMmdot = np.zeros(Riph.shape)
    AMdiss = np.zeros(Riph.shape)
    AMcool = np.zeros(Riph.shape)
    AMtorq = np.zeros(Riph.shape)

    dLdR = (L[2:]-L[:-2]) / (Rs[2:]-Rs[:-2])
    AMmdot[1:-1] = 0.5*(Mdot[2:-1]*dLdR[1:] + Mdot[1:-2]*dLdR[:-1]) * (
                        Rs[2:-1]-Rs[1:-2])
    AMdiss = (F - Mdot*L)[1:] - (F-Mdot*L)[:-1]
    AMcool = 0.5*(Qdotc[1:]+Qdotc[:-1])*(Rs[1:]-Rs[:-1])
    AMtorq = 0.5*(T[1:]+T[:-1])*(Rs[1:]-Rs[:-1])

    if shockDissDat is not None:
        AMdqdr = np.zeros(Riph.shape)
        dOm = (Om[2:] - Om[:-2]) / (Rs[2:] - Rs[:-2])
        dOm = -1.5 * Om/Rs
        tau = Rs*dQdr / dOm
        AMdqdr = -(tau[1:]-tau[:-1])
    
    ax[1,0].plot(Riph, AMmdot, 'b+')
    ax[1,0].plot(Riph, AMdiss, 'r+')
    ax[1,0].plot(Riph, AMcool, 'bx')
    ax[1,0].plot(Riph, AMtorq, 'rx')
    ax[1,0].plot(Riph, -AMdiss+AMcool+AMtorq, 'g+')
    ax[1,0].plot(Riph, AMmdot+AMdiss-AMcool-AMtorq, 'k+')

    ax[1,1].plot(Riph,-AMmdot, 'b+')
    ax[1,1].plot(Riph, AMdiss-AMcool-AMtorq, 'g+')
    if shockDissDat is not None:
        #ax[1,1].plot(Riph, AMdqdr, 'r+')
        ax[1,1].plot(Rs, 2.0/3.0 * dQdr / Om, 'rx')
    #ax[1,1].plot(Rs, Riph * dQdr / Om + AMmdot, 'k+')
    """


    ax[0,0].set_xscale('log')
    ax[0,1].set_xscale('log')
    ax[1,0].set_xscale('log')
    ax[1,1].set_xscale('log')
    ax[0,0].set_xlim(r.min(), r.max())
    ax[0,1].set_xlim(r.min(), r.max())
    ax[1,0].set_xlim(r.min(), r.max())
    ax[1,1].set_xlim(r.min(), r.max())

    figname = "plot_minidisc_fluxComp_{0}.png".format(name)
    print("Saving {0:s}...".format(figname))
    fig.savefig(figname)
    plt.close(fig)

    fig, ax = plt.subplots(1,1, figsize=(12,9))

    ax.plot(Rs,-AMmdot, marker='+', ms=10, mew=2, color='k', 
                label=r"$\dot{M} \partial_r \langle h u_\phi \rangle$")
    ax.plot(Rs, AMdiss, marker='+', ms=10, mew=2, color=blue,
                label=r"$\partial_r \delta \langle T^r_\phi \rangle$")
    ax.plot(Rs, AMdiss-AMcool-AMtorq, marker='+', ms=10, mew=2, color=green,
                label=r"$\partial_r \delta \langle T^r_\phi \rangle - \langle f_\phi \rangle$")
    """
    ax.plot(Riph,-AMmdot, marker='+', ms=10, mew=2, color='k', 
                label=r"$\dot{M} \partial_r \langle h u_\phi \rangle$")
    ax.plot(Riph, AMdiss-AMcool-AMtorq, marker='+', ms=10, mew=2, color=blue,
                label=r"$\partial_r \delta \langle T^r_\phi \rangle$")
    """
    ylim = ax.get_ylim()
    if shockDissDat is not None:
        ax.plot(Rs, dQdr / Om, marker='+', ms=10, mew=2, ls='',
                    color=red, label=r"shock $\left(\partial F_J / \partial r\right)_{global}$")
        ax.plot(Rs[2:-2], AMdqdr, marker='+', ms=10, mew=2, ls='',
                    color=orange, label=r"shock $\left (\partial F_J / \partial r\right)_{local}$")
    ax.set_xlim(r.min(), r.max())
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$r (M)$")
    ax.set_ylabel("Angular Momentum Flux")
    ax.set_xscale("log")
    plt.legend()

    figname = "plot_minidisc_fluxCompNice_{0}.pdf".format(name)
    print("Saving {0:s}...".format(figname))
    fig.savefig(figname)
    plt.close(fig)

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

    if shockDetPlot:
        shockDetDat = shockPlot(r, phi, dphi, sig, pi, u0, vr, vp, chcknum, 
                                pars, slopelimit=True, TH=0.0)
    else:
        shockDetDat = None
    if dissPlot:
        shockDissDat = dissipation_plot(t, r, phi, sig, pi, vr, vp, u0, 
                                            dphi, shockDetDat, chcknum, pars)
        phiS = shockDissDat[0]
        phiSa = shockDissDat[1]
        phiSb = shockDissDat[2]
        psiQ = shockDissDat[3]
        dQdm = shockDissDat[4]
        dQdr = shockDissDat[5]
    else:
        shockDissDat = None
        phiS = None
        phiSa = None
        phiSb = None
        psiQ = None
        dQdm = None
        dQdr = None

    angular_momentum_flux_plot(r, sig, pi, u0, vr, vp, phi, dphi, shockDissDat,
                                pars, chcknum)

    if not allTheOtherPlots:
        return

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
        if maxinds.shape[0] > 0:
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

    dpdra = 0.5*(phiSa[2:,0]+phiSa[2:,1] - phiSa[:-2,0] - phiSa[:-2,1]) / (
                    Rs[2:] - Rs[:-2])
    dpdrb = 0.5*(phiSb[2:,0]+phiSb[2:,1] - phiSb[:-2,0] - phiSb[:-2,1]) / (
                    Rs[2:] - Rs[:-2])
    tpa = -1.0 / (dpdra * Rs[1:-1])
    tpb = -1.0 / (dpdrb * Rs[1:-1])

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

    omk = np.sqrt(M/(Rs*Rs*Rs))
    tpWKB = 1.0 / (avMach * np.sqrt((1-bw/omk)*(1-bw/omk)-0.25))
    tpWKB1 = 1.0 / (avMach * np.sqrt((1-bw/omk)*(1-bw/omk)-0.6))
    tpWKB2 = 1.0 / (avMach * np.sqrt((1-bw/omk)*(1-bw/omk)-0.7))
    tpWKBrel = 1.0 / avMachNorb * np.sqrt(
            (1-2*M/Rs)*(1-3*M/Rs)
        / ((1-bw/omk)*(1-bw/omk)-0.25*(1-6*M/Rs)/(1-2*M/Rs)))
    tpWKBrel1 = 1.0 / avMachNorb * np.sqrt(
            (1-2*M/Rs)*(1-3*M/Rs)
        / ((1-bw/omk)*(1-bw/omk)-0.6*(1-6*M/Rs)/(1-2*M/Rs)))
    tpWKBrel2 = 1.0 / avMachNorb * np.sqrt(
            (1-2*M/Rs)*(1-3*M/Rs)
        / ((1-bw/omk)*(1-bw/omk)-0.7*(1-6*M/Rs)/(1-2*M/Rs)))


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
        plot_line(ax[1,2], avMach, tpWKB, color='r')
        #plot_line(ax[1,2], avMach[1:-1], tpWKB1, color='r', ls='--')
        #plot_line(ax[1,2], avMach[1:-1], tpWKB2, color='r', ls=':')
        plot_line(ax[1,2], avMach, tpWKBrel, color='g')
        #plot_line(ax[1,2], avMach[1:-1], tpWKBrel1, color='g', ls='--')
        plot_line(ax[1,2], avMach, tpWKBrel2, color='g', ls='--')

        pretty_axis(ax[1,2], pars, xlabel=r"$\mathcal{M}$",
                        ylabel=r"$\tan \theta_S$", yscale='log', xscale='log')

        figNice, axNice = plt.subplots(1,1,figsize=(12,9))
        axNice.plot(avMachNorb, tpWKBrel, '-', lw=4, color="grey", 
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
        outname = "plot_minidisc_tanq_mach_{0}.pdf".format(
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
    axNice.set_xlim(RMIN, RMAX)
    axNice.tick_params(labelsize=18)
    axNice.set_title(r"$\alpha$ Parameter", fontsize=36)
    outname = "plot_minidisc_alpha_r_{0}.pdf".format(
                "_".join(chckname.split(".")[0].split("_")[1:]))
    print("Saving {0:s}...".format(outname))
    figNice.savefig(outname)
    plt.close(figNice)

    figNice, axNice = plt.subplots(1,1,figsize=(12,9))
    axNice.plot(avMachNorb, tpWKBrel, '-', lw=4, color="grey", 
                    label="WKB")
    axNice.plot(avMachNorb[1:-1][tpa>0], tpa[tpa>0], 
                    '+', color=blue, ms=10, mew=2, label="Shock 1")
    axNice.plot(avMachNorb[1:-1][tpb>0], tpb[tpb>0], 
                    '+', color=orange, ms=10, mew=2, label="Shock 2")
    axNice.set_xlabel(r"$\mathcal{M}_N = \langle r v^\phi / c_s \rangle$",
                        fontsize=24)
    axNice.set_ylabel(r"$\tan \theta_S$", fontsize=24)
    #axNice.set_xscale('log')
    #axNice.set_yscale('log')
    axNice.tick_params(labelsize=18)
    plt.legend(loc="upper right", fontsize=24)
    axNice.set_title("Dispersion Relation", fontsize=36)
    outname = "plot_minidisc_tanq_mach_ds_{0}.pdf".format(
                "_".join(chckname.split(".")[0].split("_")[1:]))
    print("Saving {0:s}...".format(outname))
    figNice.savefig(outname)
    plt.close(figNice)

    outname = "plot_minidisc_orbit_{0}.png".format(
                "_".join(chckname.split(".")[0].split("_")[1:]))
    print("Saving {0:s}...".format(outname))
    fig.savefig(outname)
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(12,9))

    ax[0,0].plot(Rs, phiS[:,0], marker='+', color=blue, ls='')
    ax[0,0].plot(Rs, phiS[:,1], marker='+', color=orange, ls='')
    #ax[0,0].plot(Rs, phiSa[:,0], marker='+', color='k', ls='')
    #ax[0,0].plot(Rs, phiSa[:,1], marker='+', color='g', ls='')
    #ax[0,0].plot(Rs, phiSb[:,0], marker='+', color='r', ls='')
    #ax[0,0].plot(Rs, phiSb[:,1], marker='+', color='y', ls='')
    ax[0,0].set_xlabel(r"$r$")
    ax[0,0].set_ylabel(r"$\phi_S$")
    ax[0,1].plot(avMach, psiQ[:,2], marker='+', color='k', ls='')
    ax[0,1].set_xlabel(r"$\langle \mathcal{M} \rangle$")
    ax[0,1].set_ylabel(r"$\psi_Q$")
    ax[0,1].set_xscale("log")
    if (psiQ[:,2]>0).any():
        ax[0,1].set_yscale("log")
    ax[1,0].plot(Rs, psiQ[:,2], marker='+', color='k', ls='')
    ax[1,0].set_xlabel(r"$r$")
    ax[1,0].set_ylabel(r"$\psi_Q$")
    ax[1,0].set_xscale("log")
    if (psiQ[:,2]>0).any():
        ax[1,0].set_yscale("log")
    ax[1,1].plot(tanPitch0, psiQ[1:-1,0], marker='+', color=blue, ls='')
    ax[1,1].plot(tanPitch1, psiQ[1:-1,1], marker='+', color=orange, ls='')
    ax[1,1].set_xlabel(r"$\tan \theta_S$")
    ax[1,1].set_ylabel(r"$\psi_Q$")
    if (psiQ[:,0]>0).any() or (psiQ[:,1]>0).any():
        ax[1,1].set_yscale("log")

    outname = "plot_minidisc_shocks_{0}.png".format(chcknum)
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
    plt.close(fig)


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("\nGive me a parfile and checkpoint (.h5) file(s).\n")
        sys.exit()

#    elif len(sys.argv) == 3:
#        parname = sys.argv[1]
#        filename = sys.argv[2]
#        pars = dp.readParfile(parname)
#        fig = plot_r_profile(filename, pars, sca=scale)
#        plt.show()
#
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
            plot_r_profile(filename, pars, sca=scale, plot=True,
                                        bounds=all_bounds)

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
