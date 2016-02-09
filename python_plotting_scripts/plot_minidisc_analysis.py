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
RMAX = 60

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

def plot_r_profile(filename, pars, sca='linear', plot=True, bounds=None):

    print("Reading {0:s}".format(filename))
    
    M = pars['GravM']
    a = pars['GravA']
    gam = pars['Adiabatic_Index']
    A = a*M

    t, r, phi, rho, sig, T, P, pi, H, vr, vp, u0, q, dphi = allTheThings(
                                                                filename, pars)
    inds = r < RMAX
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

    u0d = g00*u0 + g0r*ur + g0p*up
    urd = g0r*u0 + grr*ur + grp*up
    upd = g0p*u0 + grp*ur + gpp*up

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

    gam = pars['Adiabatic_Index']
    sigh = sig + gam/(gam-1.0)*pi

    nmodes = 16
    nm = np.arange(1, nmodes+1)
    A0 = np.zeros(Rs.shape[0])
    An = np.zeros((Rs.shape[0], nmodes))
    PhiN = np.zeros((Rs.shape[0], nmodes))

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

        avupd = (upd[inds] * R*dphi[inds]).sum() / A
        avsigEflux = ((sig[inds]+pi[inds]/(gam-1.0))*ur[inds] * R*dphi[inds]
                        ).sum() / A
        avsigflux = -(sig[inds]*ur[inds] * R*dphi[inds]).sum() / A

        avsig[i] = sigtot / A
        avD[i] = Dtot / A
        avL[i] = Ltot / A
        avE[i] = Etot / A
        avvr[i] = Dflux / Dtot
        avvp[i] = Dfluxp / Dtot

        Mdot[i] = -Dflux
        Ldot[i] = -Lflux
        Edot[i] = -Eflux
        Lindot[i] = avsigEflux * avupd / avsigflux

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

    outpath = filename.split("/")[:-1]
    chckname = filename.split("/")[-1]
    outname = "plot_minidisc_phases_{0}.png".format(
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
