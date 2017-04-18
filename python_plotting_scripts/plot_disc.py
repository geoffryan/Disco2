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

def allTheThings(filename, pars):

    dat = dp.readCheckpoint(filename)
    t = dat[0]
    r = dat[1]
    phi = dat[2]
    vr = dat[6]
    vp = dat[7]
    q = dat[10]
    piph = dat[11]
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

    return t, r, phi, rho, sig, T, P, pi, H, vr, vp, u0, q, piph


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
                xlabel=None, ylabel=None, xlim=None, ylim=None, twin=False):

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
    A = a*M

    t, r, phi, rho, sig, T, P, pi, H, vr, vp, u0, q, piph = allTheThings(filename, pars)
    
    inds = np.argsort(r)
    r = r[inds]
    rho = rho[inds]
    sig = sig[inds]
    T = T[inds]
    P = P[inds]
    pi = pi[inds]
    H = H[inds]
    vr = vr[inds]
    vp = vp[inds]
    u0 = u0[inds]

    R = np.logspace(np.log10(r.min()), np.log10(r.max()), 100)

    W = u0 * gr.lapse(r, pars)
    U = np.sqrt(W*W-1)

    Mdot = -2*math.pi*r*rho*u0*vr*H * (eos.c * eos.rg_solar**2 * eos.year
                                        / eos.M_solar)

    cs2 = eos.cs2(rho, T, pars)
    cs = np.sqrt(cs2)
    Ucs = np.sqrt(cs2 / (1-cs2))

    mach = U / Ucs

    if bounds is None:
        bounds = []
        bounds.append([rho[rho==rho].min(), rho[rho==rho].max()])
        bounds.append([T[T==T].min(), T[T==T].max()])
        bounds.append([P.min(), P.max()])
        bounds.append([vr.min(), vr.max()])
        bounds.append([vp[vp>0].min(), vp[vp>0].max()])
        bounds.append([W.min(), W.max()])
        bounds.append([H.min(), H.max()])
        bounds.append([Mdot.min(), Mdot.max()])
        bounds.append([cs.min(),cs.max()])
        bounds = np.array(bounds)

    if plot:

        print("Plotting t = {0:g}".format(t))

        #Plot.
        fig, ax = plt.subplots(3,3,figsize=(12,9))

        # Density
        ax2 = ax[0,0].twinx()
        plot_data(ax[0,0], r, rho)
        plot_data(ax2, r, sig * eos.rg_solar, color='b')
        pretty_axis(ax[0,0], pars, xscale=scale, yscale=scale, 
                    ylabel=r"$\rho_0$ ($g/cm^3$)")
        pretty_axis(ax2, pars, xscale=scale, yscale=scale, 
                    ylabel=r"$\Sigma_0$ ($g/cm^2$)", twin=True)

        # Temperature
        plot_data(ax[0,1], r, T)
        pretty_axis(ax[0,1], pars, xscale=scale, yscale=scale, 
                    ylabel=r"$T$ ($m_p c^2$)")

        # Pressure
        ax2 = ax[0,2].twinx()
        if pars['Background'] == 3 and pars['EOSType'] == 1:
            plot_data(ax[0,2], r, eos.P_gas(rho,T,pars), color='g')
            plot_data(ax[0,2], r, eos.P_rad(rho,T,pars), color='r')
        elif pars['Background'] == 3 and pars['EOSType'] == 2:
            plot_data(ax[0,2], r, eos.P_gas(rho,T,pars), color='g')
            plot_data(ax[0,2], r, eos.P_rad(rho,T,pars), color='r')
            plot_data(ax[0,2], r, eos.P_deg(rho,T,pars), color='y')
        plot_data(ax[0,2], r, P)
        plot_data(ax2, r, pi * eos.rg_solar, color='b')
        pretty_axis(ax[0,2], pars, xscale=scale, yscale=scale, 
                    ylabel=r"$P$ ($g\ c^2 / cm^3$)")
        pretty_axis(ax2, pars, xscale=scale, yscale=scale, 
                    ylabel=r"$\Pi$ ($g\ c^2 / cm^2$)", twin=True)

        # Radial Velocity
        plot_data(ax[1,0], r, vr)
        pretty_axis(ax[1,0], pars, xscale=scale, yscale="linear", 
                    ylabel=r"$v^r$")

        # Azimuthal Velocity
        plot_data(ax[1,1], r, vp)
        pretty_axis(ax[1,1], pars, xscale=scale, yscale=scale, 
                    ylabel=r"$v^\phi$")

        # Lorentz Factor and Mach Number
        ax2 = ax[1,2].twinx()
        plot_data(ax[1,2], r, mach)
        plot_data(ax2, r, W, color='b')
        pretty_axis(ax[1,2], pars, xscale=scale, yscale=scale, 
                    ylabel=r"$\mathcal{M}$")
        pretty_axis(ax2, pars, xscale=scale, yscale="linear", 
                    ylabel=r"$W$", twin=True)

        # Scale Height
        plot_data(ax[2,0], r, H)
        pretty_axis(ax[2,0], pars, xscale=scale, yscale=scale, 
                    xlabel=r"$r$ ($G M_\odot / c^2)$", 
                    ylabel=r"$H$ ($G M_\odot / c^2$)")

        # Accretion rate
        plot_data(ax[2,1], r, Mdot)
        pretty_axis(ax[2,1], pars, xscale=scale, yscale="linear", 
                    xlabel=r"$r$ ($G M_\odot / c^2)$", 
                    ylabel=r"$\dot{M}$ ($M_\odot/y$)")

        # Sound Speed
        plot_data(ax[2,2], r, cs)
        pretty_axis(ax[2,2], pars, xscale=scale, yscale=scale, 
                    xlabel=r"$r$ ($G M_\odot / c^2)$", ylabel=r"$c_s$")

        fig.suptitle(r"DISCO $M = {0:.1g}M_\odot$ $a_* = {1:.2f}$".format(M,a))

        #plt.tight_layout()

        outpath = filename.split("/")[:-1]
        chckname = filename.split("/")[-1]
        outname = "plot_disc_{0}.png".format("_".join(chckname.split(".")[0].split("_")[1:]))

        print("Saving {0:s}...".format(outname))
        plt.savefig(outname)

    else:
        fig = None

    return fig, bounds

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
