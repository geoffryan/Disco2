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
    vr = dat[6]
    vp = dat[7]
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

    return t, r, rho, sig, T, P, pi, H, vr, vp, u0


def plot_r_profile_single(r, f, pars, sca, ylabel, bounds=None, 
                            R=None, F=None):

    M = pars['GravM']
    a = pars['GravA']

    plt.plot(r, f, 'k+')
    if R != None and F != None:
        plt.plot(R, F, 'r')

    plt.gca().set_xscale(sca)
    if (f == 0.0).all():
        plt.gca().set_yscale('linear')
    else:
        plt.gca().set_yscale(sca)
    plt.xlabel(r"$r$")
    plt.ylabel(ylabel)

    plt.axvspan(M*(1.0-math.sqrt(1.0-a*a)), M*(1.0+math.sqrt(1.0-a*a)), 
                    color='grey', alpha=0.5)
    plt.axvspan(plt.xlim()[0], 2*M, color='lightgrey', alpha=0.5)
    plt.xlim(r.min(), r.max())

    #if bounds is not None:
    #    if sca == "log":
    #        upper = 10.0 ** (math.ceil(math.log10(bounds[1]))+0.1)
    #        if bounds[0] > 0:
    #            lower = 10.0 ** (math.floor(math.log10(bounds[0]))-0.1)
    #        else:
    #            lower = 10.0 ** (math.floor(math.log10(f[f>0].min()))-0.1)
    #        plt.ylim(lower, upper)

def plot_r_profile(filename, pars, sca='linear', plot=True, bounds=None):

    print("Reading {0:s}".format(filename))
    
    M = pars['GravM']
    a = pars['GravA']
    A = a*M

    t, r, rho, sig, T, P, pi, H, vr, vp, u0 = allTheThings(filename, pars)
    
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

    Mdot = -2*math.pi*r*rho*u0*vr*H * (eos.c * eos.rg_solar**2 / eos.M_solar)

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
        fig = plt.figure(figsize=(12,9))

        plt.subplot(331)
        plot_r_profile_single(r, rho, pars, sca, r"$\rho_0$ ($g/cm^3$)", 
                                bounds[0])
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.set_ylabel(r"$\Sigma_0$ ($g/cm^2$)")
        ax2.set_yscale(sca)
        ax2.plot(r, sig * eos.rg_solar, 'r+')

        plt.subplot(332)
        plot_r_profile_single(r, T, pars, sca, r"$T$ ($m_p c^2$)", bounds[1])

        plt.subplot(333)
        if pars['Background'] == 3 and pars['EOSType'] == 1:
            plt.plot(r, eos.P_gas(rho,T,pars), 'g+')
            plt.plot(r, eos.P_rad(rho,T,pars), 'b+')
        elif pars['Background'] == 3 and pars['EOSType'] == 2:
            plt.plot(r, eos.P_gas(rho,T,pars), 'g+')
            plt.plot(r, eos.P_rad(rho,T,pars), 'b+')
            plt.plot(r, eos.P_deg(rho,T,pars), 'y+')
        plot_r_profile_single(r, P, pars, sca, r"$P$ ($g\ c^2/cm^3$)", 
                                bounds[2])
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.set_ylabel(r"$\Pi$ ($g c^2/cm^2$)")
        ax2.set_yscale(sca)
        ax2.plot(r, pi * eos.rg_solar, 'r+')

        plt.subplot(334)
        plot_r_profile_single(r, vr, pars, "linear", r"$v^r$", bounds[3])
        plt.gca().set_xscale(sca)

        plt.subplot(335)
        plot_r_profile_single(r, vp, pars, sca, r"$v^\phi$", bounds[4])

        plt.subplot(336)
        plot_r_profile_single(r, W, pars, sca, r"$W$", bounds[5])
        ax1 = plt.gca()
        ax1.set_yscale('linear')
        ax2 = ax1.twinx()
        ax2.set_ylabel(r"$\mathcal{M}$")
        ax2.set_yscale(sca)
        ax2.plot(r, mach, 'r+')

        plt.subplot(337)
        plot_r_profile_single(r, H, pars, sca, r"$H$", bounds[6])

        plt.subplot(338)
        plot_r_profile_single(r, Mdot, pars, "linear", 
                                r"$\dot{M}$ ($M_\odot / s$)", bounds[7])
        plt.gca().set_xscale("linear")

        plt.subplot(339)
        plot_r_profile_single(r, cs, pars, sca, r"$c_s$ ($c$)", bounds[8])

        plt.tight_layout()

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
