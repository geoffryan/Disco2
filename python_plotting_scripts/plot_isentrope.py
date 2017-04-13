import math
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import sys
import numpy as np
import discopy as dp
import discoEOS as eos
import discoGR as gr

scale = 'linear'

def allTheThings(filename, pars):

    dat = dp.readCheckpoint(filename)
    t = dat[0]
    r = dat[1]
    phi = dat[2]
    rho = dat[4]
    P  = dat[5]
    vr = dat[6]
    vp = dat[7]
    dV = dat[9]
    q = dat[10]
    u0, ur, up = gr.calc_u(r, vr, vp, pars)

    return t, r, phi, rho, P, vr, vp, u0, dV

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

def ceil_sig(x, sig):
    if x == 0.0:
        return 0.0
    exp = int(math.floor(math.log10(math.fabs(x))))
    y = math.ceil(x * 10.0**(-exp+sig))
    return y * 10.0**(exp-sig)

def pretty_axis(ax, pars, xscale="linear", yscale="linear", 
                xlabel=None, ylabel=None, xlim=None, ylim=None):

    ax.set_yscale(yscale)
    ax.set_ylim(ylim)
    if ylabel != None:
        ax.set_ylabel(ylabel)

    ax.set_xscale(xscale)
    ax.set_xlim(xlim)

    if xlabel != None:
        ax.set_xlabel(xlabel)

def plot_x_profile(filename, pars, sca='linear', plot=True, bounds=None):

    #print("Reading {0:s}".format(filename))
    
    t, r, phi, rho, P, vr, vp, u0, dV = allTheThings(filename, pars)
    
    x = r*np.cos(phi)
    y = r*np.sin(phi)

    if bounds is not None:
        xb = bounds[0]
        yb = bounds[0]
        inds = (x>xb[0]) * (x<xb[1]) * (y>yb[0]) * (y<yb[1])

        r = r[inds]
        phi = phi[inds]
        x = x[inds]
        y = y[inds]
        rho = rho[inds]
        P = P[inds]
        vr = vr[inds]
        vp = vp[inds]
        u0 = u0[inds]
        dV = dV[inds]

    rho0 = pars['InitPar1']
    P0 = pars['InitPar2']
    v0 = pars['InitPar3']
    L = pars['InitPar4']
    a = pars['InitPar5']
    x0 = pars['InitPar6']
    gam = pars['Adiabatic_Index']

    K = P0 * math.pow(rho0, -gam)

    rhoB = [rho0, rho0*(1+a+0.2)]
    PB = [P0, K*math.pow(rhoB[1],gam)]

    cs0 = math.sqrt(gam*P0 / (rho0 + P0*gam/(gam-1)))
    cs = np.sqrt(gam*P / (rho + P*gam/(gam-1)))
    J = 0.5*math.log((1+v0)/(1-v0)) - math.log((math.sqrt(gam-1)+cs0) 
                                    / (math.sqrt(gam-1)-cs0)) / math.sqrt(gam-1)

    X0 = np.linspace(x.min(), x.max(), 100)
    RHO = np.empty(X0.shape)
    RHO[:] = rho0
    ins = np.fabs(X0-x0)<L
    RHO[ins] = rho0 * (1+a*np.power((X0[ins]-x0)*(X0[ins]-x0)/(L*L)-1,4))
    PPP = K*np.power(RHO, gam)
    CS = np.sqrt(gam*PPP/(RHO + PPP*gam/(gam-1)))

    JS = np.exp(2*J + 2*np.log((math.sqrt(gam-1)+CS)
                                    /(math.sqrt(gam-1)-CS))/math.sqrt(gam-1))
    VVV = (JS - 1) / (JS + 1)

    X = X0 + (VVV+CS)/(1+VVV*CS)*t

    vx = vr*np.cos(phi) - r*vp*np.sin(phi)

    s = np.log(P * np.power(rho, -gam)) / (gam-1.0)
    s0 = math.log(P0 * math.pow(rho0, -gam)) / (gam-1.0)

    L1err = (np.fabs(s-s0)*dV).sum()

    if plot:


        #print("Plotting t = {0:g}".format(t))

        #Plot.
        outpath = filename.split("/")[:-1]
        chckname = filename.split("/")[-1]

        # Density
        fig, ax = plt.subplots(1,1,figsize=(12,9))
        plot_data(ax, x, rho)
        plot_line(ax, X, RHO)
        pretty_axis(ax, pars, xscale=scale, yscale=scale, 
                    ylabel=r"$\rho$", xlabel=r"$x$", ylim=rhoB, xlim=bounds[0])
        outname = "plot_cart_rho_{0}.png".format("_".join(chckname.split(".")[0].split("_")[1:]))
        #print("Saving {0:s}...".format(outname))
        plt.savefig(outname)
        plt.close(fig)
       
        # Pressure
        fig, ax = plt.subplots(1,1,figsize=(12,9))
        plot_data(ax, x, P)
        plot_line(ax, X, PPP)
        pretty_axis(ax, pars, xscale=scale, yscale=scale, 
                    ylabel=r"$P$", xlabel=r"$x$", ylim=PB, xlim=bounds[0])
        outname = "plot_cart_P_{0}.png".format("_".join(chckname.split(".")[0].split("_")[1:]))
        #print("Saving {0:s}...".format(outname))
        plt.savefig(outname)
        plt.close(fig)
        
        # Velocity
        fig, ax = plt.subplots(1,1,figsize=(12,9))
        plot_data(ax, x, vx)
        plot_line(ax, X, VVV)
        pretty_axis(ax, pars, xscale=scale, yscale=scale, 
                    ylabel=r"$v^x$", xlabel=r"$x$", ylim=[0,1], xlim=bounds[0])
        outname = "plot_cart_v_{0}.png".format("_".join(chckname.split(".")[0].split("_")[1:]))
        #print("Saving {0:s}...".format(outname))
        plt.savefig(outname)
        plt.close(fig)

    else:
        fig = None

    return fig, bounds, L1err

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("\nGive me a parfile and checkpoint (.h5) file(s).\n")
        sys.exit()

    else:
        parname = sys.argv[1]
        pars = dp.readParfile(parname)
        x0 = pars['InitPar6']
        bounds = ([x0-0.5,x0+1.0],[-0.5,0.5])

        for filename in sys.argv[2:]:
            fig, bounds, err = plot_x_profile(filename, pars, sca=scale, 
                                                plot=True, bounds=bounds)

        nr = pars['NumR']

        print nr, err



