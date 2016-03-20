import math
import pickle
import h5py as h5
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.collections as coll
import matplotlib.colors as clrs
import matplotlib.patches as pat
import matplotlib.ticker as tkr
import matplotlib.mlab as mlab
import sys
import numpy as np
import scipy.optimize as opt
import discopy as dp
import plot_disc as pd
import binary_orbit as bo
import ode

#poscmap = plt.cm.afmhot
poscmap = dp.viridis
divcmap = plt.cm.RdBu

def plot_rect_single(fig, ax, mesh, dat, pars, gridbounds=None,
                        datscale="linear", datbounds=None, label=None,
                        normal=False, **kwargs):

    N = 400

    #Set data bounds & scale.
    if datbounds is None:
        datbounds = np.array([dat.min(), dat.max()])

    if datscale == "log":
        v = np.logspace(math.floor(math.log10(datbounds[0])), 
                        math.ceil(math.log10(datbounds[1])), 
                        base=10.0, num=N)
        norm = clrs.LogNorm()
        locator = tkr.LogLocator()
        formatter = tkr.LogFormatter()
    else:
        if normal:
            datm = max(abs(datbounds[0]), abs(datbounds[1]))
            v = np.linspace(-datm, datm, N)
        else:
            v = np.linspace(datbounds[0], datbounds[1], N)
        norm = clrs.Normalize()
        locator = tkr.AutoLocator()
        formatter = tkr.ScalarFormatter()

    #Plot Disc Image
    C = ax.tricontourf(mesh, dat, v, norm=norm, **kwargs)
    colorbar = fig.colorbar(C, format=formatter, ticks=locator)

    #Formatting
    #ax.set_aspect('equal')

    if gridbounds is not None:
        ax.set_xlim(gridbounds[0])
        ax.set_ylim(gridbounds[1])

    #Labels
    if label is not None:
        colorbar.set_label(label)
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$r$')

def make_plot(mesh, dat, pars, gridbounds=None, datscale="linear", 
                datbounds=None, label=None, title=None, 
                filename=None, normal=False, **kwargs):

    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(1,1,1)
    plot_rect_single(fig, ax, mesh, dat, pars, gridbounds,
                    datscale, datbounds, label, normal, **kwargs)

    if title is not None:
        ax.set_title(title)
    if filename is not None:
        print("Saving {0:s}...".format(filename))
        fig.savefig(filename)
    plt.close()

def plot_all(filename, pars, rmax=-1.0, plot=True, bounds=None):

    print("Reading {0:s}".format(filename))

    t, r, phi, rho, sig, T, P, pi, H, vr, vp, u0, Q = pd.allTheThings(filename,
                                                                        pars)

    gam = pars['Adiabatic_Index']
    S = np.log(pi * np.power(sig, -gam)) / (gam-1.0)

    if bounds is None:
        bounds = []
        bounds.append([sig[sig==sig].min(), sig[sig==sig].max()])
        bounds.append([T[T==T].min(), T[T==T].max()])
        bounds.append([vr[vr==vr].min(), vr[vr==vr].max()])
        bounds.append([vp[vp==vp].min(), vp[vp==vp].max()])
        bounds.append([S[S==S].min(), S[S==S].max()])
        for q in Q:
            bounds.append([q[q==q].min(), q[q==q].max()])
        bounds = np.array(bounds)

    if plot:

        print("Plotting t = {0:g}".format(t))

        x = phi
        y = r

        x[x < -np.pi] += 2*np.pi
        x[x >  np.pi] -= 2*np.pi

        mesh = tri.Triangulation(x, y)

        if rmax > 0.0:
            gridbounds = np.array([[-np.pi,np.pi],[r.min(),rmax]])
        else:
            gridbounds = np.array([[-np.pi,np.pi],[r.min(),r.max()]])

        outpath = filename.split("/")[:-1]
        chckname = filename.split("/")[-1]

        title = "t = {0:.3g}".format(t)

        signame = "plot_disc_rect_{0}_{1}.png".format(
                    "_".join(chckname.split(".")[0].split("_")[1:]), "sig")
        logsigname = "plot_disc_rect_{0}_{1}.png".format(
                    "_".join(chckname.split(".")[0].split("_")[1:]), "logsig")
        Tname = "plot_disc_rect_{0}_{1}.png".format(
                    "_".join(chckname.split(".")[0].split("_")[1:]), "T")
        logTname = "plot_disc_rect_{0}_{1}.png".format(
                    "_".join(chckname.split(".")[0].split("_")[1:]), "logT")
        vrname = "plot_disc_rect_{0}_{1}.png".format(
                    "_".join(chckname.split(".")[0].split("_")[1:]), "vr")
        vpname = "plot_disc_rect_{0}_{1}.png".format(
                    "_".join(chckname.split(".")[0].split("_")[1:]), "vp")
        Sname = "plot_disc_rect_{0}_{1}.png".format(
                    "_".join(chckname.split(".")[0].split("_")[1:]), "S")
        #Plot.

        #Density
        make_plot(mesh, sig, pars, gridbounds=gridbounds, datscale="linear", 
                datbounds=bounds[0], label=r'$\Sigma_0$', title=title, 
                filename=signame, cmap=poscmap)
        make_plot(mesh, sig, pars, gridbounds=gridbounds, datscale="log", 
                datbounds=bounds[0], label=r'$\Sigma_0$', title=title, 
                filename=logsigname, cmap=poscmap)

        #T
        make_plot(mesh, T, pars, gridbounds=gridbounds, datscale="linear", 
                datbounds=bounds[1], label=r'$T$', title=title, 
                filename=Tname, cmap=poscmap)
        make_plot(mesh, T, pars, gridbounds=gridbounds, datscale="log", 
                datbounds=bounds[1], label=r'$T$', title=title, 
                filename=logTname, cmap=poscmap)

        
        #Vr
        make_plot(mesh, vr, pars, gridbounds=gridbounds, datscale="linear", 
                datbounds=bounds[2], label=r'$v^r$', title=title, 
                filename=vrname, normal=True, cmap=divcmap)

        #Vp
        make_plot(mesh, vp, pars, gridbounds=gridbounds, datscale="linear", 
                datbounds=bounds[3], label=r'$v^\phi$', title=title, 
                filename=vpname, normal=True, cmap=divcmap)
        
        #S
        make_plot(mesh, S, pars, gridbounds=gridbounds, datscale="linear", 
                datbounds=bounds[4], label=r'$S$', title=title, 
                filename=Sname, normal=True, cmap=poscmap)
        
        #q
        for i,q in enumerate(Q):
            qname = "plot_disc_rect_{0}_{1}{2:01d}.png".format(
                        "_".join(chckname.split(".")[0].split("_")[1:]), "q",
                        i)
            make_plot(mesh, q, pars, gridbounds=gridbounds, datscale="linear", 
                        datbounds=bounds[5], label=r'$q_{0:d}$'.format(i), 
                        title=title, filename=qname, cmap=poscmap)

    return bounds


if __name__ == "__main__":

    #Parse Arguments

    if 'rmax' in sys.argv:
        i = sys.argv.index('rmax')
        rmax = float(sys.argv[i+1])
        sys.argv.pop(i)
        sys.argv.pop(i)
    else:
        rmax = -1.0

    # Run

    if len(sys.argv) < 3:
        print("\nusage: python plot_disc_rect.py <parfile> <checkpoints...> [roche] [quiver] [rmax RMAX]")
        print("   Creates equatorial plots of data in checkpoint(s) created by Disco run with parfile.")
        print("      rmax RMAX: Plot only r < RMAX.\n")
        sys.exit()

    elif len(sys.argv) == 3:
        pars = dp.readParfile(sys.argv[1])
        filename = sys.argv[2]

        plot_all(filename, pars, rmax=rmax)
        plt.show()

    else:
        pars = dp.readParfile(sys.argv[1])
        bounds = None
        try:
            f = open("bounds.dat", "r")
            print("Trying to unpickle bounds...")
            bounds = pickle.load(f)
            f.close()
        except:
            print("Making bounds from scratch...")
            for filename in sys.argv[2:]:
                b = plot_all(filename, pars, rmax=rmax, plot=False)
                if bounds is None:
                    bounds = b.copy()
                else:
                    lower = b[:,0]<bounds[:,0]
                    upper = b[:,1]>bounds[:,1]
                    bounds[lower,0] = b[lower,0]
                    bounds[upper,1] = b[upper,1]
            f = open("bounds.dat", "w")
            pickle.dump(bounds, f, protocol=-1)
            f.close()
        
        for filename in sys.argv[2:]:
            plot_all(filename, pars, rmax=rmax, plot=True, bounds=bounds)

