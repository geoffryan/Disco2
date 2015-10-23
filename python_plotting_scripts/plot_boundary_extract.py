import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as coll
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
import matplotlib.tri as tri
import matplotlib.patches as pat
import discopy as dp

R1 = 0.5
R2 = 0.5
DR1 = 0.1
DR2 = 0.1

scale = "log"

def makeBoundPlot(ax, phi, dat, scale="linear", label="", **kwargs):

    ax.plot(phi, dat, **kwargs)

    ax.set_xlabel(r"$\Phi$")
    ax.set_ylabel(label)
    ax.set_yscale(scale)

def makeEquatPlot(fig, ax, mesh, dat, scale="linear", title="", label="", 
                    **kwargs):

    N = 200

    if scale == "log":
        v = np.logspace(math.floor(math.log10(dat.min())),
                        math.ceil(math.log10(dat.max())), base=10.0, num=N)
        norm = clrs.LogNorm()
        locator = tkr.LogLocator()
        formatter = tkr.LogFormatter()
    else:
        v = np.linspace(dat.min(), dat.max(), num=N)
        norm = clrs.Normalize()
        locator = tkr.AutoLocator()
        formatter = tkr.ScalarFormatter()

    C = ax.tricontourf(mesh, dat, v, norm=norm, **kwargs)
    colorbar = fig.colorbar(C, format=formatter, ticks=locator)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect('equal')
    colorbar.set_label(label)
    fig.suptitle(title, fontsize=24)

def plotBoundaryExtract(filename, pars):
   
    q = pars['MassRatio']
    M1 = 1.0 / (1.0 + q)
    M2 = 1.0 / (1.0 + 1.0/q)
    a1 = 1.0 / (1.0 + 1.0/q) 
    a2 = 1.0 / (1.0 + q)

    t, r, phi, rho, P, vr, vp, vz, w, dV = dp.readDiagEquat(filename)

    phi -= t
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    title = r"$t = {0:.2e}$".format(t)

    # Equatorial Plot
    fig, ax = plt.subplots()
    inds = r < 3.5
    mesh = tri.Triangulation(x[inds], y[inds])
    makeEquatPlot(fig, ax, mesh, rho[inds], scale="linear", title=title,    
                label=r"$\Sigma$", cmap=plt.cm.afmhot)
    primary_cut = pat.Wedge((a1,0), R1+DR1, 0, 360, width=2*DR1)
    secondary_cut = pat.Wedge((-a2,0), R2+DR2, 0, 360, width=2*DR2)
    patches = coll.PatchCollection([primary_cut, secondary_cut], color='g', 
                                    alpha=0.4)
    ax.add_collection(patches)
    fig.savefig("bound_equat_rho_{0:010.2f}.png".format(t))
    plt.close(fig)

    #Extract boundary
    ind1 = np.fabs(np.sqrt((x-a1)*(x-a1)+y*y) - R1) < DR1
    ind2 = np.fabs(np.sqrt((x+a2)*(x+a2)+y*y) - R2) < DR2

    x1 = x[ind1]-a1
    y1 = y[ind1]
    x2 = x[ind2]+a2
    y2 = y[ind2]

    phi1 = np.arctan2(y1, x1)
    phi2 = np.arctan2(y2, x2)

    # Primary plot
    fig, ax = plt.subplots(2,2, figsize=(12,9))
    makeBoundPlot(ax[0,0], phi1, rho[ind1], scale='linear', label=r'$\Sigma$',
                ls='', marker='+', color='k')
    makeBoundPlot(ax[0,1], phi1, P[ind1]/rho[ind1], scale='linear', 
                label=r'$T$', ls='', marker='+', color='k')
    makeBoundPlot(ax[1,0], phi1, vr[ind1], scale='linear', label=r'$v^r$',
                ls='', marker='+', color='k')
    makeBoundPlot(ax[1,1], phi1, vp[ind1], scale='linear', label=r'$v^\phi$',
                ls='', marker='+', color='k')
    fig.suptitle(title, fontsize=24)
    fig.savefig("bound_primary_{0:010.2f}.png".format(t))
    plt.close(fig)

    # Secondary Plot
    fig, ax = plt.subplots(2,2, figsize=(12,9))
    makeBoundPlot(ax[0,0], phi2, rho[ind2], scale='linear', label=r'$\Sigma$',
                ls='', marker='+', color='k')
    makeBoundPlot(ax[0,1], phi2, P[ind2]/rho[ind2], scale='linear', 
                label=r'$T$', ls='', marker='+', color='k')
    makeBoundPlot(ax[1,0], phi2, vr[ind2], scale='linear', label=r'$v^r$',
                ls='', marker='+', color='k')
    makeBoundPlot(ax[1,1], phi2, vp[ind2], scale='linear', label=r'$v^\phi$',
                ls='', marker='+', color='k')
    fig.suptitle(title, fontsize=24)
    fig.savefig("bound_secondary_{0:010.2f}.png".format(t))
    plt.close(fig)



if __name__ == "__main__":

    pars = dp.readParfile(sys.argv[1])

    for filename in sys.argv[2:]:
        print("Plotting {0:s}...".format(filename))
        plotBoundaryExtract(filename, pars)


