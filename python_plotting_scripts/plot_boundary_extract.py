import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as coll
import matplotlib.colors as clrs
import matplotlib.mlab as mlab
import matplotlib.ticker as tkr
import matplotlib.tri as tri
import matplotlib.patches as pat
import scipy.optimize as opt
import discopy as dp

R1 = 0.5
R2 = 0.5
DR1 = 0.05
DR2 = 0.05

scale = "log"

ROCHE = True
XROCHE = None
YROCHE = None
ZROCHE = None
L1x = 0.0
L1val = 0.0

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

def plot_roche(ax, mesh, q):

    N = 256
    BINSEP = 1.0

    global XROCHE
    global YROCHE
    global ZROCHE
    global L1x
    global L1val

    if XROCHE == None or YROCHE == None or ZROCHE == None:
        print("Building Roche Lobe")
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x = np.linspace(xlim[0], xlim[1], N)
        y = np.linspace(ylim[0], ylim[1], N)

        M = 1.0
        M1 = 1.0 / (1.0 + q)
        M2 = 1.0 / (1.0 + 1.0/q)
        a1 = 1.0 / (1.0 + 1.0/q)
        a2 = 1.0 / (1.0 + q)

        XROCHE,YROCHE = np.meshgrid(x, y)

        def phiRoche(x, y, M1, M2, a1, a2):
            phi1 = -M1 / np.sqrt((x-a1)*(x-a1) + y*y)
            phi2 = -M2 / np.sqrt((x+a2)*(x+a2) + y*y)
            phic = -0.5 * (M1+M2) * (x*x+y*y)  / ((a1+a2)*(a1+a2)*(a1+a2))
            return phi1 + phi2 + phic

        def rochel1(x, M1, M2, a1, a2):
            f1 = M1 / math.fabs((x-a1)*(x-a1)*(x-a1)) * (x-a1)
            f2 = M2 / math.fabs((x+a2)*(x+a2)*(x+a2)) * (x+a2)
            fc = -M * x / ((a1+a2)*(a1+a2)*(a1+a2))
            return f1+f2+fc

        ZROCHE = phiRoche(XROCHE, YROCHE, M1, M2, a1, a2)
        L1x = opt.newton(rochel1, 0.0, args=(M1,M2,a1,a2))
        L1val = phiRoche(L1x, 0.0, M1, M2, a1, a2)

    lvl = L1val
    ax.contour(XROCHE, YROCHE, ZROCHE, levels=[lvl], colors='m')
    #ax.contour(XROCHE, YROCHE, ZROCHE, levels=[1.4*lvl,1.35*lvl,1.3*lvl,1.25*lvl,1.2*lvl,1.15*lvl,1.1*lvl,1.05*lvl,lvl,0.95*lvl,0.9*lvl,0.85*lvl,0.8*lvl,0.75*lvl,0.7*lvl,0.65*lvl,0.6*lvl], colors='m')
    #ax.contour(XROCHE, YROCHE, ZROCHE)

    return L1x

def buildQuiver(r, phi, vr, vp, N=20):

    print("Building Quiver")

    rmax = r.max()

    xlim = (-rmax, rmax)
    ylim = (-rmax, rmax)
    xq = np.linspace(xlim[0], xlim[1], N)
    yq = np.linspace(ylim[0], ylim[1], N)

    XQ,YQ = np.meshgrid(xq, yq)

    R2 = XQ*XQ+YQ*YQ
    ind = R2 < rmax

    x = r*np.cos(phi)
    y = r*np.sin(phi)
    vx = vr*np.cos(phi) - vp*np.sin(phi)
    vy = vr*np.sin(phi) + vp*np.cos(phi)

    VX = mlab.griddata(x, y, vx, XQ, YQ)
    VY = mlab.griddata(x, y, vy, XQ, YQ)
    
    XQ = XQ[ind]
    YQ = YQ[ind]
    VX = VX[ind]
    VY = VY[ind]
    
    return (XQ, YQ, VX, VY)

def plotQuiver(ax, r, phi, vr, vp, Vmax=-1.0, **kwargs):

    N = 60

    xlim = ax.get_xlim()

    if Vmax <= 0.0:
        Vmax = math.sqrt((vr*vr+vp*vp).max())
    scale = Vmax*(N-1)/2
    blue = (31.0/255, 119.0/255, 180.0/255)

    quiver = buildQuiver(r, phi, vr, vp, N)

    ax.quiver(*quiver, scale=scale, scale_units='width', color=blue)

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
    inds = r < 2.0
    mesh = tri.Triangulation(x[inds], y[inds])
    makeEquatPlot(fig, ax, mesh, rho[inds], scale=scale, title=title,    
                label=r"$\Sigma$", cmap=plt.cm.afmhot)
    l1x = plot_roche(ax, mesh, q)
    r1 = a1-l1x
    r2 = a2+l1x
    primary_cut = pat.Wedge((a1,0), r1+DR1, 0, 360, width=2*DR1)
    secondary_cut = pat.Wedge((-a2,0), r2+DR2, 0, 360, width=2*DR2)
    patches = coll.PatchCollection([primary_cut, secondary_cut], color='g', 
                                    linewidths=0, alpha=0.4)
    ax.add_collection(patches)
    plotQuiver(ax, r[inds], phi[inds], vr[inds], vp[inds])
    fig.savefig("bound_equat_rho_{0:010.2f}.png".format(t), dpi=300)
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

    al1 = phi1 - phi[ind1]
    al2 = phi2 - phi[ind2]

    vr1 =  vr[ind1] * np.cos(al1) + vp[ind1] * np.sin(al1)
    vp1 = -vr[ind1] * np.sin(al1) + vp[ind1] * np.cos(al1)
    vr2 =  vr[ind2] * np.cos(al2) + vp[ind2] * np.sin(al2)
    vp2 = -vr[ind2] * np.sin(al2) + vp[ind2] * np.cos(al2)

    # Primary plot
    fig, ax = plt.subplots(2,2, figsize=(12,9))
    makeBoundPlot(ax[0,0], phi1, rho[ind1], scale=scale, label=r'$\Sigma$',
                ls='', marker='+', color='k')
    makeBoundPlot(ax[0,1], phi1, P[ind1]/rho[ind1], scale=scale, 
                label=r'$T$', ls='', marker='+', color='k')
    makeBoundPlot(ax[1,0], phi1, vr1, scale='linear', label=r'$v^\hat{r}$',
                ls='', marker='+', color='k')
    makeBoundPlot(ax[1,1], phi1, vp1, scale='linear', label=r'$v^\hat{\phi}$',
                ls='', marker='+', color='k')
    fig.suptitle(title, fontsize=24)
    fig.savefig("bound_primary_{0:010.2f}.png".format(t))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12,9))
    makeBoundPlot(ax, phi1, r[ind1]*rho[ind1]*vr1, scale=scale, 
                    label=r'$\dot{M}$', ls='', marker='+', color='k')
    fig.suptitle(title, fontsize=24)
    fig.savefig("bound_primary_mdot_{0:010.2f}.png".format(t))
    plt.close(fig)

    # Secondary Plot
    fig, ax = plt.subplots(2,2, figsize=(12,9))
    makeBoundPlot(ax[0,0], phi2, rho[ind2], scale=scale, label=r'$\Sigma$',
                ls='', marker='+', color='k')
    makeBoundPlot(ax[0,1], phi2, P[ind2]/rho[ind2], scale=scale, 
                label=r'$T$', ls='', marker='+', color='k')
    makeBoundPlot(ax[1,0], phi2, vr2, scale='linear', label=r'$v^\hat{r}$',
                ls='', marker='+', color='k')
    makeBoundPlot(ax[1,1], phi2, vp2, scale='linear', label=r'$v^\hat{\phi}$',
                ls='', marker='+', color='k')
    fig.suptitle(title, fontsize=24)
    fig.savefig("bound_secondary_{0:010.2f}.png".format(t))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12,9))
    makeBoundPlot(ax, phi2, r[ind2]*rho[ind2]*vr2, scale=scale, 
                    label=r'$\dot{M}$', ls='', marker='+', color='k')
    fig.suptitle(title, fontsize=24)
    fig.savefig("bound_secondary_mdot_{0:010.2f}.png".format(t))
    plt.close(fig)



if __name__ == "__main__":

    if len(sys.argv[1:]) < 2:
        print("Need a parfile and a DiagEquat.h5 file!")
        sys.exit()
    
    pars = dp.readParfile(sys.argv[1])

    for filename in sys.argv[2:]:
        print("Plotting {0:s}...".format(filename))
        plotBoundaryExtract(filename, pars)


