import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
import discopy as dp

RMAX = 3.5
scale = "log"

def makePlot(fig, ax, mesh, dat, scale="linear", title="", label="", **kwargs):

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

def plotDiagEquat(filename):
    
    t, r, phi, rho, P, vr, vp, vz, w, dV = dp.readDiagEquat(filename)


    inds = r<RMAX

    x = r[inds] * np.cos(phi[inds])
    y = r[inds] * np.sin(phi[inds])
    mesh = tri.Triangulation(x, y)

    title = r"$t = {0:.2e}$".format(t)

    fig, ax = plt.subplots()
    makePlot(fig, ax, mesh, rho[inds], scale="linear", title=title,    
                label=r"$\Sigma$", cmap=plt.cm.afmhot)
    fig.savefig("rho_lab_{0:010.2f}.png".format(t))
    plt.close(fig)

    fig, ax = plt.subplots()
    makePlot(fig, ax, mesh, rho[inds], scale="log", title=title,    
                label=r"$\Sigma$", cmap=plt.cm.afmhot)
    fig.savefig("log_rho_lab_{0:010.2f}.png".format(t))
    plt.close(fig)

    fig, ax = plt.subplots()
    makePlot(fig, ax, mesh, P[inds]/rho[inds], scale="linear", title=title,    
                label=r"$P / \rho$", cmap=plt.cm.afmhot)
    fig.savefig("P_o_rho_lab_{0:010.2f}.png".format(t))
    plt.close(fig)

    fig, ax = plt.subplots()
    makePlot(fig, ax, mesh, P[inds]/rho[inds], scale="log", title=title,    
                label=r"$P / \rho$", cmap=plt.cm.afmhot)
    fig.savefig("log_P_o_rho_lab_{0:010.2f}.png".format(t))
    plt.close(fig)


    phi -= t
    x = r[inds] * np.cos(phi[inds])
    y = r[inds] * np.sin(phi[inds])
    mesh = tri.Triangulation(x, y)

    fig, ax = plt.subplots()
    makePlot(fig, ax, mesh, rho[inds], scale="linear", title=title,    
                label=r"$\Sigma$", cmap=plt.cm.afmhot)
    fig.savefig("rho_com_{0:010.2f}.png".format(t))
    plt.close(fig)

    fig, ax = plt.subplots()
    makePlot(fig, ax, mesh, rho[inds], scale="log", title=title,    
                label=r"$\Sigma$", cmap=plt.cm.afmhot)
    fig.savefig("log_rho_com_{0:010.2f}.png".format(t))
    plt.close(fig)

    fig, ax = plt.subplots()
    makePlot(fig, ax, mesh, P[inds]/rho[inds], scale="linear", title=title,    
                label=r"$P / \rho$", cmap=plt.cm.afmhot)
    fig.savefig("P_o_rho_com_{0:010.2f}.png".format(t))
    plt.close(fig)

    fig, ax = plt.subplots()
    makePlot(fig, ax, mesh, P[inds]/rho[inds], scale="log", title=title,    
                label=r"$P / \rho$", cmap=plt.cm.afmhot)
    fig.savefig("log_P_o_rho_com_{0:010.2f}.png".format(t))
    plt.close(fig)


if __name__ == "__main__":

    for filename in sys.argv[1:]:
        print("Plotting {0:s}...".format(filename))
        plotDiagEquat(filename)


