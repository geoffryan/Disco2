import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import discopy as dp

RMAX = 10.0

def plotDiagEquat(filename):

    t, r, phi, rho, P, vr, vp, vz, w, dV = dp.readDiagEquat(filename)

    inds = r<RMAX

    x = r[inds] * np.cos(phi[inds])
    y = r[inds] * np.sin(phi[inds])
    mesh = tri.Triangulation(x, y)

    fig, ax = plt.subplots()

    C = ax.tricontourf(mesh, rho[inds], 200, cmap=plt.cm.afmhot)
    colorbar = fig.colorbar(C)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    fig.suptitle(r"$t = {0:.2g}$".format(t), fontsize=24)

    fig.savefig("rho_{0:07.2f}.png".format(t))
    plt.close()

if __name__ == "__main__":

    for filename in sys.argv[1:]:
        print("Plotting {0:s}...".format(filename))
        plotDiagEquat(filename)


