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

def plot_equat_single(fig, ax, mesh, dat, pars, gridbounds=None,
                        datscale="linear", datbounds=None, rocheData=None,
                        quiverData=None, Vmax=0.0, orbitData=None, label=None,
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

    #Plot Roche Equipotential
    if rocheData is not  None:
        plot_roche(ax, rocheData)

    if quiverData is not None:
        plot_quiver(ax, quiverData, Vmax=Vmax)

    if orbitData is not None:
        plot_orbit(ax, orbitData)

    #Patches to highlight horizon and ergosphere
    M = pars['GravM']
    a = pars['GravA']
    if M > 0.0:
        ergo = pat.Circle((0,0), 2*M)
        horizon = pat.Wedge((0,0), M*(1.0+math.sqrt(1.0-a*a)), 0, 360, 
                            2*M*math.sqrt(1.0-a*a))
        patches = coll.PatchCollection([ergo,horizon], cmap=plt.cm.Greys,
                                        alpha=0.4)
        colors = np.array([0.1,0.3])
        patches.set_array(colors)
        ax.add_collection(patches)

    #Formatting
    ax.set_aspect('equal')

    if gridbounds is not None:
        ax.set_xlim(gridbounds[0])
        ax.set_ylim(gridbounds[1])

    #Labels
    if label is not None:
        colorbar.set_label(label)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

def calcRoche(pars, xlim, ylim, N=256):

    print("Building Roche Lobe")

    M2 = pars['GravM']
    M1 = pars['BinM']
    a = pars['BinA']
    w = pars['BinW']

    M = M1+M2
    q = M2/M1
    com = (-a*M1 + 0.0*M2) / M

    x = np.linspace(xlim[0], xlim[1], N)
    y = np.linspace(ylim[0], ylim[1], N)

    if 0.0 in x or 0.0 in y:
        x = np.linspace(xlim[0], xlim[1], N-1)
        y = np.linspace(ylim[0], ylim[1], N-1)


    X, Y = np.meshgrid(x, y)

    def phiRoche(x, y, M1, M2, a, w):
        return -M2/np.sqrt(x*x+y*y) - M1/np.sqrt((x+a)*(x+a)+y*y) \
            - 0.5 * w*w * ((x-com)*(x-com)+y*y)

    def fxRoche(x, M1, M2, a, w):
        f1 = - M1 / math.fabs((x+a)*(x+a)*(x+a)) * (x+a)
        f2 = - M2 / math.fabs(x*x*x) * x
        fc = w*w * (x-com)
        return f1+f2+fc

    phi = phiRoche(X, Y, M1, M2, a, w)

    L1x = opt.newton(fxRoche, -0.5*a, args=(M1, M2, a, w))
    L2x = opt.newton(fxRoche, -2*a, args=(M1, M2, a, w))
    L3x = opt.newton(fxRoche, q*a, args=(M1, M2, a, w))

    L1phi = phiRoche(L1x, 0.0, M1, M2, a, w)
    L2phi = phiRoche(L2x, 0.0, M1, M2, a, w)
    L3phi = phiRoche(L3x, 0.0, M1, M2, a, w)

    lvls = [L1phi, L2phi, L3phi]
    print("   L1, L2, L3: ({0:f}, {1:f}, {2:g})".format(L1x, L2x, L3x))

    return X, Y, phi, lvls

def plot_roche(ax, rocheData):

    x = rocheData[0]
    y = rocheData[1]
    phi = rocheData[2]
    lvls = rocheData[3]

    ax.contour(x, y, phi, levels=lvls, colors='m', ls='--')

def calcQuiver(r, phi, vr, vp, gridbounds, N=20):

    print("Building Quiver")
    if gridbounds is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    else:
        xlim = gridbounds[0]
        ylim = gridbounds[1]
    xq = np.linspace(xlim[0], xlim[1], N)
    yq = np.linspace(ylim[0], ylim[1], N)

    XQ,YQ = np.meshgrid(xq, yq)

    rm = np.fabs(gridbounds).min()
    R2 = XQ*XQ+YQ*YQ
    ind = R2 < rm*rm

    x = r*np.cos(phi)
    y = r*np.sin(phi)
    vx = vr*np.cos(phi) - r*vp*np.sin(phi)
    vy = vr*np.sin(phi) + r*vp*np.cos(phi)

    VX = mlab.griddata(x, y, vx, XQ, YQ)
    VY = mlab.griddata(x, y, vy, XQ, YQ)
    
    XQ = XQ[ind]
    YQ = YQ[ind]
    VX = VX[ind]
    VY = VY[ind]
    
    return XQ, YQ, VX, VY

def plot_quiver(ax, quiverData, Vmax=0.0, **kwargs):

    N = 20
    xlim = ax.get_xlim()
    scale = Vmax*(N-1)
    blue = (31.0/255, 119.0/255, 180.0/255)

    ax.quiver(*quiverData, scale=scale, scale_units='width', color=blue)
    
def calcOrbit(pars, filename, N=10000):
    
    f = h5.File(filename, 'r')
    t = f['T'][0]
    f.close()

    M1 = pars['GravM']
    M2 = pars['BinM']
    a = pars['BinA']
    w = pars['BinW']
    M = M1+M2

    q = M2/M1
    a1 = a / (1.0+1.0/q)
    print M1, M2, M
    print a, w, math.sqrt(M/(a*a*a))

    # Parse orbit file
    r0, p0, vr0, vp0 = np.loadtxt("orbit.txt", usecols=[0,1,2,3], unpack=True)

    # Convert Disco frame to lab center of mass frame
    n = r0.shape[0]
    x0 = r0 * np.cos(p0)
    y0 = r0 * np.sin(p0)
    vx0 = vr0 * np.cos(p0) - r0*(vp0+w)*np.sin(p0)
    vy0 = vr0 * np.sin(p0) + r0*(vp0+w)*np.cos(p0) + a1*w

    print vr0, vp0
    print vx0, vy0

    X0 = np.zeros(4*n)
    X0[ ::4] = x0 + a1
    X0[1::4] = y0
    X0[2::4] = vx0
    X0[3::4] = vy0

    # Run orbit integrator
    T, Xlab = bo.evolve_rk(X0, 0.0, t, N, ode.rk4, M, q, a)
    bo.plot_trajectory(T, Xlab, M, q, a, 2*a)
    plt.show()

    # Convert orbits back to Disco frame
    xlab = Xlab[:, ::4]
    ylab = Xlab[:,1::4]

    coswt = np.cos(w*T)
    sinwt = np.sin(w*T)

    x = xlab[:,:]*coswt[:,None] + ylab[:,:]*sinwt[:,None] - a1
    y = -xlab[:,:]*sinwt[:,None] + ylab[:,:]*coswt[:,None]

    orbitData = []
    for i in xrange(n):
        orbitData.append(np.array([T,x[:,i],y[:,i]]))

    return orbitData

def plot_orbit(ax, orbitData):
    for orbit in orbitData:
        ax.plot(orbit[0], orbit[1], 'r')

def make_plot(mesh, dat, pars, gridbounds=None, datscale="linear", 
                datbounds=None, rocheData=None, quiverData=None, 
                Vmax=0.0, orbitData=None, label=None, title=None, 
                filename=None, normal=False, **kwargs):

    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(1,1,1)
    plot_equat_single(fig, ax, mesh, dat, pars, gridbounds,
                    datscale, datbounds, rocheData, quiverData, Vmax, 
                    orbitData, label, normal, **kwargs)

    if title is not None:
        ax.set_title(title)
    if filename is not None:
        print("Saving {0:s}...".format(filename))
        fig.savefig(filename)
    plt.close()

def plot_all(filename, pars, rmax=-1.0, plot=True, bounds=None, 
                plotRoche=False, plotQuiver=False, Vmax=0.0, orbitData=None):

    print("Reading {0:s}".format(filename))

    t, r, phi, rho, sig, T, P, pi, H, vr, vp, u0, Q = pd.allTheThings(filename,
                                                                        pars)

    gam = pars['Adiabatic_Index']
    S = np.log(pi * np.power(sig, -gam))/(gam-1.0)

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

    Vmax = max(Vmax, math.sqrt((vr*vr+r*r*vp*vp).max()))

    if plot:

        print("Plotting t = {0:g}".format(t))

        x = r*np.cos(phi)
        y = r*np.sin(phi)
        mesh = tri.Triangulation(x, y)

        if rmax > 0.0:
            gridbounds = np.array([[-rmax,rmax],[-rmax,rmax]])
        else:
            gridbounds = np.array([[-r.max(),r.max()],[-r.max(),r.max()]])

        if plotRoche:
            rocheData = calcRoche(pars, gridbounds[0], gridbounds[1])
        else:
            rocheData = None

        if plotQuiver:
            quiverData = calcQuiver(r, phi, vr, vp, gridbounds)
        else:
            quiverData = None

        outpath = filename.split("/")[:-1]
        chckname = filename.split("/")[-1]

        title = "t = {0:.3g}".format(t)

        signame = "plot_disc_equat_{0}_{1}.png".format(
                    "_".join(chckname.split(".")[0].split("_")[1:]), "sig")
        logsigname = "plot_disc_equat_{0}_{1}.png".format(
                    "_".join(chckname.split(".")[0].split("_")[1:]), "logsig")
        Tname = "plot_disc_equat_{0}_{1}.png".format(
                    "_".join(chckname.split(".")[0].split("_")[1:]), "T")
        logTname = "plot_disc_equat_{0}_{1}.png".format(
                    "_".join(chckname.split(".")[0].split("_")[1:]), "logT")
        vrname = "plot_disc_equat_{0}_{1}.png".format(
                    "_".join(chckname.split(".")[0].split("_")[1:]), "vr")
        vpname = "plot_disc_equat_{0}_{1}.png".format(
                    "_".join(chckname.split(".")[0].split("_")[1:]), "vp")
        Sname = "plot_disc_equat_{0}_{1}.png".format(
                    "_".join(chckname.split(".")[0].split("_")[1:]), "s")
        #Plot.

        localOrbit=None
        if orbitData is not None:
            localOrbit = []
            for orbit in orbitData:
                inds = orbit[0] < t
                localOrbit.append(orbit[1:,inds])

        #Density
        make_plot(mesh, sig, pars, gridbounds=gridbounds, datscale="linear", 
                datbounds=bounds[0], rocheData=rocheData, 
                quiverData=quiverData, Vmax=Vmax, orbitData=localOrbit,
                label=r'$\Sigma_0$', title=title, filename=signame, 
                cmap=poscmap)
        make_plot(mesh, sig, pars, gridbounds=gridbounds, datscale="log", 
                datbounds=bounds[0], rocheData=rocheData,
                quiverData=quiverData, Vmax=Vmax, orbitData=localOrbit,
                label=r'$\Sigma_0$', title=title, filename=logsigname, 
                cmap=poscmap)

        #T
        make_plot(mesh, T, pars, gridbounds=gridbounds, datscale="linear", 
                datbounds=bounds[1],  rocheData=rocheData,
                quiverData=quiverData, Vmax=Vmax, orbitData=localOrbit, 
                label=r'$T$', title=title, filename=Tname, cmap=poscmap)
        make_plot(mesh, T, pars, gridbounds=gridbounds, datscale="log", 
                datbounds=bounds[1],  rocheData=rocheData, 
                quiverData=quiverData, Vmax=Vmax, orbitData=localOrbit, 
                label=r'$T$', title=title, filename=logTname, cmap=poscmap)

        
        #Vr
        make_plot(mesh, vr, pars, gridbounds=gridbounds, datscale="linear", 
                datbounds=bounds[2], rocheData=rocheData, 
                quiverData=quiverData, Vmax=Vmax, orbitData=localOrbit, 
                label=r'$v^r$', title=title, filename=vrname, normal=True, 
                cmap=divcmap)

        #Vp
        make_plot(mesh, vp, pars, gridbounds=gridbounds, datscale="linear", 
                datbounds=bounds[3], rocheData=rocheData,
                quiverData=quiverData, Vmax=Vmax, orbitData=localOrbit, 
                label=r'$v^\phi$', title=title, filename=vpname, normal=True, 
                cmap=divcmap)
        #S
        make_plot(mesh, S, pars, gridbounds=gridbounds, datscale="linear", 
                datbounds=bounds[4],  rocheData=rocheData,
                quiverData=quiverData, Vmax=Vmax, orbitData=localOrbit, 
                label=r'$S$', title=title, filename=Sname, cmap=poscmap)
        
        #q
        for i,q in enumerate(Q):
            qname = "plot_disc_equat_{0}_{1}{2:01d}.png".format(
                        "_".join(chckname.split(".")[0].split("_")[1:]), "q",
                        i)
            make_plot(mesh, q, pars, gridbounds=gridbounds, datscale="linear", 
                        datbounds=bounds[5], rocheData=rocheData, 
                        quiverData=quiverData, Vmax=Vmax, 
                        orbitData=localOrbit, label=r'$q_{0:d}$'.format(i), 
                        title=title, filename=qname, cmap=poscmap)

    return bounds, Vmax


if __name__ == "__main__":

    #Parse Arguments

    if 'roche' in sys.argv:
        plotRoche = True
        sys.argv.remove('roche')
    else:
        plotRoche = False

    if 'quiver' in sys.argv:
        plotQuiver = True
        sys.argv.remove('quiver')
    else:
        plotQuiver = False

    if 'rmax' in sys.argv:
        i = sys.argv.index('rmax')
        rmax = float(sys.argv[i+1])
        sys.argv.pop(i)
        sys.argv.pop(i)
    else:
        rmax = -1.0

    if 'orbit' in sys.argv:
        sys.argv.remove('orbit')
        plotOrbit = True
    else:
        plotOrbit = False

    # Run

    if len(sys.argv) < 3:
        print("\nusage: python plot_disc_equat.py <parfile> <checkpoints...> [roche] [quiver] [rmax RMAX]")
        print("   Creates equatorial plots of data in checkpoint(s) created by Disco run with parfile.")
        print("      roche:     Plot roche lobe of system.")
        print("      quiver:    Plot velocity field as quiver of arrows.")
        print("      rmax RMAX: Plot only r < RMAX.\n")
        sys.exit()

    elif len(sys.argv) == 3:
        pars = dp.readParfile(sys.argv[1])
        filename = sys.argv[2]

        if plotOrbit:
            orbitData = calcOrbit(pars, filename)
        else:
            orbitData = None
        plot_all(filename, pars, rmax=rmax, plotRoche=plotRoche, 
                    plotQuiver=plotQuiver, orbitData=orbitData)
        plt.show()

    else:
        pars = dp.readParfile(sys.argv[1])
        bounds = None
        Vmax = 0.0
        try:
            f = open("bounds.dat", "r")
            print("Trying to unpickle bounds...")
            bounds, Vmax = pickle.load(f)
            f.close()
        except:
            print("Making bounds from scratch...")
            for filename in sys.argv[2:]:
                b, Vmax = plot_all(filename, pars, rmax=rmax, plot=False, 
                                    Vmax=Vmax)
                if bounds is None:
                    bounds = b.copy()
                else:
                    lower = b[:,0]<bounds[:,0]
                    upper = b[:,1]>bounds[:,1]
                    bounds[lower,0] = b[lower,0]
                    bounds[upper,1] = b[upper,1]
            f = open("bounds.dat", "w")
            pickle.dump((bounds, Vmax), f, protocol=-1)
            f.close()
        
        if plotOrbit:
            orbitData = calcOrbit(pars, sys.argv[-1])
        else:
            orbitData = None

        for filename in sys.argv[2:]:
            plot_all(filename, pars, rmax=rmax, plot=True, bounds=bounds, 
                        plotRoche=plotRoche, plotQuiver=plotQuiver, 
                        orbitData=orbitData)

