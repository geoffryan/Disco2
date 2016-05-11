import math
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import sys
import numpy as np
import discopy as dp

yscale = "log"

def horizon(pars):
    M = pars['GravM']
    a = pars['GravA']
    A = M*a
    return M*(1.0 + math.sqrt(1-a*a))

def calc_u0(r, phi, vr, vp, pars):

    M = pars['GravM']
    a = pars['GravA']
    A = M*a

    R = r
    SIG2 = R*R
    DEL = R*R - 2*M*R + A*A
    AAA = (R*R+A*A)*(R*R+A*A) - A*A*DEL
    B = 2*M*R/SIG2

    g00 = -1.0 + 2*M*R/SIG2
    grr = 1.0 + B
    gpp = AAA / SIG2
    g0r = B
    g0p = -B * A
    grp = -A * (1.0+B)

    u0 = 1.0/np.sqrt(-g00 - 2*g0r*vr - 2*g0p*vp - grr*vr*vr - gpp*vp*vp
                     - 2*grp*vr*vp)

    return u0

def calc_mdot(r, phi, z, rho, H, vr, vp, pars):

    inds = (z==0)
    RRR = r[inds]
    PHI = phi[inds]
    RHO = rho[inds]
    HHH = H[inds]
    VR = vr[inds]
    VP = vp[inds]
    SIG = 2*RHO[inds]*HHH[inds]

    U0 = calc_u0(r, phi, vr, vp, pars)
    UR = U0*VR

    Rs = np.unique(RRR)
    Mdot = np.zeros(Rs.shape)

    for i,R in enumerate(Rs):
        inds = (RRR==R)
        nphi = len(RRR[inds])
        dphi = 2*math.pi / nphi

        Mdot[i] = -R * (SIG[inds] * UR[inds] * dphi).sum()

    return Rs, Mdot

def get_mdot(filename, pars):

    print("Reading {0:s}".format(filename))

    dat = dp.readCheckpoint(filename)
    t = dat[0]
    r = dat[1]
    phi = dat[2]
    z = dat[3]
    rho = dat[4]
    P = dat[5]
    vr = dat[6]
    vp = dat[7]

    H = np.ones(rho.shape)

    rs, mdot = calc_mdot(r, phi, z, rho, H, vr, vp, pars)

    reh = horizon(pars)

    i = len(rs[rs < reh]) - 1
    if i == -1:
        i = 2

    return t, mdot[i], mdot[-3]

def plot_mdot(parfile, filenames):

    pars = dp.readParfile(parfile)

    t = np.zeros(len(filenames))
    mdot_inner = np.zeros(len(filenames))
    mdot_outer = np.zeros(len(filenames))

    for i,f in enumerate(filenames):
        t[i], mdot_inner[i], mdot_outer[i] = get_mdot(f, pars)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(t, mdot_inner, 'k+')
    ax.plot(t, mdot_outer, 'r+')
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\dot{M}$")
    ax.set_yscale(yscale)

    outname = "mdot.png"

    print("Saving {0:s}...".format(outname))
    plt.savefig(outname)

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("\nGive me parfile (.par) and checkpoint (.h5) files please.\n")
        sys.exit()

    else:
        fig = plot_mdot(sys.argv[1], sys.argv[2:])

