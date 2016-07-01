import math
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.signal as signal
import sys
import numpy as np
import discopy as dp
import discoGR as gr
import discoEOS as eos
import pickle

yscale = "log"

labelsize = 24
ticksize = 18

blue = (31.0/255, 119.0/255, 180.0/255)
orange = (255.0/255, 127.0/255, 14.0/255)
green = (44.0/255, 160.0/255, 44.0/255)
red = (214.0/255, 39.0/255, 40.0/255)
purple = (148.0/255, 103.0/255, 189.0/255)
brown = (140.0/255, 86.0/255, 75.0/255)
pink = (227.0/255, 119.0/255, 194.0/255)
grey = (127.0/255, 127.0/255, 127.0/255)
yellow = (188.0/255, 189.0/255, 34.0/255)
teal = (23.0/255, 190.0/255, 207.0/255)

lightblue = (174.0/255, 199.0/255, 232.0/255)
lightorange = (255.0/255, 187.0/255, 120.0/255)
lightgreen = (152.0/255, 223.0/255, 138.0/255)
lightred = (255.0/255, 152.0/255, 150.0/255)
lightpurple = (197.0/255, 176.0/255, 213.0/255)
lightbrown = (196.0/255, 156.0/255, 148.0/255)
lightpink = (247.0/255, 182.0/255, 210.0/255)
lightgrey = (199.0/255, 199.0/255, 199.0/255)
lightyellow = (219.0/255, 219.0/255, 141.0/255)
lightteal = (23.0/255, 190.0/255, 207.0/255)

def horizon(pars):
    M = pars['GravM']
    a = pars['GravA']
    A = M*a
    return M*(1.0 + math.sqrt(1-a*a))

def calc_mdot(r, phi, z, rho, H, vr, vp, pars):

    inds = (z==0)
    RRR = r[inds]
    PHI = phi[inds]
    RHO = rho[inds]
    HHH = H[inds]
    VR = vr[inds]
    VP = vp[inds] + pars['BinW']
    SIG = RHO[inds]*HHH[inds]

    U0, UR, UP = gr.calc_u(RRR, VR, VP, pars)

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

    if pars['Metric'] == 6 and pars['BoostType'] == 1:
        t /= 2*np.pi/pars['BinW']

    if pars['BoundTypeSource'] == 6:
        Mdot0_cgs = pars['BoundPar2'] * eos.M_solar / eos.year
        Mdot0_code = Mdot0_cgs / (eos.rho_scale * eos.rg_solar**2 * eos.c)
        mdot /= Mdot0_code

    return t, rs, mdot

def plot_mdot(parfile, filenames):

    pars = dp.readParfile(parfile)

    t = np.zeros(len(filenames))
    mdot_inner = np.zeros(len(filenames))
    mdot_outer = np.zeros(len(filenames))

    ts = []
    rss = []
    mdots = []

    for i,f in enumerate(filenames):
        t, rs, mdot = get_mdot(f, pars)
        ts.append(t)
        rss.append(rs)
        mdots.append(mdot)

    ts = np.array(ts)
    rss = np.array(rss)
    mdots = np.array(mdots)

    f = open("mdot.dat", "w")
    pickle.dump({'T':ts, 'R':rss, 'Mdot':mdots, 'pars':pars}, f, protocol=-1)
    f.close()

    plot_io(ts, rss, mdots)
    plot_periodogram(ts, rss, mdots)

def plot_io(ts, rss, mdots):
    fig, ax = plt.subplots(1, 1)

    ax.plot(ts, mdots[:,2], color=blue, marker='+', ls='',
            label=r'$\dot{M}(r=r_{in})$')
    ax.plot(ts, mdots[:,-3], color=orange, marker='x', ls='',
            label=r'$\dot{M}(r=r_{out})$')
    ax.set_xlabel(r"$t$ $(T_{bin})$", fontsize=labelsize)
    ax.set_ylabel(r"$\dot{M}$ $(\dot{M}_{nozzle})$", fontsize=labelsize)
    ax.set_yscale(yscale)

    legend = ax.legend(fontsize=ticksize)

    outname = "mdot.pdf"

    print("Saving {0:s}...".format(outname))
    fig.savefig(outname)
    plt.close(fig)

def plot_periodogram(ts, rss, mdots):
    fig, ax = plt.subplots(5, 1, figsize=(20,10))
    nr = rss.shape[1]

    t0 = 15.0
    fs = 1.0 / (ts[1:]-ts[:-1]).mean()

    ans = np.array([2,nr/5,2*nr/5,3*nr/5,-3])

    for i in xrange(5):
        ax[i].plot(ts, mdots[:,ans[i]], 'k+')
        ax[i].set_ylabel(r"$\dot{M}$"+r"$(r={0:.1f})$".format(rss[0,ans[i]]))
    ax[-1].set_xlabel(r"$t$ $(T_{bin})$")

    outname = "mdot_mdot.png"
    print("Saving {0:s}...".format(outname))
    fig.savefig(outname)
    plt.close(fig)


    fig, ax = plt.subplots(5, 1, figsize=(20,10))

    for i in xrange(5):
        f, p = signal.periodogram(mdots[ts>=t0,ans[i]], fs=fs)
        ax[i].plot(f, p, 'k+')
        ax[i].set_ylabel(r"$\mathcal{P}[\dot{M}$"
                            +r"$(r={0:.1f})]$".format(rss[0,ans[i]]))

    ax[-1].set_xlabel(r"$f (1/T_{bin})$")

    outname = "mdot_periodogram.png"

    print("Saving {0:s}...".format(outname))
    fig.savefig(outname)
    plt.close(fig)

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("\nGive me parfile (.par) and checkpoint (.h5) files please.\n")
        sys.exit()

    else:
        fig = plot_mdot(sys.argv[1], sys.argv[2:])

