import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import discopy as dp
import discoGR as gr
import discoEOS as eos

RMAX = 100

blue = (31.0/255, 119.0/255, 180.0/255)
orange = (255.0/255, 127.0/255, 14.0/255)
green = (44.0/255, 160.0/255, 44.0/255)
red = (214.0/255, 39.0/255, 40.0/255)
purple = (148.0/255, 103.0/255, 189.0/255)
darkgrey = (65.0/255, 68.0/255, 81.0/255)

def getPickleData(filename):
    f = open(filename, "r")
    data = pickle.load(f)
    f.close()
    return data

def getCompData(parfile, checkpoint):

    print("Reading {0:s} and {1:s}".format(parfile, checkpoint))

    pars = dp.readParfile(parfile)
    t, r, phi, z, sig, P, vr, vp, vz, dV, q, piph, dphi, gravMass = dp.readCheckpoint(checkpoint)

    RR = np.unique(r)

    SIG = np.empty(RR.shape)
    VR = np.empty(RR.shape)
    MACH = np.empty(RR.shape)
    MACHR = np.empty(RR.shape)

    a = gr.lapse(r, pars)
    u0, ur, up = gr.calc_u(r, vr, vp, pars)

    w = u0*a
    u = np.sqrt(w*w-1.0)
    D = u0*sig

    gam = pars['Adiabatic_Index']
    sigh = sig + gam/(gam-1) * P
    cs2 = gam * P / sigh

    mach = u * np.sqrt((1-cs2)/cs2)
    machr = mach

    for i,R in enumerate(RR):
        inds = r==R
        SIG[i] = (D[inds]*R*dphi[inds]).sum() / (2*np.pi*R)
        VR[i] = (D[inds]*vr[inds]*R*dphi[inds]).sum() / (2*np.pi*R*SIG[i])
        MACH[i] = (mach[inds]*R*dphi[inds]).sum() / (2*np.pi*R)
        MACHR[i] = (machr[inds]*R*dphi[inds]).sum() / (2*np.pi*R)


    return RR, SIG, VR, MACH, MACHR

def comparisonPlot(names):

    colors = [darkgrey, blue, orange, green, red, purple]
    #shapes = ['+', 'o', '^', 'x', 'v', '.']
    shapes = ['', '', '', '', '', '']
    #ls = ['', '', '', '', '', '']
    ls = ['-', (0,(16,8)), (0,(4,4))]
    #lw = [0,0,0,0,0,0]
    lw = [6,4,4,4,4,4]
    ms = [10,5,5,5,5,5]
    mew = [1,0,0,0,0,0]
    alpha = [1, 1, 1, 1, 1, 1]

    fig, ax = plt.subplots(2,1)
    for i,pair in enumerate(names):
        R, sig, vr, mach, machR = getCompData(*pair)
        inds = R<100
        ax[0].plot(R[inds], sig[inds], color=colors[i], marker=shapes[i], 
                                        ls=ls[i], ms=ms[i], mew=mew[i],
                                        alpha=alpha[i], lw=lw[i])
        ax[1].plot(R[inds], -vr[inds], color=colors[i], marker=shapes[i],
                                        ls=ls[i], ms=ms[i], mew=mew[i],
                                        alpha=alpha[i], lw=lw[i])
        #ax[2].plot(R[inds], mach[inds], color=colors[i], marker=shapes[i],
        #                                ls=ls[i], ms=ms[i], mew=mew[i],
        #                                alpha=alpha[i], lw=lw[i])
        #ax[3].plot(R[inds], machR[inds], color=colors[i], marker=shapes[i],
        #                                ls=ls[i], ms=ms[i], mew=mew[i],
        #                                alpha=alpha[i], lw=lw[i])

    #ax[0].set_xlabel(r'$R$')
    ax[0].set_ylabel(r'$\Sigma$')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_xlabel(r'$R$')
    ax[1].set_ylabel(r'$-v^r$')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    #ax[2].set_xlabel(r'$R$')
    #ax[2].set_ylabel(r'$\mathcal{M}$')
    #ax[2].set_xscale('log')
    #ax[2].set_yscale('log')
    #ax[3].set_xlabel(r'$R$')
    #ax[3].set_ylabel(r'$\mathcal{M}_R$')
    #ax[3].set_xscale('log')
    #ax[3].set_yscale('log')

    fig.tight_layout()
    fig.savefig("plot_minidisc_comp.pdf")

if __name__ == "__main__":

    nargs = len(sys.argv)-1
    if nargs%2 != 0 or nargs == 0:
        print("I need pairs of parfiles and checkpoints.")
        sys.exit()

    names = zip(sys.argv[1::2], sys.argv[2::2])

    comparisonPlot(names)

