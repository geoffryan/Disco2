import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import discopy as dp
import discoEOS as eos

def getPickleData(filename):
    f = open(filename, "r")
    data = pickle.load(f)
    f.close()
    return data

def calcAccretionTime(analysisFile, mdotFile):

    analysisDat = getPickleData(analysisFile)
    mdotDat = getPickleData(mdotFile)

    R = analysisDat['R']
    sig = analysisDat['sig']

    pars = mdotDat['pars']
    t = mdotDat['T']
    mdot = mdotDat['Mdot'][:,2]
    mdot = mdot[(t>=25)].mean()

    Mdot0_cgs = pars['BoundPar2'] * eos.M_solar / eos.year
    Mdot0_code = Mdot0_cgs / (eos.rho_scale * eos.rg_solar**2 * eos.c)

    mdot *= Mdot0_code

    g = dp.Grid(pars=pars)
    i0 = np.searchsorted(g.rFaces, R[0]) - 1
    rf = g.rFaces[i0: i0+R.shape[0]+1]
    dR = rf[1:] - rf[:-1]
    M = 2*np.pi*(R*sig*dR).sum()

    return M, mdot, Mdot0_code

def accretionTimePlot(names):

    data = []
    for pair in names:
        res = calcAccretionTime(*pair)
        data.append(res)

    data = np.array(data)

    M = data[:,0]
    Mdot = data[:,1]
    Mdot0 = data[:,2]

    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(Mdot0, M/Mdot, 'k+')
    ax[0,0].set_xlabel(r'$\dot{M}_0$')
    ax[0,0].set_ylabel(r'$t_{acc}$')
    ax[0,0].set_xscale('log')
    ax[0,0].set_yscale('log')
    ax[0,1].plot(Mdot0, M/Mdot0, 'k+')
    ax[0,1].set_xlabel(r'$\dot{M}_0$')
    ax[0,1].set_ylabel(r'$t_{acc,0}$')
    ax[0,1].set_xscale('log')
    ax[0,1].set_yscale('log')
    ax[1,0].plot(Mdot0, M, 'k+')
    ax[1,0].set_xlabel(r'$\dot{M}_0$')
    ax[1,0].set_ylabel(r'$M$')
    ax[1,0].set_xscale('log')
    ax[1,0].set_yscale('log')
    ax[1,1].plot(Mdot0, Mdot/Mdot0, 'k+')
    ax[1,1].set_xlabel(r'$\dot{M}_0$')
    ax[1,1].set_ylabel(r'$\dot{M} / \dot{M}_0$')
    ax[1,1].set_xscale('log')
    ax[1,1].set_yscale('log')

    fig.tight_layout()
    fig.savefig("plot_minidisc_accretion_time.png")

if __name__ == "__main__":

    nargs = len(sys.argv)-1
    if nargs%2 != 0 or nargs == 0:
        print("I need pairs of minidisc_analysis.dat and mdot.dat files.")
        sys.exit()

    names = zip(sys.argv[1::2], sys.argv[2::2])

    accretionTimePlot(names)

