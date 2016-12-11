import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import discopy as dp
import discoEOS as eos

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

colors = [blue, orange, green, red, purple, brown, pink, grey, yellow, teal]
lightcolors = [lightblue, lightorange, lightgreen, lightred, lightpurple,
                    lightbrown, lightpink, lightgrey, lightyellow, lightteal]

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

    fig, ax = plt.subplots(2,2,sharex=True)
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
    fig.savefig("plot_minidisc_accretion_time.pdf")

def accretionPlot(accretionDatNames):

    N = len(accretionDatNames)

    fig, ax =plt.subplots(1,1,figsize=(4,6))

    label = ['Model 1', 'Model 1.5', 'Model 2', 'Model 2.5', 'Model 3']

    mycolors = [blue, red, orange, purple, green]
    mylightcolors = [lightblue, lightred, lightorange, lightpurple, lightgreen]

    for i,name in enumerate(accretionDatNames):
        mdotDat = getPickleData(name)
        pars = mdotDat['pars']
        t = mdotDat['T']
        mdot = mdotDat['Mdot'][:,2]
        Mdot0 = pars['BoundPar2']
        Mdot0_cgs = Mdot0 * eos.M_solar / eos.year
        Mdot0_code = Mdot0_cgs / (eos.rho_scale * eos.rg_solar**2 * eos.c)
        mdot *= Mdot0_code

        T = np.linspace(0,30,100)
        ax.plot(t[t<=29], mdot[t<=29], label=label[i], color=mycolors[i])
        ax.plot(T, Mdot0_code*np.ones(T.shape), color=mylightcolors[i], 
                lw=2, ls='--', zorder=1)
        #mdot = mdot[(t>=25)].mean()

    ax.set_xlabel(r'$t (T_{bin})$', fontsize=16)
    ax.set_ylabel(r'$\dot{M} $', fontsize=16)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.1)

    ax.set_ylim(1.0e1, 1.0e4)

    fig.savefig("mdot_all.pdf")

def plotMdotSingle(name):

    fig, ax =plt.subplots(1,1,figsize=(4,3))

    #label = ['Model 1', 'Model 1.5', 'Model 2', 'Model 2.5', 'Model 3']

    mycolors = [blue, red, orange, purple, green]
    mylightcolors = [lightblue, lightred, lightorange, lightpurple, lightgreen]

    mdotDat = getPickleData(name)
    pars = mdotDat['pars']
    t = mdotDat['T']
    mdotIn = mdotDat['Mdot'][:,2]
    mdotOut = mdotDat['Mdot'][:,-3]

    ax.plot(t[t<=29], mdotOut[t<=29], color=orange, marker='', lw=2,
                label=r'$\dot{M}(r_{out})$')
    ax.plot(t[t<=29], mdotIn[t<=29], color=blue, marker='', lw=2,
                label=r'$\dot{M}(r_{in})$')

    ax.set_xlabel(r'$t (T_{bin})$', fontsize=16)
    ax.set_ylabel(r'$\dot{M} / \dot{M}_{nozzle} $', fontsize=16)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.17)

    ax.set_xlim(0.0, 30)
    ax.set_ylim(1.0e-2, 1.0e2)

    fig.savefig(".".join(name.split('.')[:-1]) + ".pdf")


if __name__ == "__main__":

    nargs = len(sys.argv)-1
    if nargs%2 != 0 or nargs == 0:
        print("I need pairs of minidisc_analysis.dat and mdot.dat files.")
        sys.exit()

    names = zip(sys.argv[1::2], sys.argv[2::2])

    accretionTimePlot(names)
    accretionPlot(sys.argv[2::2])
    for name in sys.argv[2::2]:
        plotMdotSingle(name)



