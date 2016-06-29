import sys
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt

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
markers = ['o', '+', '^', 'x', 'v']
linestyles = ['-', '--', ':', '-.']


def get_data(filename):

    f = open(filename, "r")
    data = pickle.load(f)
    f.close()

    return data

def calc_u0(R, vr, vp, M, bw):

    u0 = 1.0 / np.sqrt(1.0 - 2*M/R - 4*M/R*vr - (1+2*M/R)*vr*vr
                        - R*R*(bw+vp)*(bw+vp))

    return u0

def add_disp_plot(ax, data, marker, mc, ls, lc, mode='ABC'):
    
    R = data['R']
    sig = data['sig']
    pi = data['pi']
    vr = data['vr']
    vp = data['vp']
    gam = data['gam']
    M = data['M']
    bw = data['bw']
    phiSA = data['phiSa']
    phiSB = data['phiSb']
    
    cs2 = gam*pi/sig

    alpha = 1.0/np.sqrt(1+2*M/R)
    u0 = calc_u0(R, vr, vp, M, bw)

    w = alpha*u0
    u2 = w*w-1.0
    v2 = 1.0 - 1.0/(w*w)

    relMach = np.sqrt(u2*(1-cs2) / cs2)

    indA = data['iSa'][:,0] >= 0
    indB = data['iSb'][:,0] >= 0

    RA = R[indA]
    RB = R[indB]

    phiA = 0.5*(phiSA[indA,0]+phiSA[indA,1])
    phiB = 0.5*(phiSB[indB,0]+phiSB[indB,1])
    dpdrA = (phiA[2:] - phiA[:-2]) / (RA[2:] - RA[:-2])
    dpdrB = (phiB[2:] - phiB[:-2]) / (RB[2:] - RB[:-2])

    tpA = -1.0 / (dpdrA * RA[1:-1])
    tpB = -1.0 / (dpdrB * RB[1:-1])
    tpWKBrel = 1.0 / (relMach * np.sqrt((1-2*M/R)*((1-bw/vp)*(1-bw/vp)
                        - 0.25*(1-6*M/R)/(1-2*M/R))))

    if marker == '+':
        mew=2
    else:
        mew=0
    
    if 'A' in mode:
        ax.plot((relMach[indA])[1:-1], tpA, marker=marker, color=mc, ls='', 
                ms=10, mew=mew, mec=mc, alpha=0.5)
    if 'B' in mode:
        ax.plot((relMach[indB])[1:-1], tpB, marker=marker, color=mc, ls='', 
                ms=10, mew=mew, mec=mc, alpha=0.5)
    if 'C' in mode:
        ax.plot(relMach, tpWKBrel, ls=ls, color=lc, lw=3, alpha=1.0)

def disp_plot(datas, name, mode='ABC'):

    fig, ax = plt.subplots(1,1)

    for i,data in enumerate(datas):
        add_disp_plot(ax, data, markers[i], colors[i], '', 
                        'k', mode)

    if 'C' in mode:
        for i,data in enumerate(datas):
            add_disp_plot(ax, data, markers[i], colors[i], linestyles[i], 
                            'k', 'C')

    ax.set_xlabel(r'$\mathcal{M}$', fontsize=24)
    ax.set_ylabel(r'$\tan \theta_S$', fontsize=24)
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ylim = ax.get_ylim()
    ax.set_ylim(0,0.5)
    
    print("Saving " + name + "...")
    fig.savefig(name)
    plt.close(fig)

def add_diss_plot(ax, data, marker, color):

    R = data['R']
    psiqa = data['psiQa']
    psiqb = data['psiQb']
    dQdra = data['dQdra']
    dQdrb = data['dQdrb']
    dCool = data['dQdra']

    psiq = psiqa + psiqb
    dqdr = dQdra + dQdrb

    if marker == '+':
        mew = 2
    else:
        mew = 0

    ax[0].plot(R, psiq, marker=marker, color=color, mew=mew, ls='')
    ax[1].plot(R, dqdr, marker=marker, color=color, mew=mew, ls='')
    ax[1].plot(R, dCool, marker=marker, color=color, mew=mew, ls='')

    return R.min(), R.max()

def ceilSig(x):
    sig = int(math.floor(math.log10(abs(x))))
    fac = math.pow(10.0,sig)
    return fac * math.ceil(x/fac)

def floorSig(x):
    sig = int(math.floor(math.log10(abs(x))))
    fac = math.pow(10.0,sig)
    return fac * math.floor(x/fac)

def diss_plot(datas, name):

    fig, ax = plt.subplots(2,1, sharex=True)

    Rmin = np.inf
    Rmax = -np.inf
    for i, data in enumerate(datas):
        Rmini, Rmaxi = add_diss_plot(ax, data, markers[i], colors[i])
        Rmin = min(Rmin, Rmini)
        Rmax = max(Rmax, Rmaxi)
        
    ax[0].set_xlim(floorSig(Rmin), ceilSig(Rmax))
    ax[1].set_xlim(floorSig(Rmin), ceilSig(Rmax))
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$r$ ($M$)', fontsize=24)
    ax[0].set_ylabel(r'$\psi_Q$', fontsize=24)
    ax[1].set_ylabel(r'$\langle \dot{Q} \rangle$', fontsize=24)

    #ax[0].set_xticklabels([])
    #yticklabels = ax[1].get_yticklabels()
    #print yticklabels
    #ax[1].set_yticklabels()
    #fig.subplots_adjust(hspace=0.0, wspace=0.0)

    print("Saving " + name + "...")
    fig.savefig(name)
    plt.close(fig)

def torque_plot_single(data, name):

    R = data['R']
    Tmdot = data['Tmdot']
    Tre = data['Tre']
    Text = data['Text']
    Tcool = data['Tcool']
    Tpsiq = data['Tpsiq']
    qdot = data['dQdra'] + data['dQdrb']
    vp = data['vp']
    vr = data['vr']
    M = data['M']
    bw = data['bw']
    
    u0 = calc_u0(R, vr, vp, M, bw)
    up = u0*vp

    fig, ax = plt.subplots(1,1)
    ax.plot(R, Tmdot, color='k', marker='+', mew=2, ms=10, ls='',
            label=r'$\tau_{\dot{M}}$')
    ax.plot(R, -Tre, color=blue, marker='+', mew=2, ms=10, ls='',
            label=r'$-\tau_{Re}$')
    ax.plot(R, -(Tre+Text+Tcool), color=green, marker='+', mew=2, ms=10, ls='',
            label=r'$-\tau_{Re}-\tau_{ext}-\tau_{cool}$')
    ylim = ax.get_ylim()
    ax.plot(R, -Tpsiq, color=orange, marker='+', mew=2, ms=10, ls='',
            label=r'$-\tau_{loc}$')
    ax.plot(R, qdot/vp, color=red, marker='+', mew=2, ms=10, ls='',
            label=r'$-\tau_{glob}$')

    ax.set_xlim(floorSig(R.min()), ceilSig(R.max()))
    ax.set_ylim(0,ylim[1])
    ax.set_xscale('log')
    ax.set_xlabel(r'$r$ $(M)$', fontsize=24)
    ax.set_ylabel(r'$\tau$', fontsize=24)

    legend = ax.legend()

    print("Saving " + name + "...")
    fig.savefig(name)
    plt.close(fig)

    fig, ax = plt.subplots(1,1)
    ax.plot(R, np.fabs((qdot/vp + Tre) / Tre), 'k+')
    ax.set_xscale('log')
    ax.set_ylim([0,1])
    ax.set_xlim(floorSig(R.min()), ceilSig(R.max()))
    fig.savefig("torqfrac.pdf")
    

def plot_data(data):

    R = data['R']
    sig = data['sig']
    pi = data['pi']
    vr = data['vr']
    vp = data['vp']
    gam = data['gam']
    M = data['M']
    bw = data['bw']
    phiSA = data['phiSa']
    phiSB = data['phiSb']
    sigA = data['siga'][:,0]
    piA = data['pia'][:,0]
    vrA = data['vra'][:,0]
    vpA = data['vpa'][:,0]
    sigB = data['sigb'][:,0]
    piB = data['pib'][:,0]
    vrB = data['vrb'][:,0]
    vpB = data['vpb'][:,0]

    cs2 = gam*pi/sig

    alpha = 1.0/np.sqrt(1+2*M/R)
    u0 = 1.0 / np.sqrt(1.0 - 2*M/R - 4*M/R*vr - (1+2*M/R)*vr*vr
                        - R*R*(bw+vp)*(bw+vp))

    w = alpha*u0
    u2 = w*w-1.0
    v2 = 1.0 - 1.0/(w*w)

    newtMach = np.sqrt(v2 / cs2)
    relMach = np.sqrt(u2*(1-cs2) / cs2)

    indA = data['iSa'][:,0] >= 0
    indB = data['iSb'][:,0] >= 0

    RA = R[indA]
    RB = R[indB]

    phiA = 0.5*(phiSA[indA,0]+phiSA[indA,1])
    phiB = 0.5*(phiSB[indB,0]+phiSB[indB,1])
    dpdrA = (phiA[2:] - phiA[:-2]) / (RA[2:] - RA[:-2])
    dpdrB = (phiB[2:] - phiB[:-2]) / (RB[2:] - RB[:-2])

    tpA = -1.0 / (dpdrA * RA[1:-1])
    tpB = -1.0 / (dpdrB * RB[1:-1])
    tpWKBrel = 1.0 / (relMach * np.sqrt((1-2*M/R)*((1-bw/vp)*(1-bw/vp)
                        - 0.25*(1-6*M/R)/(1-2*M/R))))
    tpWKBrelNoCorr = 1.0 / (relMach * np.sqrt((1-bw/vp)*(1-bw/vp)-0.25))
    tpWKBnewt = 1.0 / (newtMach * np.sqrt((1-bw/vp)*(1-bw/vp)-0.25))

    fig, ax = plt.subplots(1,1)
    ax.plot((relMach[indA])[1:-1], tpA, marker='+', ls='', mew=2, ms=10, 
            color=blue)
    ax.plot((relMach[indB])[1:-1], tpB, marker='+', ls='', mew=2, ms=10, 
            color=orange)
    ax.plot(relMach, tpWKBrel, ls='-', lw=2, color='grey')
    ax.plot(relMach, tpWKBrelNoCorr, ls='-', lw=2, color=green)

    fig.savefig("plot_minidisc_data_disp.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(1,1)
    ax.plot((newtMach[indA])[1:-1], tpA, marker='+', ls='', mew=2, ms=10, color=blue)
    ax.plot((newtMach[indB])[1:-1], tpB, marker='+', ls='', mew=2, ms=10, color=orange)
    ax.plot(newtMach, tpWKBnewt, ls='-', lw=2, color='grey')

    fig.savefig("plot_minidisc_data_disp_newt.pdf")
    plt.close(fig)


if __name__ == "__main__":

    #for fname in sys.argv[1:]:
    #    data = get_data(fname)
    #    plot_data(data)

    datas = []
    for filename in sys.argv[1:]:
        datas.append(get_data(filename))

    disp_plot(datas, "disp_plot_AC.pdf", 'AC')
    disp_plot(datas, "disp_plot_BC.pdf", 'BC')
    disp_plot(datas, "disp_plot_ABC.pdf", 'ABC')
    diss_plot(datas, "diss_plot.pdf")

    torque_plot_single(datas[0], "torque_plot_single.pdf")
