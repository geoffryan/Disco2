import sys
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import discoGR as gr
import discoEOS as eos

labelsize = 24

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
    lapse = 1.0/np.sqrt(1+2*M/R)
    pars = {'Metric': 6,
            'GravM': M,
            'GravA': 0.0,
            'BoostType': 1,
            'BinW': bw,
            'BinA': data['ba'],
            'BinM': data['M2']}
            
    u0, ur, up = gr.calc_u(R, vr, vp, pars)

    w = lapse*u0
    u2 = w*w-1.0
    v2 = 1.0 - 1.0/(w*w)

    relMach = np.sqrt(u2*(1-cs2) / cs2)

    indA = data['iSa'][:,0] >= 0
    indB = data['iSb'][:,0] >= 0

    RA = R[indA]
    RB = R[indB]

    phiA = 0.5*(phiSA[indA,0]+phiSA[indA,1])
    phiB = 0.5*(phiSB[indB,0]+phiSB[indB,1])
    diffA = phiA[2:] - phiA[:-2]
    diffA[diffA>np.pi] -= 2*np.pi
    diffA[diffA<-np.pi] += 2*np.pi
    diffB = phiB[2:] - phiB[:-2]
    diffB[diffB>np.pi] -= 2*np.pi
    diffB[diffB<-np.pi] += 2*np.pi
    dpdrA = diffA / (RA[2:] - RA[:-2])
    dpdrB = diffB / (RB[2:] - RB[:-2])

    tpA = -1.0 / (dpdrA * RA[1:-1])
    tpB = -1.0 / (dpdrB * RB[1:-1])
    tpWKBrel = 1.0 / (relMach * np.sqrt((1-2*M/R)*((1-bw/(vp+bw))*(1-bw/(vp+bw))
                        - 0.25*(1-6*M/R))))

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

    ax.set_xlabel(r'$\mathcal{M}$', fontsize=labelsize)
    ax.set_ylabel(r'$\tan \theta_S$', fontsize=labelsize)
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
    ax[1].set_xlabel(r'$r$ ($M$)', fontsize=labelsize)
    ax[0].set_ylabel(r'$\psi_Q$', fontsize=labelsize)
    ax[1].set_ylabel(r'$\langle \dot{Q} \rangle$', fontsize=labelsize)

    #ax[0].set_xticklabels([])
    #yticklabels = ax[1].get_yticklabels()
    #print yticklabels
    #ax[1].set_yticklabels()
    #fig.subplots_adjust(hspace=0.0, wspace=0.0)

    print("Saving " + name + "...")
    fig.savefig(name)
    plt.close(fig)

def diss_plot_single(data, name):

    fig, ax = plt.subplots(3,1, sharex=True, figsize=(9,9))

    R = data['R']
    Rmin = R.min()
    Rmax = R.max()
    
    gam = data['gam']
    siga = data['siga']
    pia = data['pia']
    sigb = data['sigb']
    pib = data['pib']

    sa = np.log(pia * np.power(siga, -gam)) / (gam-1.0)
    sb = np.log(pib * np.power(sigb, -gam)) / (gam-1.0)

    dsa = sa[:,1] - sa[:,0]
    dsb = sb[:,1] - sb[:,0]

    psiqa = data['psiQa']
    psiqb = data['psiQb']
    dQdra = data['dQdra']
    dQdrb = data['dQdrb']
    dCool = data['dCool']

    M = data['M']
    Mdot0_cgs = data['Mdot'] * eos.M_solar / eos.year
    Mdot0 = Mdot0_cgs / (eos.rho_scale * eos.rg_solar**2 * eos.c)

    ri = 6*M
    C = 6.0e-2
    Pfunc = 1 - np.sqrt(ri/R) + np.sqrt(3*M/R)*(
            np.arctanh(np.sqrt(3*M/R)) - np.arctanh(np.sqrt(3*M/ri)))
    QNTr = 3*data['Jr'] / (4*np.pi) * (M/(R*R*R)) / (1-3*M/R) * Pfunc
    QNT = 3*(0.75*Mdot0) / (4*np.pi) * (M/(R*R*R)) / (1-3*M/R) * Pfunc
    QNTc = 3*(0.75*Mdot0) / (4*np.pi) * (M/(R*R*R)) / (1-3*M/R) * (Pfunc
            + C/np.sqrt(R))

    ds = dsa+dsb
    psiq = psiqa + psiqb
    dqdr = dQdra + dQdrb

    marker = '+'
    mew = 2

    ax[0].plot(R, dsa, marker=marker, color=blue, mew=mew, ls='')
    ax[0].plot(R, dsb, marker=marker, color=orange, mew=mew, ls='')
    ax[1].plot(R, psiqa, marker=marker, color=blue, mew=mew, ls='')
    ax[1].plot(R, psiqb, marker=marker, color=orange, mew=mew, ls='')
    ax[2].plot(R, dQdra, marker=marker, color=blue, mew=mew, ls='', 
                label=r'$\dot{Q}_{irr,A}$')
    ax[2].plot(R, dQdrb, marker=marker, color=orange, mew=mew, ls='',
                label=r'$\dot{Q}_{irr,B}$')
    ax[2].plot(R, dqdr, marker=marker, color='k', mew=mew, ls='',
                label=r'$\langle \dot{Q}_{irr} \rangle$')
    ax[2].plot(R, dCool, marker=marker, color=green, mew=mew, ls='',
                label=r'$\langle \dot{Q}_{cool} \rangle$')
    ax[2].plot(R, 2*np.pi*R*QNT, marker='', color='grey', ls='-', lw=4,
                label=r'$\langle \dot{Q}_{NT} \rangle$', alpha=0.75)
    #ax[2].plot(R, 2*np.pi*R*QNTc, marker='', color='grey', ls='--', lw=4,
    #            label=r'$\langle \dot{Q}_{NT} \rangle$', alpha=0.75)
    #ax[2].plot(R, QNT, marker='', color='grey', ls='-', lw=2,
    #            label=r'$\langle \dot{Q}_{NT} \rangle$')
    #ax[2].plot(R, QNTr, marker='', color='grey', ls='-.', lw=2,
    #            label=r'$\langle \dot{Q}_{NT} \rangle$')
    #ax[2].plot(R, 2*np.pi*R*QNTr, marker='', color='grey', ls=':', lw=2,
    #            label=r'$\langle \dot{Q}_{NT} \rangle$')

    ax[0].set_xlim(floorSig(Rmin), ceilSig(Rmax))
    ax[1].set_xlim(floorSig(Rmin), ceilSig(Rmax))
    ax[2].set_xlim(floorSig(Rmin), ceilSig(Rmax))
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel(r'$r$ ($M$)', fontsize=labelsize)
    ax[0].set_ylabel(r'$\Delta s$', fontsize=labelsize)
    ax[1].set_ylabel(r'$\psi_Q$', fontsize=labelsize)
    ax[2].set_ylabel(r'$\langle \dot{Q} \rangle$', fontsize=labelsize)

    legend = ax[2].legend()

    print("Saving " + name + "...")
    fig.savefig(name)
    plt.close(fig)

def dissCorrPlot(data, name):
#(r, sig, pi, vr, vp, u0, shockDissDat, pars, name):

    R = data['R']
    M = data['M']
    bw = data['bw']

    pars = {'Metric': 6,
            'GravM': M,
            'GravA': 0.0,
            'BoostType': 1,
            'BinW': bw,
            'BinA': data['ba'],
            'BinM': data['M2']}

    phiSa = data['phiSa']
    phiSb = data['phiSb']
    psiqa = data['psiQa']
    psiqb= data['psiQb']
    dqdra = data['dQdra']
    dqdrb = data['dQdrb']

    sigA = data['siga'][:,0]
    vrA = data['vra'][:,0]
    vpA = data['vpa'][:,0]
    sigB = data['sigb'][:,0]
    vrB = data['vrb'][:,0]
    vpB = data['vpb'][:,0]

    dpdra = np.zeros(Rs.shape)
    dpdrb = np.zeros(Rs.shape)
    dpdra[1:-1] = (phiSa[2:,0]-phiSa[:-2,0]) / (R[2:] - R[:-2])
    dpdrb[1:-1] = (phiSb[2:,0]-phiSb[:-2,0]) / (R[2:] - R[:-2])
    
    g00, g0r, g0p, grr, grp, gpp = gr.calc_g(R, pars)
    pars = {'Metric': 6,
            'GravM': M,
            'GravA': 0.0,
            'BoostType': 1,
            'BinW': bw,
            'BinA': data['ba'],
            'BinM': data['M2']}    
    u0A, urA, upA = gr.calc_u(R, vrA, vpA, pars)
    u0B, urB, upB = gr.calc_u(R, vrB, vpB, pars)

    dgam = grr*gpp-grp*grp

    igamrr = gpp/dgam
    igamrp = -grp/dgam
    igampp = grr/dgam
    br = igamrr*g0r + igamrp*g0p
    bp = igamrp*grp + igampp*gpp
    b2 = br*g0r + bp*g0p
    a2 = b2 - g00

    igrr = igamrr - br*br/a2
    igrp = igamrp - br*bp/a2
    igpp = igampp - bp*bp/a2

    jaca = np.sqrt(-g00*(grr+2*dpdra*grp+dpdra*dpdra*gpp)
                    + (g0r+dpdra*g0p)*(g0r+dpdra*g0p))
    jacb = np.sqrt(-g00*(grr+2*dpdrb*grp+dpdra*dpdrb*gpp)
                    + (g0r+dpdrb*g0p)*(g0r+dpdrb*g0p))

    norma = np.sqrt(igrr*dpdra*dpdra + 2*igrp*dpdra + igpp)
    normb = np.sqrt(igrr*dpdrb*dpdrb + 2*igrp*dpdrb + igpp)

    fluxa = np.zeros(R.shape)
    fluxb = np.zeros(R.shape)
    
    fluxa = jaca * sigA*u0A * (-vrA*dpdra+vpA)/norma
    fluxb = jacb * sigB*u0B * (-vrB*dpdrb+vpB)/normb

    dQdrCorr = np.zeros((Rs.shape[0], 2))
    dQdrCorr[:,0] = dQdm[:,0]*fluxa
    dQdrCorr[:,1] = dQdm[:,1]*fluxb

    fig, ax = plt.subplots(1,1)
    ax.plot(Rs, dQdr[:,0], 'k+')
    ax.plot(Rs, dQdr[:,1], 'g+')
    ax.plot(Rs, dQdrCorr[:,0], 'b+')
    ax.plot(Rs, dQdrCorr[:,1], 'r+')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    figname = "plot_minidisc_dissCorr_{0}.png".format(name)
    print("Saving {0:s}...".format(figname))
    fig.savefig(figname)
    plt.close(fig)

def torque_plot_single(data, name):

    R = data['R']
    Tmdot = data['Tmdot']
    Tre = data['Tre']
    Text = data['Text']
    Tcool = data['Tcool']
    Tpsiq = data['Tpsiq']
    qdot = data['dQdra'] + data['dQdrb']
    l = data['l']
    Jr = -data['Jr']
    qdot = data['dQdra'] + data['dQdrb']
    qdot = data['dQdra'] + data['dQdrb']
    vp = data['vp']
    vr = data['vr']
    M = data['M']
    bw = data['bw']

    pars = {'Metric': 6,
            'GravM': M,
            'GravA': 0.0,
            'BoostType': 1,
            'BinW': bw,
            'BinA': data['ba'],
            'BinM': data['M2']}
    
    u0, ur, up = gr.calc_u(R, vr, vp, pars)

    Tgrad = np.zeros(R.shape)
    dJrdR = (Jr[2:] - Jr[:-2]) / (R[2:] - R[:-2])
    Tgrad[1:-1] = l[1:-1] * dJrdR

    fig, ax = plt.subplots(1,1)
    ax.plot(R, Tmdot, color='k', marker='+', mew=2, ms=10, ls='',
            label=r'$\tau_{\dot{M}}$')
    ax.plot(R, -Tre, color=blue, marker='+', mew=2, ms=10, ls='',
            label=r'$-\tau_{Re}$')
    ax.plot(R, -(Tre+Text+Tcool), color=green, marker='+', mew=2, ms=10, ls='',
            label=r'$-\tau_{Re}-\tau_{ext}-\tau_{cool}$')
    #ax.plot(R, -(Tre+Text+Tcool+Tgrad), color=purple, marker='+', mew=2, ms=10,
    #        ls='', label=r'$-\tau_{Re}-\tau_{ext}-\tau_{cool} - \tau_{grad}$')
    ylim = ax.get_ylim()
    ax.plot(R, -Tpsiq, color=orange, marker='+', mew=2, ms=10, ls='',
            label=r'$-\tau_{loc}$')
    ax.plot(R, qdot/vp, color=red, marker='+', mew=2, ms=10, ls='',
            label=r'$-\tau_{glob}$')

    ax.set_xlim(floorSig(R.min()), ceilSig(R.max()))
    ax.set_ylim(0,ylim[1])
    ax.set_xscale('log')
    ax.set_xlabel(r'$r$ $(M)$', fontsize=labelsize)
    ax.set_ylabel(r'$\tau$', fontsize=labelsize)

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

def plot_decomTest(data, name):

    R = data['R']
    M = data['M']
    l = data['l']
    vp = data['vp']
    sig = data['sig']
    pi = data['pi']

    vp_kep_newt = np.sqrt(M/(R*R*R))
    vp_kep_rel = np.sqrt(M/(R*R*R))
    l_kep_newt = np.sqrt(M*R)
    l_kep_rel = np.sqrt(M*R) / np.sqrt(1-3*M/R)

    fig, ax = plt.subplots(2,2,sharex=True)
    ax[0,0].plot(R, vp, ls='', color=blue, marker='+')
    ax[0,0].plot(R, vp_kep_rel, ls='-', color=orange)
    ax[0,1].plot(R, (vp-vp_kep_rel) / vp_kep_rel, ls='', color=blue, 
                        marker='+')
    ax[0,1].plot(R, (vp_kep_rel-vp_kep_rel) / vp_kep_rel, ls='-', color=orange)

    ax[1,0].plot(R, l, ls='', color=blue, marker='+')
    ax[1,0].plot(R, l_kep_rel, ls='-', color=orange)
    ax[1,1].plot(R, (l-l_kep_rel) / l_kep_rel, ls='', color=blue, 
                        marker='+')
    ax[1,1].plot(R, (l_kep_rel-l_kep_rel) / l_kep_rel, ls='-', color=orange)

    ax[0,0].set_xscale('log')
    ax[0,1].set_xscale('log')
    ax[1,0].set_xscale('log')
    ax[1,1].set_xscale('log')
    ax[0,0].set_yscale('log')
    ax[1,0].set_yscale('log')
    ax[0,0].set_xlim(floorSig(R.min()), ceilSig(R.max()))
    ax[0,1].set_xlim(floorSig(R.min()), ceilSig(R.max()))
    ax[1,0].set_xlim(floorSig(R.min()), ceilSig(R.max()))
    ax[1,1].set_xlim(floorSig(R.min()), ceilSig(R.max()))

    print("Saving " + name + "...")
    fig.savefig(name)
    plt.close(fig)

def plot_mdot_single(data, name):

    R = data['R']
    Mdot = data['Jr']
    Mdot0_cgs = data['Mdot'] * eos.M_solar / eos.year
    Mdot0 = Mdot0_cgs / (eos.rho_scale * eos.rg_solar**2 * eos.c)

    fig, ax = plt.subplots(2,1)
    ax[0].plot(R, Mdot, '+', color=blue)
    ax[1].plot(R, Mdot/Mdot0, '+', color=blue)

    ax[0].set_xscale('log')
    ax[1].set_xscale('log')

    print("Saving " + name + "...")
    fig.savefig(name)
    plt.close(fig)


def plot_mdot(datas, name):

    N = len(datas)
    if N == 1:
        return
    data0 = datas[0]
    bw = data0['bw']
    Torb = 2*math.pi / bw
    t = np.zeros(N)
    Mdot = np.zeros((N, data0['R'].shape[0]))

    for i, data in enumerate(datas):
        t[i] = data['T'] / Torb
        Mdot[i,:] = data['Jr']

    fig, ax = plt.subplots(1,1)
    ax.plot(t, Mdot[:,2], color=blue, label=r'$\dot{M} (r=r_{in}))$')
    ax.plot(t, Mdot[:,-3], color=orange, label=r'$\dot{M} (r=r_{out}))$')
    
    ax.set_xlabel(r'$t$ $(T_{orb})$', fontsize=labelsize)
    ax.set_ylabel(r'$\dot{M}$', fontsize=labelsize)
    ax.set_xscale('linear')
    ax.set_yscale('log')
    legend = ax.legend()

    print("Saving " + name + "...")
    fig.savefig(name)
    plt.close(fig)


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

    doMdot = False
    doDisp = False
    doDiss = False
    doTorq = False
    doTest = False

    if "mdot" in sys.argv:
        doMdot = True
        sys.argv.remove("mdot")
    if "diss" in sys.argv:
        doDiss = True
        sys.argv.remove("diss")
    if "disp" in sys.argv:
        doDisp = True
        sys.argv.remove("disp")
    if "torq" in sys.argv:
        doTorq = True
        sys.argv.remove("torq")
    if "test" in sys.argv:
        doTest = True
        sys.argv.remove("test")

    datas = []
    for filename in sys.argv[1:]:
        datas.append(get_data(filename))
    
    if doDisp:
        disp_plot(datas, "disp_plot_AC.pdf", 'AC')
        disp_plot(datas, "disp_plot_BC.pdf", 'BC')
        disp_plot(datas, "disp_plot_ABC.pdf", 'ABC')

    if doDiss:
        diss_plot(datas, "diss_plot.pdf")
        diss_plot_single(datas[0], "diss_plot_single.pdf")

    if doTorq:
        torque_plot_single(datas[0], "torque_plot_single.pdf")

    if doMdot:
        plot_mdot(datas, "mdot_plot.pdf")
        plot_mdot_single(datas[0], "mdot_plot_single.pdf")

    if doTest:
        plot_decomTest(datas[0], "test_plot.pdf")
