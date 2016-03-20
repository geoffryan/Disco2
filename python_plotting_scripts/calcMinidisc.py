import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import discopy as dp
import discoEOS as eos
import plot_thin as pt
import plot_disc_equat as pde

if __name__ == "__main__":

    M = 1.0e1
    a = 100 * M
    q = 1.0
    rs = 6*M
    rmin = rs*1.1
    rmax = 10000*rs
    Mdot = 2.0e0
    N = 200

    alpha = 0.1
    gam = 5.0/3.0
    EOStype = 0
    EOSPar1 = 1.0
    EOSPar2 = 0.0

    M1 = M/(1+q)
    M2 = M/(1+1/q)

    binPars = {"GravM": M2, "BinM": M1, "BinA": a, 
                "BinW": math.sqrt(M/(a*a*a))}

    print("Mtot: {0:g} M1: {1:g} M2: {2:g} q: {3:g}".format(M, M1, M2, q))
    rocheData = pde.calcRoche(binPars, (-1,1), (-1,1), N=4)

    pars = {"GravM": M,
            "GravA": 0.0,
            "Adiabatic_Index": gam,
            "AlphaVisc": alpha,
            "EOSType": EOStype,
            "EOSPar1": EOSPar1,
            "EOSPar2": EOSPar2,
            "NumR": 100,
            "NumZ": 1,
            "ng": 2,
            "NUM_N": 0,
            "NoInnerBC": 0,
            "R_Min": rmin,
            "R_Max": rmax,
            "Z_Min": -1.0,
            "Z_Max": 1.0,
            "RLogScale": 0.001,
            "ZLogScale": 1000.0,
            "HiResSigma": 1000.0,
            "HiResR0": 1000000.0,
            "HiResFac": 0.0,
            "NP_CONST": -1,
            "aspect": 1.0
            }

    g = dp.Grid(pars=pars)
    r = np.logspace(np.log10(rmin), np.log10(rmax), num=N, base=10.0)

    datSS, datNT = pt.calcNT(g, r, rs, (Mdot*eos.M_solar/eos.year)
                            / (eos.rho_scale*eos.rg_solar*eos.rg_solar*eos.c))
    sigSS = datSS[0] * eos.rho_scale * eos.rg_solar
    piSS = datSS[1] * eos.rho_scale * eos.rg_solar * eos.c * eos.c
    vrSS = datSS[2] * eos.c
    vpSS = datSS[3] * eos.c/eos.rg_solar
    QdotSS = datSS[4] * eos.rho_scale * eos.c*eos.c * eos.c
    machSS = np.sqrt((vrSS*vrSS+r*r*vpSS*vpSS*eos.rg_solar*eos.rg_solar)
                        / (gam*piSS/sigSS))
    TSS = piSS/sigSS / (eos.c*eos.c)

    sigNT = datNT[0] * eos.rho_scale * eos.rg_solar
    piNT = datNT[1] * eos.rho_scale * eos.rg_solar * eos.c * eos.c
    vrNT = datNT[2] * eos.c
    vpNT = datNT[3] * eos.c/eos.rg_solar
    QdotNT = datNT[4] * eos.rho_scale * eos.c*eos.c * eos.c
    machNT = np.sqrt((vrNT*vrNT+r*r*vpNT*vpNT*eos.rg_solar*eos.rg_solar)
                        / (gam*piNT/sigNT))
    TNT = piNT/sigNT / (eos.c*eos.c)

    TKSS = TSS * eos.mp*eos.c*eos.c / eos.kb
    TKNT = TNT * eos.mp*eos.c*eos.c / eos.kb

    q0SS = QdotSS / (piSS * vpSS)
    q0NT = QdotNT / (piNT * vpNT * np.power(1-3*M/r,-0.5))

    ia = len(r[r<a])
    w = math.sqrt(M/(a*a*a))
    print "Mtot={0:g} a={1:g} --> w={2:g} (c/rg_solar)".format(M, a, w)
    print "pi/sig: SS {0:.12g} NT {1:.12g}".format(TSS[ia],TNT[ia])

    fig, ax = plt.subplots(3, 3, figsize=(14,9))

    ax[0,0].plot(r*eos.rg_solar, sigSS, 'k+')
    ax[0,0].plot(r*eos.rg_solar, sigNT, 'r+')
    ax[0,0].set_xlabel(r"$r$ ($cm$)")
    ax[0,0].set_ylabel(r"$\Sigma$ ($g/cm^2$)")
    ax[0,0].set_xscale("log")
    ax[0,0].set_yscale("log")
    ax[0,0].axvline(a*eos.rg_solar, lw=2, color='b')

    ax[0,1].plot(r*eos.rg_solar, piSS, 'k+')
    ax[0,1].plot(r*eos.rg_solar, piNT, 'r+')
    ax[0,1].set_xlabel(r"$r$ ($cm$)")
    ax[0,1].set_ylabel(r"$\Pi$ ($erg/cm^2$)")
    ax[0,1].set_xscale("log")
    ax[0,1].set_yscale("log")
    ax[0,1].axvline(a*eos.rg_solar, lw=2, color='b')

    ax[0,2].plot(r*eos.rg_solar, QdotSS, 'k+')
    ax[0,2].plot(r*eos.rg_solar, QdotNT, 'r+')
    ax[0,2].set_xlabel(r"$r$ ($cm$)")
    ax[0,2].set_ylabel(r"$\dot{Q}$ ($erg/cm^2s$)")
    ax[0,2].set_xscale("log")
    ax[0,2].set_yscale("log")
    ax[0,2].axvline(a*eos.rg_solar, lw=2, color='b')

    ax[1,0].plot(r*eos.rg_solar, vrSS, 'k+')
    ax[1,0].plot(r*eos.rg_solar, vrNT, 'r+')
    ax[1,0].set_xlabel(r"$r$ ($cm$)")
    ax[1,0].set_ylabel(r"$v^r$ ($cm/s$)")
    ax[1,0].set_xscale("log")
    ax[1,0].set_yscale("linear")
    ax[1,0].axvline(a*eos.rg_solar, lw=2, color='b')

    ax[1,1].plot(r*eos.rg_solar, vpSS, 'k+')
    ax[1,1].plot(r*eos.rg_solar, vpNT, 'r+')
    ax[1,1].set_xlabel(r"$r$ ($cm$)")
    ax[1,1].set_ylabel(r"$v^\phi$ ($rad/s$)")
    ax[1,1].set_xscale("log")
    ax[1,1].set_yscale("log")
    ax[1,1].axvline(a*eos.rg_solar, lw=2, color='b')

    ax[1,2].plot(r*eos.rg_solar, machSS, 'k+')
    ax[1,2].plot(r*eos.rg_solar, machNT, 'r+')
    ax[1,2].set_xlabel(r"$r$ ($cm$)")
    ax[1,2].set_ylabel(r"$\mathcal{M}$")
    ax[1,2].set_xscale("log")
    ax[1,2].set_yscale("log")
    ax[1,2].axvline(a*eos.rg_solar, lw=2, color='b')

    ax[2,0].plot(r*eos.rg_solar, TSS, 'k+')
    ax[2,0].plot(r*eos.rg_solar, TNT, 'r+')
    ax[2,0].set_xlabel(r"$r$ ($cm$)")
    ax[2,0].set_ylabel(r"$T$ ($m_p c^2$)")
    ax[2,0].set_xscale("log")
    ax[2,0].set_yscale("log")
    ax[2,0].axvline(a*eos.rg_solar, lw=2, color='b')

    ax[2,1].plot(r*eos.rg_solar, TKSS, 'k+')
    ax[2,1].plot(r*eos.rg_solar, TKNT, 'r+')
    ax[2,1].set_xlabel(r"$r$ ($cm$)")
    ax[2,1].set_ylabel(r"$T$ ($K$)")
    ax[2,1].set_xscale("log")
    ax[2,1].set_yscale("log")
    ax[2,1].axvline(a*eos.rg_solar, lw=2, color='b')

    ax[2,2].plot(r*eos.rg_solar, q0SS, 'k+')
    ax[2,2].plot(r*eos.rg_solar, q0NT, 'r+')
    ax[2,2].set_xlabel(r"$r$ ($cm$)")
    ax[2,2].set_ylabel(r"$q_0 = \dot{Q}/\Pi v^\phi$ ()")
    ax[2,2].set_xscale("log")
    ax[2,2].set_yscale("log")
    ax[2,2].axvline(a*eos.rg_solar, lw=2, color='b')

    plt.tight_layout()

    plt.show()






