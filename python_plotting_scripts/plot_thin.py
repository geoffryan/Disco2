import sys
import math
import discopy as dp
import numpy as np
import matplotlib.pyplot as plt

# All constants in c.g.s.
sb = 1.56055371e59
c = 2.99792458e10
mp = 1.672621777e-24
h = 6.62606957e-27
ka_bbes = 0.2
rg_solar = 1.4766250385e5
r_scale = rg_solar
rho_scale = 1.0

nu = np.logspace(14,20,num=7,base=10.0)

def gToArr(g):
    R = np.zeros(g.np.sum())
    prim = np.zeros((g.np.sum(),5))

    ind = 0
    for k in xrange(g.nz_tot):
        for i in xrange(g.nr_tot):
            r1 = g.rFaces[i]
            r2 = g.rFaces[i+1]
            r = 0.5*(r1+r2)

            numP = g.np[k,i]

            R[ind:ind+numP] = r
            prim[ind:ind+numP,:] = g.prim[k][i][:,:5]

            ind += numP

    return R, prim

def calcNT(g, r, rs, Mdot):

    M = g._pars['GravM']
    GAM = g._pars['Adiabatic_Index']
    AL = g._pars['AlphaVisc']

    C = 1.0 - 3.0*M/r
    D = 1.0 - 2.0*M/r
    P = 1.0 - np.sqrt(rs/r) + np.sqrt(3*M/r)*(np.arctanh(np.sqrt(3*M/r))
                                            - np.arctanh(np.sqrt(3*M/rs)))
    OMK = np.sqrt(M/(r*r*r))

    U0 = 1.0/np.sqrt(C)

    pi = Mdot / (3*np.pi*AL*math.sqrt(GAM)) * OMK
    PI = Mdot / (3*np.pi*AL*math.sqrt(GAM)) * OMK * np.sqrt(C) * P / (D*D)

    Qdot = 3*Mdot/(4*np.pi) * OMK*OMK
    QDOT = 3*Mdot/(4*np.pi) * OMK*OMK * P / C

    sig = np.power(8.0*sb/(3*ka_bbes*Qdot) * (pi*mp*c*c)**4
                    / (c*c*c*r_scale*rho_scale), 0.2)
    SIG = np.power(8.0*sb/(3*ka_bbes*QDOT) * (PI*mp*c*c)**4
                    / (c*c*c*r_scale*rho_scale), 0.2)

    vr = -Mdot / (2*np.pi*r*sig)
    VR = -Mdot / (2*np.pi*r*SIG * U0)

    vp = OMK
    VP = OMK

    return (sig, pi, vr, vp, Qdot), (SIG, PI, VR, VP, QDOT)

def calcGEO(g, r, Mdot, K):

    M = g._pars['GravM']
    GAM = g._pars['Adiabatic_Index']
    AL = g._pars['AlphaVisc']

    ur = -np.power(6*M/r - 1.0, 1.5) / 3.0
    up = 2*math.sqrt(3.0)*M/(r*r)
    u0 = (-2*M/r*ur - np.sqrt(4*M*M/(r*r)*ur*ur - (-1+2.0*M/r) * (1.0+(1.0+2.0*M/r)*ur*ur + r*r*up*up))) / (-1+2*M/r)

    vr = ur/u0
    vp = up/u0

    rho = -Mdot / (2*math.pi*r*ur)

    P = K * np.power(rho, GAM)

    return rho, P, vr, vp

def plotNT(g):

    r, prim = gToArr(g)

    M = g._pars['GravM']
    Risco = 6.0*M
    R = np.logspace(math.log10(Risco)+0.05, math.log10(r.max()), base=10.0, num=200)
    R2 = np.logspace(math.log10(r.min()), math.log10(Risco)-0.05, base=10.0, num=200)

    RS = 6.0*M
    MDOT = 182.3
    K = 2.5e-5

    SSdat, NTdat = calcNT(g, R, RS, MDOT)
    GEOdat = calcGEO(g, R2, MDOT, K)

    fig, ax = plt.subplots(2,3, figsize=(14,9))

    print r.shape
    print prim.shape
    print R.shape
    print SSdat[0].shape

    blue = (31.0/255, 119.0/255, 180.0/255)
    orange = (255.0/255, 127.0/255, 14.0/255)
    green = (44.0/255, 160.0/255, 44.0/255)

    xlim = (1.0, 1.0e3)

    ax[0,0].plot(r, prim[:,0], 'k+')
    ax[0,0].plot(R, SSdat[0], ls='-', lw=3.0, color=orange)
    ax[0,0].plot(R2, GEOdat[0], ls='-', lw=3.0, color=green)
    ax[0,0].plot(R, NTdat[0], ls='-', lw=3.0, color=blue)
    ax[0,0].set_xlabel(r"$r$")
    ax[0,0].set_ylabel(r"$\Sigma$")
    ax[0,0].set_xscale('log')
    ax[0,0].set_yscale('log')
    ax[0,0].set_xlim(xlim)
    
    ax[0,1].plot(r, prim[:,1], 'k+')
    ax[0,1].plot(R, SSdat[1], ls='-', lw=3.0, color=orange)
    ax[0,1].plot(R2, GEOdat[1], ls='-', lw=3.0, color=green)
    ax[0,1].plot(R, NTdat[1], ls='-', lw=3.0, color=blue)
    ax[0,1].set_xlabel(r"$r$")
    ax[0,1].set_ylabel(r"$\Pi$")
    ax[0,1].set_xscale('log')
    ax[0,1].set_yscale('log')
    ax[0,1].set_xlim(xlim)

    ax[1,0].plot(r, -prim[:,2], 'k+')
    ax[1,0].plot(R, -SSdat[2], ls='-', lw=3.0, color=orange)
    ax[1,0].plot(R2, -GEOdat[2], ls='-', lw=3.0, color=green)
    ax[1,0].plot(R, -NTdat[2], ls='-', lw=3.0, color=blue)
    ax[1,0].set_xlabel(r"$r$")
    ax[1,0].set_ylabel(r"$v^r$")
    ax[1,0].set_xscale('log')
    ax[1,0].set_yscale('log')
    ax[1,0].set_xlim(xlim)

    ax[1,1].plot(r, prim[:,3], 'k+')
    ax[1,1].plot(R, SSdat[3], ls='-', lw=3.0, color=orange)
    ax[1,1].plot(R2, GEOdat[3], ls='-', lw=3.0, color=green)
    ax[1,1].plot(R, NTdat[3], ls='-', lw=3.0, color=blue)
    ax[1,1].set_xlabel(r"$r$")
    ax[1,1].set_ylabel(r"$v^\phi$")
    ax[1,1].set_xscale('log')
    ax[1,1].set_yscale('log')
    ax[1,1].set_xlim(xlim)
    
    sig = prim[:,0]
    pi = prim[:,1]
    vr = prim[:,2]
    vp = prim[:,3]
    T = mp*c*c * pi/sig
    qdot = 8*sb * T*T*T*T / (3*ka_bbes*sig * c*c*c*rho_scale*r_scale)

    ax[0,2].plot(r, qdot, 'k+')
    ax[0,2].plot(R, SSdat[4], ls='-', lw=3.0, color=orange)
    ax[0,2].plot(R, NTdat[4], ls='-', lw=3.0, color=blue)
    ax[0,2].set_xlabel(r"$r$")
    ax[0,2].set_ylabel(r"$\dot{Q}$")
    ax[0,2].set_xscale('log')
    ax[0,2].set_yscale('log')
    ax[0,2].set_xlim(xlim)

    u0 = 1.0 / np.sqrt(1 - 2*M/r - 4*M/r*vr - (1+2*M/r)*vr*vr - r*r*vp*vp)
    mdot = -2*np.pi*r*sig*u0*vr

    ax[1,2].plot(r, mdot, 'k+')
    ax[1,2].plot(R, MDOT*np.ones(R.shape), ls='-', lw=3.0, color=orange)
    ax[1,2].plot(R, MDOT*np.ones(R.shape), ls='-', lw=3.0, color=blue)
    ax[1,2].set_xlabel(r"$r$")
    ax[1,2].set_ylabel(r"$\dot{M}$")
    ax[1,2].set_xscale('log')
    ax[1,2].set_yscale('log')
    ax[1,2].set_xlim(xlim)

    plt.tight_layout()

    fig.savefig("thin.png")

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Need a parfile and checkpoint, numbnuts!")

    par = dp.readParfile(sys.argv[1])
    g = dp.Grid(par)
    g.loadCheckpoint(sys.argv[2])

    plotNT(g)

    plt.show()


