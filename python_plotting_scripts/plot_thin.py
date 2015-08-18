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

def plotNT(g):

    r, prim = gToArr(g)

    M = g._pars['GravM']
    Risco = 6.0*M
    R = np.logspace(math.log10(Risco), math.log10(r.max()), base=10.0, num=200)

    RS = 6.0*M
    MDOT = 17.5

    SSdat, NTdat = calcNT(g, R, RS, MDOT)

    fig, ax = plt.subplots(2,3, figsize=(12,9))

    print r.shape
    print prim.shape
    print R.shape
    print SSdat[0].shape

    xlim = (1.0, 1.0e3)

    ax[0,0].plot(r, prim[:,0], 'k+')
    ax[0,0].plot(R, SSdat[0], 'r')
    ax[0,0].plot(R, NTdat[0], 'b')
    ax[0,0].set_xlabel(r"$r$")
    ax[0,0].set_ylabel(r"$\Sigma$")
    ax[0,0].set_xscale('log')
    ax[0,0].set_yscale('log')
    ax[0,0].set_xlim(xlim)
    
    ax[0,1].plot(r, prim[:,1], 'k+')
    ax[0,1].plot(R, SSdat[1], 'r')
    ax[0,1].plot(R, NTdat[1], 'b')
    ax[0,1].set_xlabel(r"$r$")
    ax[0,1].set_ylabel(r"$\Pi$")
    ax[0,1].set_xscale('log')
    ax[0,1].set_yscale('log')
    ax[0,1].set_xlim(xlim)

    ax[1,0].plot(r, -prim[:,2], 'k+')
    ax[1,0].plot(R, -SSdat[2], 'r')
    ax[1,0].plot(R, -NTdat[2], 'b')
    ax[1,0].set_xlabel(r"$r$")
    ax[1,0].set_ylabel(r"$v^r$")
    ax[1,0].set_xscale('log')
    ax[1,0].set_yscale('log')
    ax[1,0].set_xlim(xlim)

    ax[1,1].plot(r, prim[:,3], 'k+')
    ax[1,1].plot(R, SSdat[3], 'r')
    ax[1,1].plot(R, NTdat[3], 'b')
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
    ax[0,2].plot(R, SSdat[4], 'r')
    ax[0,2].plot(R, NTdat[4], 'b')
    ax[0,2].set_xlabel(r"$r$")
    ax[0,2].set_ylabel(r"$\dot{Q}$")
    ax[0,2].set_xscale('log')
    ax[0,2].set_yscale('log')
    ax[0,2].set_xlim(xlim)

    u0 = 1.0 / np.sqrt(1 - 2*M/r - 4*M/r*vr - (1+2*M/r)*vr*vr - r*r*vp*vp)
    mdot = -2*np.pi*r*sig*u0*vr

    ax[1,2].plot(r, mdot, 'k+')
    ax[1,2].plot(R, MDOT*np.ones(R.shape), 'r')
    ax[1,2].plot(R, MDOT*np.ones(R.shape), 'b')
    ax[1,2].set_xlabel(r"$r$")
    ax[1,2].set_ylabel(r"$\dot{M}$")
    ax[1,2].set_xscale('log')
    ax[1,2].set_yscale('log')
    ax[1,2].set_xlim(xlim)

    plt.tight_layout()

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Need a parfile and checkpoint, numbnuts!")

    par = dp.readParfile(sys.argv[1])
    g = dp.Grid(par)
    g.loadCheckpoint(sys.argv[2])

    plotNT(g)

    plt.show()


