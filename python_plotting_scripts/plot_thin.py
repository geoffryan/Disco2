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

def ISCO(M, a):

    Z1 = 1.0 + math.pow((1-a*a)*(1+a),1.0/3.0) \
            + math.pow((1-a*a)*(1-a),1.0/3.0)
    Z2 = math.sqrt(3*a*a + Z1*Z1)
    if a > 0:
        Risco = M*(3.0 + Z2 - math.sqrt((3.0-Z1)*(3.0+Z1+2*Z2)))
    else:
        Risco = M*(3.0 + Z2 + math.sqrt((3.0-Z1)*(3.0+Z1+2*Z2)))

    return Risco

def calcNT(g, r, rs, Mdot):

    M = g._pars['GravM']
    a = g._pars['GravA']
    GAM = g._pars['Adiabatic_Index']
    AL = g._pars['AlphaVisc']

    A = M*a
    OMK = np.sqrt(M/(r*r*r))

    B = 1.0 + A*OMK
    C = 1.0 - 3.0*M/r + 2*A*OMK
    D = 1.0 - 2.0*M/r + A*A/(r*r)
    if A == 0:
        P = 1.0 - np.sqrt(rs/r) + np.sqrt(3*M/r)*(np.arctanh(np.sqrt(3*M/r))
                                            - np.arctanh(np.sqrt(3*M/rs)))
    else:
        th = math.acos(a) / 3.0
        ct = math.cos(th)
        st = math.sin(th)
        x1 = ct - math.sqrt(3.0)*st
        x2 = ct + math.sqrt(3.0)*st
        x3 = -2*ct
        c0 = -a*a/(x1*x2*x3)
        c1 = (x1-a)*(x1-a) / (x1*(x2-x1)*(x3-x1))
        c2 = (x2-a)*(x2-a) / (x2*(x1-x2)*(x3-x2))
        c3 = (x3-a)*(x3-a) / (x3*(x1-x3)*(x2-x3))

        x = np.sqrt(r/M)
        xs = math.sqrt(rs/M)

        #fig, ax = plt.subplots()
        P = 1.0 - xs/x - 3/x * (c0 * np.log(x/xs)
                                + c1 * np.log((x-x1)/(xs-x1))
                                + c2 * np.log((x-x2)/(xs-x2))
                                + c3 * np.log((x-x3)/(xs-x3)))
        #ax.plot(r, P)
        #P = 1.0 - np.sqrt(rs/r) + np.sqrt(3*M/r)*(np.arctanh(np.sqrt(3*M/r))
        #                                    - np.arctanh(np.sqrt(3*M/rs)))
        #ax.plot(r, P)
        #plt.show()


    U0 = B/np.sqrt(C)

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
    VP = OMK / B

    return (sig, pi, vr, vp, Qdot), (SIG, PI, VR, VP, QDOT)

def calcGEO(g, r, Mdot, K):

    M = g._pars['GravM']
    A = g._pars['GravA']
    GAM = g._pars['Adiabatic_Index']

    if A == 0:
        ur = -np.power(6*M/r - 1.0, 1.5) / 3.0
        up = 2*math.sqrt(3.0)*M/(r*r)
        u0 = (-2*M/r*ur - np.sqrt(4*M*M/(r*r)*ur*ur - (-1+2.0*M/r) * (1.0+(1.0+2.0*M/r)*ur*ur + r*r*up*up))) / (-1+2*M/r)
    
    else:

        Risco = ISCO(M, A)

        OMK = math.sqrt(M/(Risco*Risco*Risco))
        U0 = (1.0 + A*M*OMK) / math.sqrt(1.0 - 3*M/Risco + 2*A*M*OMK)
        UP = OMK / math.sqrt(1.0 - 3*M/Risco + 2*A*M*OMK)

        EPS = -(1-2*M/Risco)*U0 - 2*M*M*A/Risco*UP
        LLL = -2*M*M*A/Risco*U0 + (Risco*Risco+M*M*A*A+2*M*M*M*A*A/Risco)*UP

        ur = np.zeros(r.shape)
        up = np.zeros(r.shape)
        u0 = np.zeros(r.shape)


    vr = ur/u0
    vp = up/u0

    rho = -Mdot / (2*math.pi*r*ur)

    P = K * np.power(rho, GAM)

    return rho, P, vr, vp

def plotNT(g):

    r, prim = gToArr(g)

    M = g._pars['GravM']
    a = g._pars['GravA']
    Risco = ISCO(M,a)

    R = np.logspace(math.log10(Risco)+0.01, math.log10(r.max()), base=10.0, num=200)
    R2 = np.logspace(math.log10(r.min()), math.log10(Risco)-0.05, base=10.0, num=200)

    RS = Risco
    MDOT = 347
    K = 2.5e-5

    SSdat, NTdat = calcNT(g, R, RS, MDOT)
    GEOdat = calcGEO(g, R2, MDOT, K)

    if a != 0.0:
        g._pars['GravA'] = 0.0
        R3 = np.logspace(math.log10(6*M)+0.01, math.log10(r.max()), base=10.0, num=200)
        SSdat2, NTdat2 = calcNT(g, R3, 6*M, MDOT)

    fig, ax = plt.subplots(2,3, figsize=(14,9))

    print r.shape
    print prim.shape
    print R.shape
    print SSdat[0].shape

    blue = (31.0/255, 119.0/255, 180.0/255)
    orange = (255.0/255, 127.0/255, 14.0/255)
    green = (44.0/255, 160.0/255, 44.0/255)
    red = (214.0/255, 39.0/255, 40.0/255)

    xlim = (1.0, 1.0e3)

    ax[0,0].plot(r, prim[:,0], 'k+')
    ax[0,0].plot(R, SSdat[0], ls='-', lw=3.0, color=orange)
    ax[0,0].plot(R2, GEOdat[0], ls='-', lw=3.0, color=green)
    ax[0,0].plot(R, NTdat[0], ls='-', lw=3.0, color=blue)
    if a != 0.0:
        ax[0,0].plot(R3, NTdat2[0], ls='-', lw=3.0, color=red)
    ax[0,0].set_xlabel(r"$r$")
    ax[0,0].set_ylabel(r"$\Sigma$")
    ax[0,0].set_xscale('log')
    ax[0,0].set_yscale('log')
    ax[0,0].set_xlim(xlim)
    
    ax[0,1].plot(r, prim[:,1], 'k+')
    ax[0,1].plot(R, SSdat[1], ls='-', lw=3.0, color=orange)
    ax[0,1].plot(R2, GEOdat[1], ls='-', lw=3.0, color=green)
    ax[0,1].plot(R, NTdat[1], ls='-', lw=3.0, color=blue)
    if a != 0.0:
        ax[0,1].plot(R3, NTdat2[1], ls='-', lw=3.0, color=red)
    ax[0,1].set_xlabel(r"$r$")
    ax[0,1].set_ylabel(r"$\Pi$")
    ax[0,1].set_xscale('log')
    ax[0,1].set_yscale('log')
    ax[0,1].set_xlim(xlim)

    ax[1,0].plot(r, -prim[:,2], 'k+')
    ax[1,0].plot(R, -SSdat[2], ls='-', lw=3.0, color=orange)
    ax[1,0].plot(R2, -GEOdat[2], ls='-', lw=3.0, color=green)
    ax[1,0].plot(R, -NTdat[2], ls='-', lw=3.0, color=blue)
    if a != 0.0:
        ax[1,0].plot(R3, -NTdat2[2], ls='-', lw=3.0, color=red)
    ax[1,0].set_xlabel(r"$r$")
    ax[1,0].set_ylabel(r"$v^r$")
    ax[1,0].set_xscale('log')
    ax[1,0].set_yscale('log')
    ax[1,0].set_xlim(xlim)

    ax[1,1].plot(r, prim[:,3], 'k+')
    ax[1,1].plot(R, SSdat[3], ls='-', lw=3.0, color=orange)
    ax[1,1].plot(R2, GEOdat[3], ls='-', lw=3.0, color=green)
    ax[1,1].plot(R, NTdat[3], ls='-', lw=3.0, color=blue)
    if a != 0.0:
        ax[1,1].plot(R3, NTdat2[3], ls='-', lw=3.0, color=red)
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


