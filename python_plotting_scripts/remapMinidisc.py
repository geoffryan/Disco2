import math
import sys
import numpy as np
import discopy as dp
import discoGR as gr

def extrapolatePlunge(g1, g2):
    
    rmin1 = g1._pars["R_Min"]
    M = g1._pars["GravM"]
    gam = g1._pars["Adiabatic_Index"]

    sig1 = g1.prim[0][0][:,0]
    pi1 = g1.prim[0][0][:,1]
    vr1 = g1.prim[0][0][:,2]
    vp1 = g1.prim[0][0][:,3]
    r1 = 0.5*(g1.rFaces[0]+g1.rFaces[1])
    #u01 = np.power(1-2*M/r1 - 4*M/r1*vr1 - (1+2*M/r1)*vr1*vr1 - r1*r1*vp1*vp1, 
    #                -0.5)
    u01, ur1, up1 = gr.calc_u(r1, vr1, vp1, g1._pars)
    dphi1 = np.zeros(g1.pFaces[0][0].shape[0])
    dphi1[1:] = g1.pFaces[0][0][1:] - g1.pFaces[0][0][:-1]
    dphi1[0] = g1.pFaces[0][0][1] - g1.pFaces[0][0][0]
    dphi1[dphi1 > 2*np.pi] -= 2*np.pi
    dphi1[dphi1 < 0] += 2*np.pi

    s1 = np.log(np.power(pi1/sig1, 1/(gam-1)) / sig1)
    Mdot = -(r1*sig1*u01*vr1*dphi1).sum()
    Sdot = -(r1* s1 *sig1*u01*vr1*dphi1).sum()

    s = Sdot / Mdot

    M = g2._pars["GravM"]

    for i in xrange(g2.nr_tot):
        if g2.rFaces[i] > g1.rFaces[0]:
            break

        r = 0.5*(g2.rFaces[i]+g2.rFaces[i])
        ur = -np.power(6*M/r-1, 1.5) / 3.0
        up = 2.0*math.sqrt(3.0)*M/(r*r)
        u0 = (-2*M/r*ur - np.sqrt(4*M*M/(r*r)*ur*ur 
                    - (-1+2*M/r) * (1+(1+2*M/r)*ur*ur + r*r*up*up)))\
                / (-1+2*M/r)
        if g2._pars['BoostType'] == 1:
            bw = g2._pars['BinW']
            up -= bw*u0

        vr = ur/u0
        vp = up/u0
        sig = -Mdot / (2*np.pi*r*ur)
        pi = np.power(sig, gam) * np.exp((gam-1)*s)

        g2.prim[0][i][:,0] = sig
        g2.prim[0][i][:,1] = pi
        g2.prim[0][i][:,2] = vr
        g2.prim[0][i][:,3] = vp
        g2.prim[0][i][:,4:] = 0.0


if __name__ == "__main__":

    archive1 = sys.argv[1]
    parfile = sys.argv[2]
    filename = sys.argv[3]

    print("Loading grids...")
    g1 = dp.Grid(archive=archive1)
    g2 = dp.Grid(dp.readParfile(parfile))

    print("Remapping...")
    dp.remapGrid(g1, g2, gradOrder=0)

    print("Extrapolating inner boundary")
    extrapolatePlunge(g1, g2)

    g2.saveArchive(filename)



