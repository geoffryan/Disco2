import sys
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

rho_scale = 1.0
r_scale = rg_solar

nu = np.logspace(12,20.3,num=500,base=10.0)
#nu = np.logspace(17,20,num=7,base=10.0)

def calcSpec(g, nu):

    R = np.zeros(g.np.sum())
    prim = np.zeros((g.np.sum(),5))
    dA = np.zeros(g.np.sum())

    ind = 0
    for k in xrange(g.nz_tot):
        for i in xrange(g.nr_tot):
            r1 = g.rFaces[i]
            r2 = g.rFaces[i+1]
            dr = r2-r1
            r = 0.5*(r1+r2)

            numP = g.np[k,i]
            dphi = np.zeros(numP)
            dphi[1:] = g.pFaces[k][i][1:] - g.pFaces[k][i][:-1]
            dphi[0] = g.pFaces[k][i][0] - g.pFaces[k][i][-1]
            dphi[dphi < 0] += 2*np.pi
            dphi[dphi > 2*np.pi] -= 2*np.pi

            R[ind:ind+numP] = r
            prim[ind:ind+numP,:] = g.prim[k][i][:,:5]
            dA[ind:ind+numP] = r*dr*dphi

            ind += numP

    Sig = prim[:,0] * rho_scale * r_scale # g/cm^2
    Pi = prim[:,1] * rho_scale * c * c * r_scale # erg/cm^2
    T = mp * Pi/Sig # erg

    qdot = 8*sb * T*T*T*T / (3*ka_bbes*Sig) # erg/(cm^2 s)

    Teff = np.power(qdot/sb, 0.25) # erg

    LnuR = (2*h*(nu*nu*nu)[:,None] / 
            (c*c*(np.exp(h*nu[:,None]/Teff[None,:]) - 1.0)))

    Lnu = (LnuR * dA[None,:]).sum(axis=1)

    #print h*nu[:,None]/Teff[None,:]

    return R, Teff, LnuR


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Need a parfile and checkpoints, numbnuts!")

    par = dp.readParfile(sys.argv[1])
    g = dp.Grid(par)

    nc = len(sys.argv[2:])
    nn = len(nu)

    Lnu = np.zeros((nc,nn))
    T = np.zeros(nc)

    for i,f in enumerate(sys.argv[2:]):

        print("Calculating {0:s}...".format(f))
        g.loadCheckpoint(f)
        T[i] = g.T
        R, Teff, LnuR, Lnu = calcSpec(g, nu)
        Lnu[i,:] = Lnu

    figlc, axlc = plt.subplots()
    for i in xrange(nn):
        axlc.plot(T, Lnu[:,i], label=str(nu[i]))
    axlc.set_yscale('log')
    axlc.set_xlabel(r'$t$ ($M_\odot$)')
    axlc.set_ylabel(r'$L_\nu$ ($erg$ / $s\cdot Hz$)')
    plt.legend()
    figlc.savefig("lc.png")
    plt.close()

    for i,t in enumerate(T):
        fig, ax =  plt.subplots()
        ax.plot(nu, Lnu[i], 'k+')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\nu$ ($Hz$)')
        ax.set_ylabel(r'$L_\nu$ ($erg$ / $s\cdot Hz$)')
        ax.set_title(r'$T$ = {0:s}'.format(str(t)))
        fig.savefig("spec_{0:04d}.png".format(i))
        plt.close()

