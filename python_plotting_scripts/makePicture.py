import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import discopy as dp

# All constants in c.g.s.
sb = 1.56055371e59
c = 2.99792458e10
mp = 1.672621777e-24
h = 6.62606957e-27
ka_bbes = 0.2
rg_solar = 1.4766250385e5
r_scale = rg_solar
rho_scale = 1.0
eV = 6.24150934e11

class RayData:

    X = None
    U = None
    ixy = None
    extent = None

    def __init__(self, rayfile):
        self.loadRayfile(rayfile)

    def loadRayfile(self, rayfile):
        print("Loading Rays")
        dat = np.loadtxt(rayfile)
        xi = dat[:,2]
        yi = dat[:,3]
        self.X = dat[:,12:16]
        self.U = dat[:,16:20]
        self.ixy = dat[:,0:2].astype(np.int64)

        ximax = xi.max()
        ximin = xi.min()
        yimax = yi.max()
        yimin = yi.min()
        self.extent = (ximin, ximax, yimin, yimax)

def makeImage(g, rays, nus):

    print("Generating Temperature and redshift maps")
    Tmap, zmap = getTz(g, rays)

    #zmap[:,:] = 1.0

    print("Generating intensity map")
    Inus = specificIntensity(Tmap, zmap, nus)

    #Transform images into screen coordinates, origin at top left.
    Inus = np.transpose(Inus, (0,2,1))[:,:,::-1]
    
    return Inus

def specificIntensity(Tmap, zmap, nu):

    #Effective Temp in eV
    T = Tmap * zmap * eV

    Inu = np.zeros((len(nu), Tmap.shape[0], Tmap.shape[1]))
    for i,v in enumerate(nu):
        Inu[i] = 2*v*v*v/(np.exp(v/T)-1.0)
        Inu[i][Tmap==-1.0] = 0.0

    return Inu

def getTz(g, rays):

    nx = rays.ixy[:,0].max()+1
    ny = rays.ixy[:,1].max()+1

    Tmap = np.zeros((nx,ny))
    zmap = np.zeros((nx,ny))

    for ind,ij in enumerate(rays.ixy):
        i = ij[0]
        j = ij[1]
        # if ray hit horizon, T & z are zero.
        if rays.U[ind,0] == 0:
            Tmap[i,j] = -1.0
            zmap[i,j] = -1.0
            continue

        R = rays.X[ind,1]
        Phi = rays.X[ind,3]
        ir = np.searchsorted(g.rFaces, R) - 1
        if ir >= g.nr_tot or ir < 0:
            Tmap[i,j] = -1.0
            zmap[i,j] = -1.0
            continue
        shift = np.argmin(g.pFaces[0][ir])
        piph = np.roll(g.pFaces[0][ir], -shift)
        ip = np.searchsorted(piph, Phi)
        if ip == piph.shape[0]:
            ip = 0

        prim = g.prim[0][ir][ip]
        M = g._pars['GravM']

        sig = prim[0]
        pi = prim[1]
        vr = prim[2]
        vp = prim[3]

        Tc = pi/sig * mp*c*c
        qdot = 8*sb * Tc*Tc*Tc*Tc / (3*ka_bbes*sig * rho_scale*r_scale)
        Tmap[i,j] = math.pow(qdot/sb, 0.25)
        
        #print ir, ip
        #print Tc

        u0 = 1.0/math.sqrt(1.0-2*M/R-4*M/R*vr-(1+2*M/R)*vr*vr-R*R*vp*vp)
        ur = u0*vr
        ut = 0.0
        up = u0*vp
        U0 = rays.U[ind,0]
        UR = rays.U[ind,1]
        UT = rays.U[ind,2]
        UP = rays.U[ind,3]

        #u2 = (-1+2*M/R)*u0*u0 + 4*M/R*u0*ur + (1+2*M/R)*ur*ur + R*R*up*up
        #U2 = (-1-2*M/R)*U0*U0 + 4*M/R*U0*UR + (1-2*M/R)*UR*UR \
        #        + (UP*UP + UT*UT)/(R*R)
        #print u2, U2

        z1 = -(u0*U0 + ur*UR + ut*UT + up*UP)
        zmap[i,j] = 1.0/z1
        #print z1

    return Tmap, zmap


if __name__ == "__main__":

    if len(sys.argv) < 5:
        print("makePicture.py: generates a ray-traced image of emmision from thegiven checkpoints.")
        print("usage: python makePicture.py parfile rayfile <checkpoints ...> prefix")
        sys.exit()

    parfile = sys.argv[1]
    rayfile = sys.argv[2]
    chkfile = sys.argv[3:-1]
    prefix = sys.argv[-1]

    pars = dp.readParfile(parfile)
    g = dp.Grid(pars)
    g.loadCheckpoint(chkfile[0])

    rays = RayData(rayfile)
    
    nus = [100.0, 200.0, 500.0, 1000.0, 2000.0, 8000.0]

    imgs = makeImage(g, rays, nus)

    for i,img in enumerate(imgs):
        fig, ax = plt.subplots()
        im = ax.imshow(img, cmap=plt.cm.afmhot, extent=rays.extent, 
                    aspect='equal')
        ax.set_title(r"$I_\nu$ ($\nu = $ {0:.2f} $keV$)".format(nus[i]/1000.0))
        ax.set_xlabel(r"$X$ ($M_\odot$)")
        ax.set_ylabel(r"$Y$ ($M_\odot$)")
        plt.colorbar(im)
        fig.savefig("{0:s}_{1:d}.png".format(prefix,i))
        plt.close()

