import sys
import math
import numpy as np
import matplotlib.pyplot as plt
#import colormath.color_objects as cmco
#import colormath.color_conversions as cmcc
import discopy as dp
import discoGR as gr
import plot_thin as pt

# All constants in c.g.s.
sb = 1.56055371e59
c = 2.99792458e10
mp = 1.672621777e-24
h = 6.62606957e-27
ka_bbes = 0.4
rg_solar = 1.4766250385e5
r_scale = rg_solar
rho_scale = 1.0
eV = 6.24150934e11
kpc = 3.08567758149e21

blue = (31.0/255, 119.0/255, 180.0/255)
orange = (255.0/255, 127.0/255, 14.0/255)
green = (44.0/255, 160.0/255, 44.0/255)
red = (214.0/255, 39.0/255, 40.0/255)
purple = (148.0/255, 103.0/255, 189.0/255)

class RayData:

    X = None
    U = None
    xi = None
    ixy = None
    extent = None
    nx = -1
    ny = -1
    M = -1.0
    mode = 0
    inc = 0.0

    def __init__(self, rayfile, M=1.0):
        self.loadRayfile(rayfile, M)

    def loadRayfile(self, rayfile, M):
        print("Loading Rays")

        f = open(rayfile, "r")
        line0 = f.readline()  # ### Input Params ###
        line1 = f.readline()  # Metric: ? 
        line2 = f.readline()  # Grid: ?
        line3 = f.readline()  # N1: ?
        line4 = f.readline()  # N2: ?
        line5 = f.readline()  # X1a: ?
        line6 = f.readline()  # X1b: ?
        line7 = f.readline()  # X2a: ?
        line8 = f.readline()  # X2b: ?
        line9 = f.readline()  # distance: ?
        line10 = f.readline()  # inclination: ?
        line11 = f.readline()  # azimuth: ?
        f.close()
        self.mode = int(line2.split(':')[1])
        self.inc = (math.pi/180.0) * float(line10.split(':')[1])

        dat = np.loadtxt(rayfile, skiprows=12)
        xi = dat[:,2]
        yi = dat[:,3]
        self.xi = dat[:,2:4]
        self.X = dat[:,12:16]
        self.U = dat[:,16:20]
        self.ixy = dat[:,0:2].astype(np.int64)

        self.X[:,0:2] *= M
        self.U[:,2:4] *= M

        ximax = xi.max()
        ximin = xi.min()
        yimax = yi.max()
        yimin = yi.min()
        self.extent = (M*ximin, M*ximax, M*yimin, M*yimax)

        self.nx = self.ixy[:,0].max()+1
        self.ny = self.ixy[:,1].max()+1

    def rotate(self, phi):
        self.X[:,3] += phi


def getTz(g, rays, massScale=1.0):
#Returns the effective temperature (in the comoving frame) at the source of 
# each ray and the redshift between the source (in the comoving frame) and 
# the end point of the ray.  massScale is the mass of the central BH in solar 
# masses, it provides a weak scaling on T: Teff ~ M_BH^-1/8. massScale can be
# different than the DISCO mass, which is assumed to be 1 M_solar.

    shape = (rays.X.shape[0],)

    Tmap = np.zeros(shape)
    zmap = np.zeros(shape)

    #for ind,ij in enumerate(rays.ixy):
    for i in xrange(shape[0]):
        #i = ij[0]
        #j = ij[1]
        # if ray hit horizon, T & z are zero.
        if rays.U[i,0] == 0:
            Tmap[i] = -1.0
            zmap[i] = -1.0
            continue

        R = rays.X[i,1]
        Phi = rays.X[i,3]
        ir = np.searchsorted(g.rFaces, R) - 1
        if ir >= g.nr_tot or ir < 0:
            Tmap[i] = -1.0
            zmap[i] = -1.0
            continue
        shift = np.argmin(g.pFaces[0][ir])
        piph = np.roll(g.pFaces[0][ir], -shift)
        while Phi > piph.max():
            Phi -= 2*math.pi
        while Phi < piph.min():
            Phi += 2*math.pi
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
        qdot = 4*sb * Tc*Tc*Tc*Tc / (3*ka_bbes*sig * rho_scale*r_scale)
        Tmap[i] = math.pow(qdot/sb, 0.25)
        
        #print ir, ip
        #print Tc

        u0, ur, up = gr.calc_u(R, vr, vp, g._pars)

        if g._pars['Metric'] == 6 and g._pars['BoostType'] == 1:
            binW = g._pars['BinW']
            up += u0*binW

        ut = 0.0

        U0 = rays.U[i,0]
        UR = rays.U[i,1]
        UT = rays.U[i,2]
        UP = rays.U[i,3]

        #u2 = (-1+2*M/R)*u0*u0 + 4*M/R*u0*ur + (1+2*M/R)*ur*ur + R*R*up*up
        #U2 = (-1-2*M/R)*U0*U0 + 4*M/R*U0*UR + (1-2*M/R)*UR*UR \
        #        + (UP*UP + UT*UT)/(R*R)
        #print u2, U2

        z1 = -(u0*U0 + ur*UR + ut*UT + up*UP)
        zmap[i] = 1.0/z1
        #print z1

    Tmap *= math.pow(massScale, -1.0/8.0)

    return Tmap, zmap

def specificIntensity(Tmap, zmap, nu):
    # Calculates specific Intensity given source temperatues and redshifts
    # for each image pixel.  Assumes temperature is in ergs, nu is in eV
    # and returns I_\nu in cgs: erg/cm^2 s Hz ster.


    #Effective Temp in eV
    T = Tmap * zmap * eV

    #Inu = np.zeros((len(nu), Tmap.shape[0], Tmap.shape[1]))
    #for i,v in enumerate(nu):
    #    Inu[i] = 2*v*v*v/(np.exp(v/T)-1.0)
    #    Inu[i][Tmap==-1.0] = 0.0

    nu = np.atleast_1d(nu)

    if len(nu) > 1:
        Inu = 2*(nu*nu*nu)[:,None] / (
                np.exp(nu[:,None]/T[None,:])-1.0)
        for i in xrange(len(nu)):
            Inu[i][Tmap==-1.0] = 0.0
    else:
        Inu = 2*nu*nu*nu / (np.exp(nu/T) - 1.0)
        Inu[Tmap == -1.0] = 0.0

    return Inu/(eV*eV*eV*h*h*c*c)

def makeImage(g, rays, nus, redshift='yes', massScale=1.0):

    print("Generating Temperature and redshift maps")
    Tmap, zmap = getTz(g, rays, massScale=massScale)

    if redshift == 'no':
        zmap[:] = 1.0

    print("Generating intensity map")
    Inus = specificIntensity(Tmap, zmap, nus)

    #Transform images into screen coordinates, origin at top left.
    #Inus = np.transpose(Inus, (0,2,1))[:,:,::-1]
    
    return rays.xi, Inus

def makePicture(g, rays, numin, numax, redshift='yes'):

    print("Generating Temperature and redshift maps")
    Tmap, zmap = getTz(g, rays)

    if redshift == 'no':
        zmap[:,:] = 1.0

    optical_lambda = np.linspace(340.0, 830.0, num=50) * 1.0e-7 # in cm
    optical_nu = c/optical_lambda * h * eV # in eV

    pic = np.zeros((rays.nx, rays.ny, 3))

    print("Generating intensity map")
    Inus = specificIntensity(Tmap, zmap, optical_nu)

    for i in xrange(rays.nx):
        for j in xrange(rays.ny):
            print i,j
            spd = Inus[:,i,j] * optical_nu*optical_nu / (c * h*h*eV*eV)
            specColor = cmco.SpectralColor(*spd)
            rgb = cmcc.convert_color(specColor, cmco.sRGBColor)
            pic[i,j,:] = rgb.get_value_tuple()


    #Transform images into screen coordinates, origin at top left.
    pic = np.transpose(pic, (1,0,2))[:,::-1,:]
    
    return pic

def makeSpectrum(g, rays, nus, D=1.0, redshift='yes', massScale=1.0):

    print("Generating Temperature and redshift maps")
    Tmap, zmap = getTz(g, rays, massScale=massScale)

    if redshift == 'no':
        zmap[:,:] = 1.0

    Fnu = np.zeros(nus.shape)

    print("Generating intensity maps")

    if rays.mode == 0:
        dx = (rays.extent[1]-rays.extent[0]) / (rays.nx-1)
        dy = (rays.extent[3]-rays.extent[2]) / (rays.ny-1)
        dA = dx*dy

        for i,nu in enumerate(nus):
            Inu = specificIntensity(Tmap, zmap, nu)
            Fnu[i] = (Inu * dA).sum() / (D*D*kpc*kpc)

    elif rays.mode == 1:
        Nr = rays.nx
        Np = rays.ny
        ra = rays.xi[0,0]
        rb = rays.xi[(Nr-1)*Np,0]

        R = ra * np.power(rb/ra, np.arange(Nr) / float(Nr-1))

        print ra, rb
        
        Inu = specificIntensity(Tmap, zmap, nus)
        print Inu.shape
        Inu.resize((len(nus), Nr, Np))
        print Inu.shape
        Inup = Inu.sum(axis=2) * 2.0*np.pi/Np
        print Inup.shape

        Inurp = 0.5*(R[None,:-1]*Inup[:,:-1]+R[None,1:]*Inup[:,1:])
        print Inurp.shape
        dR = R[1:] - R[:-1]

        Fnu = (Inurp[:,:]*dR[None,:]).sum(axis=1) * (
                math.cos(rays.inc) / (D*D*kpc*kpc))

    return Fnu

def genNTgrid(pars, Mdot, alpha, gam):

    pars2 = pars.copy()

    pars2['AlphaVisc'] = alpha
    pars2['Adiabatic_Index'] = gam
    pars2['NumR'] = 2048
    pars2['NP_CONST'] = 3
    pars2['NumZ'] = 1

    M = pars2['GravM']
    rs = 6*M

    pars2['R_Min'] = 1.1*rs
    pars2['R_Max'] = 1.0e3

    g = dp.Grid(pars2)

    rf = g.rFaces
    r = 0.5*(rf[1:] + rf[:-1])

    SSdat, NTdat = pt.calcNT(g, r, rs, Mdot)

    #These are in code units
    sigNT = NTdat[0]
    piNT = NTdat[1]
    vrNT = NTdat[2]
    vpNT = NTdat[3]
    
    g.prim = []

    for k in xrange(g.nz_tot):
        slice = []
        for i in xrange(g.nr_tot):
            annulus = np.zeros((g.np[k,i], g.nq))
            annulus[:,0] = sigNT[i]
            annulus[:,1] = piNT[i]
            annulus[:,2] = vrNT[i]
            annulus[:,3] = vpNT[i]
            slice.append(annulus)

        g.prim.append(slice)

    return g


if __name__ == "__main__":

    if len(sys.argv) < 6:
        print("makePicture.py: generates a ray-traced image of emission from thegiven checkpoints.")
        print("usage: python makePicture.py mode parfile rayfile <checkpoints ...> prefix [SS, NT]")
        print("   'mode' can be: spectrum - Compute spectrum of each chkpt")
        print("                  lightcurve - Compute lightcurve of all chkpts")
        print("                  picture - Optical rendering with colors (not implemented)")
        print("                  image - Specific intensity image")
        sys.exit()

    mode = sys.argv[1]
    parfile = sys.argv[2]
    rayfile = sys.argv[3]
    chkfiles = sys.argv[4:-1]
    prefix = sys.argv[-1]

    pars = dp.readParfile(parfile)
    g = dp.Grid(pars)
    rays = RayData(rayfile, g._pars["GravM"])
   
    if mode == "spectrum":
        nus = np.logspace(0.0, 5.0, base=10.0, num=100)

        FnuNT1 = None

        Mdot = 0.3
        alpha = 0.01
        gam = 5.0/3.0
        gNT = genNTgrid(pars, Mdot, alpha, gam) 
        FnuNT1 = makeSpectrum(gNT, rays, nus, D=1.0, redshift='yes')
        gNT = genNTgrid(pars, 10*Mdot, alpha, gam) 
        FnuNT2 = makeSpectrum(gNT, rays, nus, D=1.0, redshift='yes')
        gNT = genNTgrid(pars, 0.1*Mdot, 0.1*alpha, gam) 
        FnuNT3 = makeSpectrum(gNT, rays, nus, D=1.0, redshift='yes')
        #gNT = genNTgrid(pars, 8*Mdot, 0.1*alpha, gam) 
        #FnuNT4 = makeSpectrum(gNT, rays, nus, D=1.0, redshift='yes')

        Nrot = 8

        for i,chkpt in enumerate(chkfiles):
            g.loadCheckpoint(chkpt)
            
            for n in xrange(Nrot):
                Fnu = makeSpectrum(g, rays, nus, D=1.0, redshift='yes')
                rays.rotate(2*np.pi/Nrot)

                fig, ax = plt.subplots()
                ax.plot(nus/1000.0, Fnu / (h*nus), 'k+')
                ax.set_xscale("log")
                ax.set_yscale("log")
                ylim = ax.get_ylim()

                if FnuNT1 is not None:
                    ax.plot(nus/1000.0, FnuNT1 / (h*nus), lw=10.0, 
                            color=blue, alpha=0.5)
                    ax.plot(nus/1000.0, FnuNT2 / (h*nus), lw=10.0, 
                            color=orange, alpha=0.5)
                    ax.plot(nus/1000.0, FnuNT3 / (h*nus), lw=10.0, 
                            color=green, alpha=0.5)
             #       ax.plot(nus/1000.0, FnuNT4 / (h*nus), lw=10.0, 
             #                  color=red, alpha=0.5)
                ax.set_ylim(ylim)
                ax.set_xlabel(r"$\nu$ ($keV$)")
                ax.set_ylabel(r"$F_\nu / h\nu$ ($cts/cm^2 s Hz$)")
                ax.set_title(chkpt)
                fig.savefig("{0:s}_spectrum_{1:03d}_{2:03d}.png".format(
                                prefix,i,n))
                plt.close()

    elif mode == "lightcurve":
        nus = np.logspace(2.0, 4.0, base=10.0, num=3)
        Fnu = np.zeros((len(chkfiles), len(nus)))
        T = np.zeros(len(chkfiles))

        for i,chkpt in enumerate(chkfiles):
            g.loadCheckpoint(chkpt)
            T[i] = g.T
            Fnu[i] = makeSpectrum(g, rays, nus, D=1.0, redshift='yes')

        for i,nu in enumerate(nus):
            fig, ax = plt.subplots()
            ax.plot(T, Fnu[:,i])
            ax.set_yscale("log")
            ax.set_xlabel(r"$T$ ($G M_\odot / c^3$)")
            ax.set_ylabel(r"$F_\nu$ ($erg/cm^2 s Hz$)")
            ax.set_title(r"$\nu = $ {0:.2f} $keV$)".format(nu/1000.0))
            fig.savefig("{0:s}_lightcurve_{1:d}.png".format(prefix, i))
            plt.close()

    elif mode == "picture":

        numin = 0.0
        numax = 1.0

        for i,chkpt in enumerate(chkfiles):
            g.loadCheckpoint(chkpt)

            pic = makePicture(g, rays, numin, numax, redshift='yes')

            fig, ax = plt.subplots()
            im = ax.imshow(pic, extent=rays.extent, aspect='equal')
            ax.set_title(chkpt)
            ax.set_xlabel(r"$X$ ($M_\odot$)")
            ax.set_ylabel(r"$Y$ ($M_\odot$)")
            fig.savefig("{0:s}_picture_{1:03d}.png".format(prefix,i))
            plt.close()

    else:
        nus = np.array([100.0, 200.0, 500.0, 1000.0, 2000.0, 8000.0])

        for i,chkpt in enumerate(chkfiles):
            g.loadCheckpoint(chkpt)

            nrot = 8;
            for n in xrange(nrot):

                xi, Inus = makeImage(g, rays, nus, redshift='yes')
                rays.rotate(2.0*np.pi/8.0)

                for j,Inu in enumerate(Inus):
                    fig, ax = plt.subplots()
                    #im = ax.imshow(img, cmap=plt.cm.afmhot, extent=rays.extent, 
                    #            aspect='equal')
                    im = ax.tricontourf(xi[:,0], xi[:,1], Inu, 256,
                                        antialiased=True, cmap=plt.cm.afmhot, 
                                        aspect='equal')
                    for coll in im.collections:
                        coll.set_edgecolor("face")
                    ax.set_title(r"$I_\nu$ ($\nu = $ {0:.2f} $keV$)".format(
                                nus[j]/1000.0))
                    ax.set_xlabel(r"$X$ ($M_\odot$)")
                    ax.set_ylabel(r"$Y$ ($M_\odot$)")
                    ax.set_axis_bgcolor('k')
                    width = np.fabs(xi).max()
                    ax.set_xlim(-width, width)
                    ax.set_ylim(-width, width)
                    plt.colorbar(im)
                    ax.set_aspect("equal")
                    #fig.savefig("{0:s}_{1:03d}_{2:03d}.pdf".format(prefix,i,j))
                    fig.savefig("{0:s}_{1:03d}_{2:03d}_{3:03d}.png".format(
                                    prefix,i,n,j), dpi=200)
                    plt.close()

