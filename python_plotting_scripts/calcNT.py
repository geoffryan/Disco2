import math
import numpy as np
import matplotlib.pyplot as plt
import discopy as dp
import discoGR as gr
import discoEOS as eos

kappa = 0.4

def PfuncNT(r, M, a, rs):
    x = np.sqrt(r/M)
    xs = math.sqrt(rs/M)
    
    x1 = 2*math.cos((math.acos(a)+math.pi) / 3.0)
    x2 = 2*math.cos((math.acos(a)-math.pi) / 3.0)
    x3 = -2*math.cos(math.acos(a)/3.0)

    A = -a*a / (x1*x2*x3)
    B = (x1-a)*(x1-a) / (x1*(x2-x1)*(x3-x1))
    C = (x2-a)*(x2-a) / (x2*(x3-x2)*(x1-x2))
    D = (x3-a)*(x3-a) / (x3*(x1-x3)*(x2-x3))

    P = 1.0 - xs/x - 3.0/x * (
              A * np.log(np.fabs(x/xs))
            + B * np.log(np.fabs((x-x1)/(xs-x1)))
            + C * np.log(np.fabs((x-x2)/(xs-x2)))
            + D * np.log(np.fabs((x-x3)/(xs-x3))))

    return P

def isco(M, a):
    z1 = 1.0 + math.pow((1-a*a)*(1+a),1/3.0) + math.pow((1-a*a)*(1-a),1/3.0)
    z2 = math.sqrt(3.0*a*a + z1*z1)

    if a > 0.0:
        return M * (3.0 + z2 - math.sqrt((3.0-z1)*(3.0+z1+2.0*z2)))
    elif a < 0.0:
        return M * (3.0 + z2 + math.sqrt((3.0-z1)*(3.0+z1+2.0*z2)))
    else:
        return 6.0*M

def rhoSolve(SIG, T, up, pars):

    tol = 1.0e-10
    Tnat = T/(eos.mp*eos.c*eos.c)
    c2 = eos.c * eos.c

    rho0 = SIG / (np.sqrt(T/eos.mp)/up)
    rho1 = rho0

    i = 0

    while True:
        print(i)
        rho = rho1
        P = c2 * eos.ppp(rho,Tnat,pars)
        dPdrho = c2 * eos.dPdr(rho,Tnat,pars)

        sig = np.sqrt(P*rho)/up
        dsig = 0.5*sig * (dPdrho/P + 1.0/rho)
        drho = - (sig-SIG) / dsig

        rho1 = rho + drho
        rho1[rho1<0.0] = 0.5*rho[rho1<0.0]
        i += 1

        if (np.fabs(rho1-rho)/rho < tol).all() or i > 100:
            break

    return rho1


def calcNT(r, M, a, Mdot, alpha, type="gas"):

    if type is "rad":
        gam = 4.0/3.0
        pars = {"EOSType": 1,
                "EOSPar1": 1.0,
                "EOSPar2": 1.0,
                "EOSPar3": 0.0,
                "Adiabatic_Index": gam}
    else:
        gam = 5.0/3.0
        pars = {"EOSType": 0,
                "EOSPar1": 1.0,
                "EOSPar2": 0.0,
                "EOSPar3": 0.0,
                "Adiabatic_Index": gam}

    omk = eos.c * np.sqrt(M/(r*r*r))
    rs = isco(M, a)
    A = 1.0 + a*a*M*M/(r*r) + 2*a*a*M*M*M*M/(r*r*r*r*r*r)
    B = 1.0 + a*M*omk
    C = 1.0 - 3.0*M/r + 2.0*a*M*omk
    D = 1.0 - 2.0*M/r + a*a*M*M/(r*r)
    P = PfuncNT(r, M, a, rs)

    al = alpha * math.sqrt(gam)

    Pi = Mdot / (3.0*math.pi*al) * omk * np.sqrt(C) * P / (D*D)
    Qdot = 3.0*Mdot / (4.0*math.pi) * omk*omk * P / C

    if type is "rad":
        Sig = 4.0*eos.c*eos.c/(kappa*kappa) * omk*omk * Pi/(C*Qdot*Qdot)
    else:
        Sig = np.power(8.0*eos.sb/(3.0*kappa) * eos.mp**4 * Pi*Pi*Pi*Pi/Qdot, 
                        0.2)
    T = np.power(3.0*kappa/(8.0*eos.sb) * Sig * Qdot, 0.25)

    ur = -Mdot / (2.0*math.pi*r * Sig)

    #TODO: USE UP NOT VP
    u0 = np.sqrt((1 + ur/eos.c*ur/eos.c/(1-2*M/r+a*a*M*M/(r*r)))
            / (1 - 2*M/r + 4*M*M*a*omk/eos.c/r
                - (r*r+a*a*M*M*(1+2*M/r))*omk/eos.c*omk/eos.c))
    up = u0 * omk
    rho = rhoSolve(Sig, T, up, pars)

    cs = eos.c * np.sqrt(eos.cs2(rho, T/(eos.mp*eos.c*eos.c), pars))
    u = np.sqrt(r*r*up*up / (1.0-r*r*up*up/(eos.c*eos.c)))
    mach = u / (cs/np.sqrt(1.0-cs*cs/(eos.c*eos.c)))

    Mstar = M / (3*eos.rg_solar)
    Mdotstar = Mdot / 1.0e17
    x = np.sqrt(r/M)
    H = np.sqrt(Pi/Sig)/up

    if type is "rad":
         QC = Qdot / (Mdotstar/Mstar**2 * np.power(x,-6) * P/C)
         SC = Sig / (Mstar/(Mdotstar*alpha) * x**3 *np.power(C,1.5)/(D**2 *P))
         HC = H / (Mdotstar * P)
         TC = T / (math.pow(alpha*Mstar,-0.25) * np.power(x,-0.75)
                    * np.power(C,0.125) * np.power(D, -0.5)) / eos.kb
         rC = SC/HC
    else:
         QC = Qdot / (Mdotstar/Mstar**2 * np.power(x,-6) * P/C)
         SC = Sig / (math.pow(alpha,-0.8) * math.pow(Mstar,-0.4)
                * math.pow(Mdotstar,0.6) * np.power(x,-1.2) * np.power(C,0.6)
                * np.power(D,-1.6) * np.power(P,0.6))
         HC = H / (math.pow(alpha,-0.1) * math.pow(Mstar,-1.1)
                 * math.pow(Mdotstar,0.4) * np.power(x,2.1) * np.power(C,0.45)
                 * np.power(D,-0.2) * np.power(P,0.2))
         TC = T / (math.pow(alpha,0.2) * math.pow(Mstar,-0.6)
                 * math.pow(Mdotstar,0.4) * np.power(x,-1.8)
                    * np.power(C,-0.1) * np.power(D, -0.4) * np.power(P,0.4))\
                / eos.kb
         rC = SC/HC

    print("Qdot: {0:g} ({1:g}, {2:g})".format(QC.mean(), QC.min(), QC.max()))
    print("Sig:  {0:g} ({1:g}, {2:g})".format(SC.mean(), SC.min(), SC.max()))
    print("H:    {0:g} ({1:g}, {2:g})".format(HC.mean(), HC.min(), HC.max()))
    print("rho:  {0:g} ({1:g}, {2:g})".format(rC.mean(), rC.min(), rC.max()))
    print("T:    {0:g} ({1:g}, {2:g})".format(TC.mean(), TC.min(), TC.max()))

    return rho, Sig, T, ur, omk, Qdot, mach, pars

def plotNT(R, M, a, Mdot, alpha, dat, title):
    
    fig, ax = plt.subplots(2,3,figsize=(12,9))

    rho, Sig, T, ur, omk, Qdot, mach, pars = dat

    r = R/M

    ax[0,0].plot(r, Sig, 'k+')
    prettify(ax[0,0], xscale="log", yscale="log", xlabel=r"$r$ $(M)$",
                                                ylabel=r"$\Sigma$ $(g/cm^2)$")
    ax[0,1].plot(r, T, 'k+')
    prettify(ax[0,1], xscale="log", yscale="log", xlabel=r"$r$ $(M)$",
                                                ylabel=r"$T$ $(erg)$")
    ax[0,2].plot(r, Qdot, 'k+')
    prettify(ax[0,2], xscale="log", yscale="log", xlabel=r"$r$ $(M)$",
                                            ylabel=r"$\dot{Q}$ $(erg/cm^2 s)$")
    ax[1,0].plot(r, ur, 'k+')
    prettify(ax[1,0], xscale="log", yscale="linear", xlabel=r"$r$ $(M)$",
                                                ylabel=r"$u^r$ $(cm/s)$")
    ax[1,1].plot(r, omk, 'k+')
    prettify(ax[1,1], xscale="log", yscale="log", xlabel=r"$r$ $(M)$",
                                                ylabel=r"$v^\phi$ $(1/s)$")
    ax[1,2].plot(r, mach, 'k+')
    prettify(ax[1,2], xscale="log", yscale="log", xlabel=r"$r$ $(M)$",
                                                ylabel=r"$\mathcal{M}$")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig("{0:s}.png".format(title))
    plt.close(fig)

def prettify(ax, xscale="linear", yscale="linear", xlim=None, ylim=None, 
                xlabel=None, ylabel=None):
    
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

if __name__ == "__main__":

    M = 1.0 * eos.rg_solar
    a = 0.0
    #Mdot = 1.0e-10 * eos.M_solar / eos.year
    Mdot = 1.0e17
    R = M * np.logspace(1.0, 4.0, base=10.0, num=200)
    alpha = 1.0

    gasDat = calcNT(R, M, a, Mdot, alpha, "gas")
    radDat = calcNT(R, M, a, Mdot, alpha, "rad")

    plotNT(R, M, a, Mdot, alpha, gasDat, "Gas")
    plotNT(R, M, a, Mdot, alpha, radDat, "Radiation")

    fig, ax = plt.subplots(1,1)
    rhoG = gasDat[0]
    TG = gasDat[2]
    parG = gasDat[7]
    rhoR = radDat[0]
    TR = radDat[2]
    parR = radDat[7]

    parR["Adiabatic_Index"] = 5.0/3.0

    PgasG = eos.P_gas(rhoG, TG/(eos.mp*eos.c**2), parR)
    PradG = eos.P_rad(rhoG, TG/(eos.mp*eos.c**2), parR)
    PgasR = eos.P_gas(rhoR, TR/(eos.mp*eos.c**2), parR)
    PradR = eos.P_rad(rhoR, TR/(eos.mp*eos.c**2), parR)
    ax.plot(R/M, PgasG, color='g', ls='-')
    ax.plot(R/M, PradG, color='r', ls='-')
    ax.plot(R/M, PgasR, color='g', ls='--')
    ax.plot(R/M, PradR, color='r', ls='--')
    ax.set_xscale("log")
    ax.set_yscale("log")

    fig.savefig("Pcomparison.png")
    plt.close(fig)
