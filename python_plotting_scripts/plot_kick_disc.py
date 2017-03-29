import math
from numpy import *
import matplotlib.pyplot as plt
import sys
import numpy as np
import discopy as dp
import discoEOS as eos
import discoGR as gr

RMAX = np.inf
scale = 'log'

def calc_spectrum(filename, nu, pars):

    dat = dp.readCheckpoint(filename)
    t = dat[0]
    r = dat[1]
    sig = dat[4]
    pi  = dat[5]
    vr = dat[6]
    vp = dat[7]
    dA = dat[9]
    qs = np.array(dat[10])

    M = pars['GravM']
    a = pars['GravA']
    A = a*M

    u0, ur, up = gr.calc_u(r, vr, vp, pars)
    g00, g0r, g0p, grr, grp, gpp = gr.calc_g(r, pars)
    l0 = g00*u0 + g0r*ur + g0p*up
    lr = g0r*u0 + grr*ur + grp*up
    lp = g0p*u0 + grp*ur + gpp*up

    gam = pars['Adiabatic_Index']

    eps = pi/((gam-1)*sig)
    oms = np.sqrt(lp*lp - A*A*(l0*l0-1)) / (r*r)
    sigh = sig + gam/(gam-1)*pi
    H = np.sqrt(pi/sigh)/oms
    P = pi / (2*H)
    rho = sig / (2*H)
    rho_cgs = rho * eos.rho_scale
    P_cgs = P * eos.rho_scale*eos.c*eos.c
    H_cgs = H * eos.r_scale
    sig_cgs = sig * eos.rho_scale * eos.r_scale
    pi_cgs = pi * eos.rho_scale * eos.r_scale * eos.c*eos.c

    t_cgs = t * eos.r_scale / eos.c

    if pars['CoolingType'] == 6:
        T_erg = np.power(0.25*eos.c/eos.sb * P_cgs, 0.25)
    else:
        T_erg = eos.mp*pi_cgs/sig_cgs
    T_K = T_erg / eos.kb

    tau = eos.ka_es * sig_cgs

    F = 4*eos.sb/(3*tau) * T_erg*T_erg*T_erg*T_erg

    T_eff = np.power(F/eos.sb, 0.25)

    Inu = 2*eos.h*(nu*nu*nu)[:,None] / (eos.c*eos.c*
            (np.exp(eos.h * nu[:,None]/T_erg[None,:])-1.0))

    Fnu = (Inu[:,:] * dA[None,:]).sum(axis=1)

    return t_cgs, Inu, Fnu

def plot_r_profile_single(ax, r, f, sca, ylabel, pars):

    ax.plot(r, f, 'k+')

    ax.set_xscale('log')
    if (f == 0.0).all():
        ax.set_yscale('linear')
    else:
        ax.set_yscale(sca)

    Risco = gr.isco(pars)
    Rergo = gr.ergo(pars)
    Reh = gr.horizon(pars)

    xmin = math.pow(10, math.floor(math.log10(r.min())))
    if Risco is not None:
        ax.axvline(Risco, ls='--', lw='3', color='grey', alpha=0.5)
    if Rergo is not None:
        ax.axvspan(xmin, Rergo, color='lightgrey', alpha=0.5)
    if Reh is not None:
        ax.axvspan(xmin, Reh,   color='grey', alpha=0.5)
    ax.set_xlim(xmin, r.max())
    
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(ylabel)

def plot_r_profile(filename, pars, sca='linear'):

    dat = dp.readCheckpoint(filename)
    t = dat[0]
    r = dat[1]
    sig = dat[4]
    pi  = dat[5]
    vr = dat[6]
    vp = dat[7]
    qs = np.array(dat[10])


    real = (r>-1.0) * (r<RMAX)
    r = r[real]
    sig = sig[real]
    pi = pi[real]
    vr = vr[real]
    vp = vp[real]
    qs = qs[:,real]

    GAM = pars['Adiabatic_Index']
    M = pars['GravM']
    a = pars['GravA']
    A = M*a

    R = np.linspace(r.min(), r.max(), 500)
    
    sigh = sig + GAM/(GAM-1.0)*pi

    cs = np.sqrt(GAM * pi / sigh)
    eps =  pi / ((GAM-1.0)*sig)

    lapse = gr.lapse(r, pars)
    u0, ur, up = gr.calc_u(r, vr, vp, pars)
    g00, g0r, g0p, grr, grp, gpp = gr.calc_g(r, pars)
    l0 = g00*u0 + g0r*ur + g0p*up
    lr = g0r*u0 + grr*ur + grp*up
    lp = g0p*u0 + grp*ur + gpp*up
    w = lapse*u0
    u2 = w*w-1.0
    mach = np.sqrt(u2*(1-cs*cs)/(cs*cs))
    
    oms = np.sqrt(lp*lp - A*A*(l0*l0-1)) / (r*r)
    H = np.sqrt(pi/sigh)/oms

    Mdot = -2*math.pi* r * sig * ur

    sig_cgs = sig * eos.rho_scale*eos.r_scale
    pi_cgs = pi * eos.rho_scale*eos.r_scale*eos.c*eos.c
    vr_cgs = vr * eos.c
    vp_cgs = vp * eos.c/eos.r_scale
    oms_cgs = oms * eos.c/eos.r_scale
    H_cgs = H * eos.r_scale
    t_cgs = t * eos.r_scale/eos.c
    
    tau = eos.ka_es * sig_cgs

    if pars['CoolingType'] == 6:
        tcool = (tau+1.0/tau)*u0 / (3.0*oms_cgs) * np.sqrt(np.fmin(eps,1))
    elif pars['CoolingType'] == 2:
        tcool = 0.375 * tau * sig_cgs * u0 / (eos.sb*(eos.mp**4)*(gam-1)**4)\
                    * np.power(eps*eos.c*eos.c, -3)
    else:
        tcool = np.inf * np.ones(tau.shape)

    print("Plotting t = {0:g}".format(t))

    #Plot.
    fig, ax = plt.subplots(3, 3, figsize=(12,9))

    plot_r_profile_single(ax[0,0], r, sig_cgs, sca, r"$\Sigma$ (g/cm$^2$)",
                            pars)
    plot_r_profile_single(ax[0,1], r, pi_cgs,  sca,       r"$\Pi$ (erg/cm$^2$)",
                            pars)
    plot_r_profile_single(ax[0,2], r, eps,  sca, r"$\epsilon$ ($c^2$)",
                            pars)
    plot_r_profile_single(ax[1,0], r, vr_cgs,  "linear",  r"$v^r$ (cm/s)",
                            pars)
    plot_r_profile_single(ax[1,1], r, vp_cgs,  "linear",  r"$v^\phi$ (s$^{-1})$",
                            pars)
    if pars['CoolingType'] != 0:
        ax[2,0].axhline(t_cgs, lw=2, color='grey', alpha=0.5)
        ax[2,0].plot(R, 2*np.pi*R*np.sqrt(R/M)*eos.r_scale/eos.c, 
                        lw=2, ls='--', color='grey', alpha=0.5)
        plot_r_profile_single(ax[2,0], r, tcool,  sca,  r"$t_{cool}$ (s)",
                                pars)
    plot_r_profile_single(ax[2,1], r, tau,  sca,  r"$\tau$",
                            pars)
    plot_r_profile_single(ax[2,2], r, mach,  sca,  r"$\mathcal{M}$",
                            pars)

    plt.suptitle(r"$T = {0:.3g}$".format(t))

    plt.tight_layout()

    outname = "plot_kick_disc_{0}.png".format("_".join( filename.split("/")[-1].split(".")[0].split("_")[1:] ))

    print("Saving {0:s}...".format(outname))
    plt.savefig(outname)

    return fig

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("\nGive me a parameter (.par) and checkpoint (.h5) file.\n")
        sys.exit()
    
    parname = sys.argv[1]
    pars = dp.readParfile(parname)

    nu_min = 3.0e11  # 300GHz, far IR
    nu_max = 3.0e21  # 3000 PHz, gamma-ray
    Nnu = 300
    nu = np.logspace(math.log10(nu_min), math.log10(nu_max), 
                        num=Nnu, base=10.0)

    Fnus = np.empty((len(sys.argv[2:]), Nnu), dtype=np.float)
    ts = np.empty(len(sys.argv[2:]), dtype=np.float)

    for i, filename in enumerate(sys.argv[2:]):
        fig = plot_r_profile(filename, pars, sca=scale)
        plt.close(fig)
        t, Inu, Fnu = calc_spectrum(filename, nu, pars)
        Fnus[i,:] = Fnu[:]
        ts[i] = t

    f = open("emission.txt", "w")
    line = " ".join([str(x) for x in nu])
    f.write(line + "\n")
    for i in range(ts.shape[0]):
        line = str(ts[i]) + " " + " ".join([str(fnu) for fnu in Fnus[i,:]])
        f.write(line + "\n")
    f.close()
