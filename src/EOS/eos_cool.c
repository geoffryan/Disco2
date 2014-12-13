#define EOS_PRIVATE_DEFS
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "../Headers/header.h"
#include "../Headers/EOS.h"
#include "../Headers/Metric.h"
#include "../Headers/Sim.h"

double eos_cool_none(double *prim, double H, struct Sim *theSim)
{
    return 0.0;
}

double eos_cool_isotherm(double *prim, double H, struct Sim *theSim)
{
    double T0 = sim_CoolPar1(theSim);
    double tau = sim_CoolPar2(theSim);

    return (prim[PPP]/prim[RHO] - T0) / tau;
}

double eos_cool_bb_es(double *prim, double H, struct Sim *theSim)
{
    //TODO: Work in 2D and 3D?
    double kappa = 0.2;
    double h = H*eos_r_scale;
    double rho = (prim[RHO]/H)*eos_rho_scale;
    double P = (prim[PPP]/H)*eos_rho_scale;
    double temp = 0.5 * eos_mp * P / rho * eos_c * eos_c;
    double q0 = sim_CoolPar1(theSim);
    double q = q0 * 8.0*eos_sb * temp*temp*temp*temp/(3.0*kappa*rho*h);
    double Q = q / (eos_c*eos_c*eos_c * eos_rho_scale);

    return Q;
}

double eos_cool_bb_ff(double *prim, double H, struct Sim *theSim)
{
    //TODO: Work in 2D and 3D?
    double h = H*eos_r_scale;
    double rho = (prim[RHO]/H)*eos_rho_scale;
    double P = (prim[PPP]/H)*eos_rho_scale;
    double temp = 0.5 * eos_mp * P / rho * eos_c * eos_c; //in ergs
    double kappa = 6.6e22 * rho * pow(temp/eos_k, -3.5);
    double q0 = sim_CoolPar1(theSim);
    double q = q0 * 8.0*eos_sb * temp*temp*temp*temp/(3.0*kappa*rho*h);
    double Q = q / (eos_c*eos_c*eos_c * eos_rho_scale);

    return Q;
}

double eos_cool_neutrino(double *prim, double H, struct Sim *theSim)
{
    double rho = (prim[RHO]/H)*eos_rho_scale;
    double P = (prim[PPP]/H)*eos_rho_scale;
    double h = H * eos_r_scale;
    double t11 = 0.5 * eos_mp * P / rho * eos_c * eos_c / (1.0e11 * eos_k);
    double q = 5.0e33 * pow(t11,9) * h;
    double Q = q / (eos_c*eos_c*eos_c * eos_rho_scale);

    return 0*Q;
}
