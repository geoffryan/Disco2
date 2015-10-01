#define CELL_PRIVATE_DEFS
#define EOS_PRIVATE_DEFS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../Headers/Cell.h"
#include "../Headers/Sim.h"
#include "../Headers/EOS.h"
#include "../Headers/Face.h"
#include "../Headers/GravMass.h"
#include "../Headers/Metric.h"
#include "../Headers/header.h"

//Novikov-Thorne Disc

double isco(double M, double a);
double Pfunction(double x, double xisco, double a);
void calc_NT(double r, double Mdot, double rs, double GAM, double M, double A, 
                double alpha, double *sig, double *pi, double *T, double *H, 
                double *vr, double *vp);
double rho_solve_NT(double sigma, double T, double r, double M, double u0, 
                    struct Sim *theSim);

//Thompson Cooling
void cell_init_ntdisc_thompson(struct Cell *c, double r, double phi, double z, struct Sim *theSim)
{
    //In outer regions this is a Novikov-Thorne disc with gas pressure
    //and thompson opacity.  At an inner radius it transitions to fully 
    //geodesic flow with constant temperature and surface density.

    double Mdot = sim_InitPar1(theSim); // Solar Masses per year
    double r0 = sim_InitPar2(theSim);   // radius to transition to geodesic
    double DR = sim_InitPar3(theSim);   // width of transition region
    double GAM = sim_GAMMALAW(theSim);
    double M = sim_GravM(theSim);
    double a = M*sim_GravA(theSim);
    double alpha = sim_AlphaVisc(theSim);

    struct Metric *g = metric_create(time_global, r, phi, z , theSim);
    double U0 = metric_frame_U_u_geo(g, 0, theSim);
    double UR = metric_frame_U_u_geo(g, 1, theSim);
    double UP = metric_frame_U_u_geo(g, 2, theSim);
    metric_destroy(g);

    Mdot *= eos_Msolar/eos_year;                         // Msolar/year -> g/s
    Mdot /= eos_rho_scale*eos_r_scale*eos_r_scale*eos_c; // g/s -> code units
    double rs = isco(M, a/M);

    double SigOut, PiOut, TOut, HOut, vrOut, vpOut;
    double Sig0, Pi0, T0, H0, vr0, vp0;

    calc_NT(r, Mdot, rs, GAM, M, a, alpha, &SigOut, &PiOut, &TOut, &HOut, 
                                        &vrOut, &vpOut);
    if(r < r0+DR)
        calc_NT(r0, Mdot, rs, GAM, M, a, alpha, &Sig0, &Pi0, &T0, &H0, 
                                                &vr0, &vp0);

    if(sim_BoostType(theSim) == BOOST_RIGID)
    {
        double w = sim_BinW(theSim);
        vp0 -= w;
        vpOut -= w;
    }

    double rho, T, sig, pi, vr, vp;

    if(r >= r0 + DR)
    {
        rho = SigOut/HOut;
        T = TOut;
        sig = SigOut;
        pi = PiOut;
        vr = vrOut;
        vp = vpOut;
    }
    else if(r < r0 - DR)
    {
        sig = Sig0;
        T = T0;
        rho = rho_solve_NT(Sig0, T0, r, M, U0, theSim);
        double p[5];
        p[RHO] = rho;
        p[TTT] = T0;
        p[URR] = UR/U0;
        p[UPP] = UP/U0;
        p[UZZ] = 0.0;
        pi = eos_ppp(p, theSim) * sig/rho;
        vr = UR/U0;
        vp = UP/U0;
    }
    else
    {
        double POW = 18.0;
        sig = pow(pow(SigOut,-POW)+pow(Sig0,-POW),-1.0/POW);
        T = pow(pow(TOut,-POW)+pow(T0,-POW),-1.0/POW);
        rho = rho_solve_NT(sig, T, r, M, U0, theSim);

        double vrIn = UR/U0;
        double vpIn = UP/U0;
        double w = 0.5*(tanh(4*(r-r0)/DR) + 1.0);
        vr = w*vrOut + (1-w)*vrIn;
        vp = w*vpOut + (1-w)*vpIn;
        
        double p[5];
        p[RHO] = rho;
        p[TTT] = T;
        p[URR] = vr;
        p[UPP] = vp;
        p[UZZ] = 0.0;
        pi = eos_ppp(p, theSim) * sig/rho;
    }

    if(sim_Background(theSim) == GRDISC)
    {
        c->prim[RHO] = rho;
        c->prim[TTT] = T;
    }
    else
    {
        c->prim[RHO] = sig;
        c->prim[PPP] = pi;
    }
    c->prim[URR] = vr;
    c->prim[UPP] = vp;
    c->prim[UZZ] = 0.0;

    if(sim_NUM_C(theSim)<sim_NUM_Q(theSim)) 
    {
        int i;

        for(i=sim_NUM_C(theSim); i<sim_NUM_Q(theSim); i++)
        {
            if(r*cos(phi) < 0)
                c->prim[i] = 0.0;
            else
                c->prim[i] = 1.0;
        }
    }
}

void cell_init_ntdisc_calc(struct Cell *c, double r, double phi, double z, 
                            struct Sim *theSim)
{
    int disc_num = sim_InitPar0(theSim);

    if(disc_num == 0)
        cell_init_ntdisc_thompson(c, r, phi, z, theSim);
    else
        printf("ERROR: cell_init_ntdisc given bad option.\n");
}

void cell_single_init_ntdisc(struct Cell *theCell, struct Sim *theSim,int i,int j,int k)
{
    double rm = sim_FacePos(theSim,i-1,R_DIR);
    double rp = sim_FacePos(theSim,i,R_DIR);
    double r = 0.5*(rm+rp);
    double zm = sim_FacePos(theSim,k-1,Z_DIR);
    double zp = sim_FacePos(theSim,k,Z_DIR);
    double z = 0.5*(zm+zp);
    double t = theCell->tiph-.5*theCell->dphi;

    cell_init_ntdisc_calc(theCell, r, t, z, theSim);
}

void cell_init_ntdisc(struct Cell ***theCells,struct Sim *theSim,struct MPIsetup * theMPIsetup)
{
    int i, j, k;
    for (k = 0; k < sim_N(theSim,Z_DIR); k++) 
    {
        double zm = sim_FacePos(theSim,k-1,Z_DIR);
        double zp = sim_FacePos(theSim,k,Z_DIR);
        double z = 0.5*(zm+zp);
        
        for (i = 0; i < sim_N(theSim,R_DIR); i++) 
        {
            double rm = sim_FacePos(theSim,i-1,R_DIR);
            double rp = sim_FacePos(theSim,i,R_DIR);
            double r = 0.5*(rm+rp);

            for (j = 0; j < sim_N_p(theSim,i); j++) 
            {
                double t = theCells[k][i][j].tiph-.5*theCells[k][i][j].dphi;
             
                cell_init_ntdisc_calc(&(theCells[k][i][j]), r, t, z, theSim);
                
                if(PRINTTOOMUCH)
                {
                    printf("(%d,%d,%d) = (%.12lg, %.12lg, %.12lg): (%.12lg, %.12lg, %.12lg, %.12lg, %.12lg)\n", i,j,k,r,t,z,theCells[k][i][j].prim[RHO],theCells[k][i][j].prim[URR],theCells[k][i][j].prim[UPP],theCells[k][i][j].prim[UZZ],theCells[k][i][j].prim[PPP]);
                }
            }
        }
    }
}

double isco(double M, double a)
{
    //ISCO in Kerr spacetime. a is dimensionless.

    double Z1 = 1.0 + pow((1-a*a)*(1+a),1.0/3.0) + pow((1-a*a)*(1-a),1.0/3.0);
    double Z2 = sqrt(3*a*a + Z1*Z1);

    double risco;
    if(a > 0)
        risco = M * (3.0 + Z2 - sqrt((3.0-Z1)*(3.0+Z1+2.0*Z2)));
    else
        risco = M * (3.0 + Z2 + sqrt((3.0-Z1)*(3.0+Z1+2.0*Z2)));

    return risco;
}

double Pfunction(double X, double Xisco, double a)
{
    // Novikov and Thorne P-function, as calculated in Thorne & Page.
    // X = r/M, Xisco = risco/M
    double th = acos(a)/3.0;
    double ct = cos(th);
    double st = sin(th);
    double x1 = ct - sqrt(3.0)*st;
    double x2 = ct + sqrt(3.0)*st;
    double x3 = -2.0*ct;
    double c0 = -a*a / (x1*x2*x3);
    double c1 = (x1-a)*(x1-a) / (x1*(x2-x1)*(x3-x1));
    double c2 = (x2-a)*(x2-a) / (x2*(x1-x2)*(x3-x2));
    double c3 = (x3-a)*(x3-a) / (x3*(x1-x3)*(x2-x3));

    double x = sqrt(X);
    double xs = sqrt(Xisco);
    double P = 1.0 - xs/x - 3.0/x * (c0*log(x/xs) + c1*log((x-x1)/(xs-x1))
                            + c2*log((x-x2)/(xs-x2)) + c3*log((x-x3)/(xs-x3)));

    return P;
}

void calc_NT(double r, double Mdot, double rs, double GAM, double M, double A,
                double alpha, double *sig, double *pi, double *T, double *H, 
                double *vr, double *vp)
{
    double kappa = 0.4;             //Pure Hydrogen
    double omk = sqrt(M/(r*r*r));
    double B = 1.0+A*omk;
    double C = 1.0-3*M/r+2*A*omk;
    double D = 1.0-2*M/r+A*A/(r*r);
    double P = Pfunction(r/M, rs/M, A/M);

    double Pi = Mdot / (3.0*M_PI*alpha*sqrt(GAM)) * omk * sqrt(C)*P/(D*D);
    double Qdot = 3.0*Mdot / (4.0*M_PI) * omk*omk * P/C;
    double Sig = pow(8*eos_sb/(3*kappa*Qdot)
                            * pow(eos_mp*Pi*eos_rho_scale*eos_c*eos_c,4), 0.2)
                        / (eos_rho_scale*eos_r_scale);
    double ur = - Mdot / (2*M_PI*r*Sig);
    double up = omk / sqrt(C);
    double u0 = B / sqrt(C);
    double HH = sqrt(r*r*r*Pi / (M*(Sig+GAM/(GAM-1.0)*Pi))) / u0;

    *sig = Sig;
    *pi = Pi;
    *T = Pi/Sig;
    *H = HH;
    *vr = ur/u0;
    *vp = up/u0;
}

double rho_solve_NT(double sigma, double T, double r, double M, double u0, 
                    struct Sim *theSim)
{
    /*
     * A Newton-Raphson solver to find the central density 'rho' for a
     * given surface density 'sigma' and temperature 'T'.
     * Returns non-zero if an error occurs.
     */

    double rho0, rho1, rho;
    double rhoH, drhoH;
    double c = sqrt(r*r*r/M)/u0;
    int err = 0;
    double tol = 1.0e-12;

    rho0 = sigma / (c * sqrt(T/(1+T)));

    double p[5];
    p[RHO] = rho0;
    p[TTT] = T;
    p[URR] = 0.0;
    p[UPP] = 0.0;
    p[UZZ] = 0.0;

    rho0 = p[RHO];

    rho1 = rho0;
    int i = 0;
    do
    {
        rho = rho1;
        p[RHO] = rho;
        double eps = eos_eps(p, theSim);
        double P   = eos_ppp(p, theSim);
        double dedp = eos_depsdrho(p, theSim);
        double dPdp = eos_dpppdrho(p, theSim);

        double rhoh = rho + rho*eps + P;
        rhoH = rho * c * sqrt(P/rhoh);
        drhoH = 0.5 * rhoH/rhoh * (1.0 + eps + 2*P/rho - rho*dedp
                        + (rho+rho*eps)/P*dPdp);

        rho1 = rho - (rhoH-sigma) / drhoH;
        if(rho1 < 0.0)
            rho1 = 0.5*rho;

        i++;
    }
    while(fabs(rho-rho1)/rho > tol && i < 100);
    if(i>=100)
    {
        printf("ERROR: NR failed to converge in rho_solve().\n");
        printf("    rho0=%.12lg, rho1=%.12lg, T=%.12lg, err=%.12lg\n",
                rho0, rho1, p[TTT], (rho-rho1)/rho);
        err = 1;
    }

    return rho1;
}
