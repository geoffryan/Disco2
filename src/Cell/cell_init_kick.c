#define CELL_PRIVATE_DEFS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../Headers/Cell.h"
#include "../Headers/EOS.h"
#include "../Headers/Face.h"
#include "../Headers/GravMass.h"
#include "../Headers/header.h"
#include "../Headers/Metric.h"
#include "../Headers/Sim.h"

#define DEBUG 0

//Kicked Discs
//
void calc_pos_orig(double *X, double *X0, double M0, double a0, 
                    double M1, double a1, double V);
void calc_vel_kick(double *X0, double *U0, double *X, double *U,
                    double M0, double a0, double M1, double a1, double V);

void calc_pos_orig(double *X, double *X0, double M0, double a0, 
                    double M1, double a1, double V)
{
    double t = X[0];
    double r = X[1];
    double p = X[2];
    double z = X[3];

    double A0 = M0*a0;
    double A1 = M1*a1;
    
    double R = sqrt(r*r+z*z);
    double sinth = r/R;
    double costh = z/R;
    double cosp = cos(p);
    double sinp = sin(p);

    double x = R*sinth*cosp - A1*sinth*sinp;
    double y = R*sinth*sinp + A1*sinth*cosp;

    double gam = 1.0 / sqrt(1-V*V);
    double x0 = gam*(x + V*t);
    double t0 = gam*(t + V*x);
    double y0 = y;
    double z0 = z;

    double b = x0*x0 + y0*y0 + z0*z0 - A0*A0;
    double R0 = sqrt(0.5*(b + sqrt(b*b - 4*A0*A0*z0*z0)));
    
    double costh0 = z0 / R0;
    double sinth0 = sqrt(1-costh0*costh0);
    double cosp0 = (R0*x0 + A0*y0) / ((R0*R0+A0*A0) * sinth0);
    double sinp0 = (R0*y0 - A0*x0) / ((R0*R0+A0*A0) * sinth0);

    double r0 = sqrt(R0*R0 - z0*z0);
    double p0 = atan2(sinp0, cosp0);

    X0[0] = t0;
    X0[1] = r0;
    X0[2] = p0;
    X0[3] = z0;
}

void calc_vel_kick(double *X0, double *U0, double *X, double *U,
                    double M0, double a0, double M1, double a1, double V)
{
    /* OK, let's lay this out.  This function transforms a velocity U0 at
     * position X0 in spheroidal kerr-schild coordinates in a black hole
     * spacetime of mass M0 and spin a0 to a new blackhole spacetime with
     * mass M1 and spin a1 boosted at speed V.
     *
     * Step 1: Transform U0 to cartesian coordinates.
     * Step 2: Abrupt metric change to mass M1, spin a1, speed V. 
     *          u_i preserved.
     * Step 3: Boost to BH rest frame.
     * Step 4: Cartesian --> spheroidal Kerr-Schild --> Disco cylindrical
     */

    double t0 = X0[0];
    double r0 = X0[1];
    double p0 = X0[2];
    double z0 = X0[3];

    double A0 = M0*a0;
    double A1 = M1*a1;

    double R0 = sqrt(r0*r0 + z0*z0);
    double sinth0 = r0 / R0;
    double costh0 = z0 / R0;
    double cosp0 = cos(p0);
    double sinp0 = sin(p0);

    // U0 in spheroidal Kerr-Schild coordinates (t, R, th, phi)
    double Uks[4] = {U0[0], sinth0*U0[1]+costh0*U0[3], 
                        costh0*U0[1]-sinth0*U0[3], U0[2]};
    double x0 = R0*sinth0*cosp0 - A0*sinth0*sinp0;
    double y0 = R0*sinth0*sinp0 + A0*sinth0*cosp0;

    double J[4][4]; // J[mu][nu] = dx'^mu/dx^nu
    J[0][0] = 1.0; J[0][1] = 0.0; J[0][2] = 0.0; J[0][3] = 0.0;
    J[1][0] = 0.0; J[1][1] = sinth0*cosp0; 
        J[1][2] = costh0*(R0*cosp0 - A0*sinp0);
        J[1][3] = -R0*sinth0*sinp0 - A0*sinth0*cosp0;
    J[2][0] = 0.0; J[2][1] = sinth0*sinp0; 
        J[2][2] = costh0*(R0*sinp0 + A0*cosp0);
        J[2][3] = R0*sinth0*cosp0 - A0*sinth0*sinp0;
    J[3][0] = 0.0; J[3][1] = costh0; J[3][2] = -R0*sinth0; J[3][3] = 0.0;

    double Uc[4] = {0.0, 0.0, 0.0, 0.0};
    int mu,nu;
    for(mu=0; mu<4; mu++)
        for(nu=0; nu<4; nu++)
            Uc[mu] += J[mu][nu]*Uks[nu];
    
    //OK! Uc is our pre-merger 4-velocity in cartesian coordinates. Need to
    // compute the covariant components, they'll be preserved post-merge.

    double H = 2*M0*R0 / (R0*R0 + A0*A0*costh0*costh0);
    double l[4] = {1.0, (R0*x0+A0*y0)/(R0*R0+A0*A0),
                    (R0*y0-A0*x0)/(R0*R0+A0*A0), z0/R0};
    double UL = Uc[0]*l[0] + Uc[1]*l[1] + Uc[2]*l[2] + Uc[3]*l[3];
    double Ucd[4];
    Ucd[0] = -Uc[0] + H*UL*l[0];
    for(mu=1; mu<4; mu++)
        Ucd[mu] = Uc[mu] + H*UL*l[mu];
   
    if(DEBUG)
    {
        printf("    U0:   (%.6lg %.6lg %.6lg %.6lg)\n", 
                U0[0], U0[1], U0[2], U0[3]);
        printf("    Uks:  (%.6lg %.6lg %.6lg %.6lg)\n", 
                Uks[0], Uks[1], Uks[2], Uks[3]);
        printf("    Uc:   (%.6lg %.6lg %.6lg %.6lg)\n", 
                Uc[0], Uc[1], Uc[2], Uc[3]);
        printf("    Ucd:  (%.6lg %.6lg %.6lg %.6lg)\n", 
                Ucd[0], Ucd[1], Ucd[2], Ucd[3]);
    }

    //OK! Covariant components calculated.  Now we change-up the metric!
    // Cartesian coordinates haven't changed, but 'a' has.
    
    double gam = 1.0/sqrt(1-V*V);
    double t1 = gam*(t0 - V*x0);
    double x1 = gam*(x0 - V*t0);
    double y1 = y0;
    double z1 = z0;

    double b = x1*x1 + y1*y1 + z1*z1 - A1*A1;
    double R1 = sqrt(0.5*(b + sqrt(b*b - 4*A1*A1*z1*z1)));
    H = 2*M1*R1*R1*R1 / (R1*R1*R1*R1 + A1*A1*z1*z1);

    // Kerr-Schild 'l' vector for boosted Kerr black hole in Lab frame.
    double l1[4] = {1.0, (R1*x1+A1*y1)/(R1*R1+A1*A1),
                    (R1*y1-A1*x1)/(R1*R1+A1*A1), z1/R1}; // In boosted frame.
    //In lab frame.
    l[0] = gam*(l1[0] - V*l1[1]);
    l[1] = gam*(l1[1] - V*l1[0]);
    l[2] = l1[2];
    l[3] = l1[3];

    //Construct lab-frame metric.  l_i = l^i, l_0 = -l^0.
    double lapse, n[4];
    lapse = 1.0 / sqrt(1+H*l[0]*l[0]);
    n[0] = 1.0/lapse;
    for(mu=1; mu<4; mu++)
        n[mu] = -H * l[0]*l[mu] / n[0];
    double ig[3][3];
    for(mu=0; mu<3; mu++)
        for(nu=0; nu<3; nu++)
            ig[mu][nu] = -H*l[mu+1]*l[nu+1]+n[mu+1]*n[nu+1];
    ig[0][0] += 1;
    ig[1][1] += 1;
    ig[2][2] += 1;

    double igUU = 0.0;
    for(mu=0; mu<3; mu++)
        for(nu=0; nu<3; nu++)
            igUU += ig[mu][nu] * Ucd[mu+1] * Ucd[nu+1];
    UL = l[1]*Ucd[1] + l[2]*Ucd[2] + l[3]*Ucd[3];

    Uc[0] = sqrt(1.0 + igUU) / lapse;
    Ucd[0] = -lapse*lapse*(Uc[0] - H*l[0]*UL);

    UL += l[0]*Ucd[0];
    for(mu=1; mu<4; mu++)
        Uc[mu] = Ucd[mu] - H*l[mu]*UL;

    // Now Uc is the 4-velocity in the lab frame, post merger!
    //
    // Time to boost to BH frame!

    double Ucd1[4] = {gam*(Ucd[0]+V*Ucd[1]), gam*(Ucd[1]+V*Ucd[0]),
                        Ucd[2], Ucd[3]};

    //Wow, that was easy!  Now to go to spheroidal KS coords.
    double costh1 = z1 / R1;
    double sinth1 = sqrt(1-costh1*costh1);
    double cosp1 = (R1*x1 + A1*y1) / ((R1*R1+A1*A1) * sinth1);
    double sinp1 = (R1*y1 - A1*x1) / ((R1*R1+A1*A1) * sinth1);

    // As earlier, J[mu][nu] = dx'^mu / dx_nu, x' = cart, x = KS.
    J[0][0] = 1.0; J[0][1] = 0.0; J[0][2] = 0.0; J[0][3] = 0.0;
    J[1][0] = 0.0; J[1][1] = sinth1*cosp1; 
        J[1][2] = costh1*(R1*cosp1 - A1*sinp1);
        J[1][3] = -R1*sinth1*sinp1 - A1*sinth1*cosp1;
    J[2][0] = 0.0; J[2][1] = sinth1*sinp1; 
        J[2][2] = costh1*(R1*sinp1 + A1*cosp1);
        J[2][3] = R1*sinth1*cosp1 - A1*sinth1*sinp1;
    J[3][0] = 0.0; J[3][1] = costh1; J[3][2] = -R1*sinth1; J[3][3] = 0.0;

    double Uksd[4] = {0.0, 0.0, 0.0, 0.0};
    for(mu=0; mu<4; mu++)
        for(nu=0; nu<4; nu++)
            Uksd[mu] += J[nu][mu] * Ucd1[nu];

    //Kerr-Schild l_mu in spheroidal components.
    l[0] = 1.0;
    l[1] = 1.0;
    l[2] = 0.0;
    l[3] = -A1*sinth1*sinth1;

    double eta[4][4];
    double isig2 = 1.0 / (R1*R1 + A1*A1*costh1*costh1);
    eta[0][0] = -1.0; eta[0][1] = 0.0; eta[0][2] = 0.0; eta[0][3] = 0.0;
    eta[1][0] = 0.0; eta[1][1] = (R1*R1+A1*A1)*isig2; eta[1][2] = 0.0; 
        eta[1][3] = A1*isig2;
    eta[2][0] = 0.0; eta[2][1] = 0.0; eta[2][2] = isig2; eta[2][3] = 0.0; 
    eta[3][0] = 0.0; eta[3][1] = A1*isig2; eta[3][2] = 0.0; 
        eta[3][3] = isig2 / (sinth1*sinth1);

    double lu[4] = {0.0, 0.0, 0.0, 0.0};
    for(mu=0; mu<4; mu++)
        for(nu=0; nu<4; nu++)
            lu[mu] += eta[mu][nu]*l[nu];

    for(mu=0; mu<4; mu++)
    {
        Uks[mu] = 0.0;
        for(nu=0; nu<4; nu++)
            Uks[mu] += (eta[mu][nu] - H*lu[mu]*lu[nu])*Uksd[nu];
    }

    //Now to DISCO's cylindrical coordinates!

    U[0] = Uks[0];
    U[1] = sinth1*Uks[1] + costh1*Uks[2];
    U[2] = Uks[3];
    U[3] = costh1*Uks[1] - sinth1*Uks[2];

    if(DEBUG)
    {
        printf("    Ucd:  (%.6lg %.6lg %.6lg %.6lg)\n", 
                Ucd[0], Ucd[1], Ucd[2], Ucd[3]);
        printf("    Ucd1: (%.6lg %.6lg %.6lg %.6lg)\n", 
                Ucd1[0], Ucd1[1], Ucd1[2], Ucd1[3]);
        printf("    Uksd: (%.6lg %.6lg %.6lg %.6lg)\n", 
                Uksd[0], Uksd[1], Uksd[2], Uksd[3]);
        printf("    Uks:  (%.6lg %.6lg %.6lg %.6lg)\n", 
                Uks[0], Uks[1], Uks[2], Uks[3]);
        printf("    U:    (%.6lg %.6lg %.6lg %.6lg)\n", 
                U[0], U[1], U[2], U[3]);
    }
}

void calc_u_geo(double *x, double *u, double M, double a)
{
    double r = x[1];
    double phi = x[2];
    double z = 0.0;
    
    double Z1 = 1.0 + pow((1.0-a*a)*(1.0+a), 1.0/3.0)
                     + pow((1.0-a*a)*(1.0-a), 1.0/3.0);
    double Z2 = sqrt(3.0*a*a + Z1*Z1);
    double Risco;
    if(a >= 0.0)
        Risco = M*(3.0 + Z2 - sqrt((3.0-Z1)*(3.0+Z1+2.0*Z2)));
    else
        Risco = M*(3.0 + Z2 + sqrt((3.0-Z1)*(3.0+Z1+2.0*Z2)));

    if(r > Risco)
    {
        double u0 = (r + a*M*sqrt(M/r)) / sqrt(r*r - 3*M*r
                                                    + 2*a*sqrt(M*M*M*r));
        double omk = sqrt(M/(r*r*r));
        u[0] = u0;
        u[1] = 0.0;
        u[2] = u0 * omk / (1.0 + a*M*omk);
        u[3] = 0.0;
    }
    else
    {
        double OMK = sqrt(M/(Risco*Risco*Risco));
        double U0 = (1.0 + a*M*OMK) / sqrt(1.0 - 3*M/Risco + 2*a*M*OMK);
        double UP = U0 * OMK / (1.0 + a*M*OMK);
        double eps = (-1.0 + 2*M/Risco) * U0 + (-2.0*M*M*a/Risco) * UP;
        double lll = (-2.0*M*M*a/Risco) * U0
                        + (Risco*Risco + M*M*a*a + 2*M*M*M*a*a/Risco) * UP;

        //We're in for a treat.
        double H = 2*M/r;
        double kt = -1.0;
        double kr = 1.0;
        double kp = 0.0;

        double gtt = -1 - H*kt*kt;
        double grr = 1+M*M*a*a/(r*r) - H*kr*kr;
        double gpp = 1.0/(r*r) - H*kp*kp;
        double gtr = 0.0 - H*kt*kr;
        double gtp = 0.0 - H*kt*kp;
        double grp = M*a/(r*r) - H*kr*kp;

        double A = grr;
        double hB = gtr*eps + grp*lll;
        double C = gtt*eps*eps + 2*gtp*eps*lll + gpp*lll*lll + 1.0; 

        double urd = (-hB - sqrt(hB*hB - A*C)) / A;

        u[0] = gtt*eps + gtr*urd + gtp*lll;
        u[1] = gtr*eps + grr*urd + grp*lll;
        u[2] = gtp*eps + grp*urd + gpp*lll;
        u[3] = 0.0;
    }
}

void cell_init_kick_isothermal(struct Cell *c, double r, double phi, double z,
                                struct Sim *theSim)
{
    int i;
    double M = sim_GravM(theSim);
    double a = sim_GravA(theSim);
    double al = sim_AlphaVisc(theSim);
    double GAM = sim_GAMMALAW(theSim);
    double VMAX = 0.5;


    double r0 = sim_InitPar1(theSim);
    double rho0 = sim_InitPar2(theSim);
    double T = sim_InitPar3(theSim);
    double kickV = sim_InitPar4(theSim);
    double massFac = sim_InitPar5(theSim);
    double a0 = sim_InitPar6(theSim);

    double X[4] = {time_global, r, phi, z};
    double X0[4], u[4], u0[4];

    double M0 = M / massFac;

    calc_pos_orig(X, X0, M0, a0, M, a, kickV);
    calc_u_geo(X0, u0, M0, a0);
    calc_vel_kick(X0, u0, X, u, M0, a0, M, a, kickV);

    if(DEBUG)
    {
        printf("(%.6lg %.6lg %.6lg %.6lg) <--", X0[0], X0[1], X0[2], X0[3]);
        printf(" (%.6lg %.6lg %.6lg %.6lg)\n", X[0], X[1], X[2], X[3]);
        printf("(%.6lg %.6lg %.6lg %.6lg) -->", u0[0], u0[1], u0[2], u0[3]);
        printf(" (%.6lg %.6lg %.6lg %.6lg)\n", u[0], u[1], u[2], u[3]);
    }

    double rho = rho0 * pow(X0[1]/r0, -1.5);
    double P = T * rho;
    double vr = u[1] / u[0];
    double vp = u[2] / u[0];
    double vz = u[3] / u[0];
    double q = r>r0 ? 0.0 : 1.0;

    c->prim[RHO] = rho;
    c->prim[PPP] = P;
    c->prim[URR] = vr;
    c->prim[UPP] = vp;
    c->prim[UZZ] = 0.0;

    for(i=sim_NUM_C(theSim); i<sim_NUM_Q(theSim); i++)
        c->prim[i] = q;
}

void cell_init_kick_calc(struct Cell *c, double r, double phi, double z,
                            struct Sim *theSim)
{
    int opt = sim_InitPar0(theSim);

    if(opt == 0)
        cell_init_kick_isothermal(c, r, phi, z, theSim);
    else
        printf("ERROR: cell_init_kick given bad option.\n");
}

void cell_single_init_kick(struct Cell *theCell, struct Sim *theSim,
                            int i, int j, int k)
{
    double rm = sim_FacePos(theSim,i-1,R_DIR);
    double rp = sim_FacePos(theSim,i,R_DIR);
    double r = 0.5*(rm+rp);
    double zm = sim_FacePos(theSim,k-1,Z_DIR);
    double zp = sim_FacePos(theSim,k,Z_DIR);
    double z = 0.5*(zm+zp);
    double t = theCell->tiph-.5*theCell->dphi;

    cell_init_kick_calc(theCell, r, t, z, theSim);
}

void cell_init_kick(struct Cell ***theCells,struct Sim *theSim,
                    struct MPIsetup *theMPIsetup)
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

                cell_init_kick_calc(&(theCells[k][i][j]), r, t, z, theSim);

                if(PRINTTOOMUCH)
                {
                    printf("(%d,%d,%d) = (%.12lg, %.12lg, %.12lg): (%.12lg, %.12lg, %.12lg, %.12lg, %.12lg)\n",
                          i,j,k,r,t,z,theCells[k][i][j].prim[RHO], theCells[k][i][j].prim[URR],
                          theCells[k][i][j].prim[UPP], theCells[k][i][j].prim[UZZ],
                          theCells[k][i][j].prim[PPP]);
                }
            }
        }
    }
}
