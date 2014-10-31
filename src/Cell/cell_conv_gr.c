#define CELL_PRIVATE_DEFS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../Headers/Cell.h"
#include "../Headers/Sim.h"
#include "../Headers/Face.h"
#include "../Headers/GravMass.h"
#include "../Headers/Metric.h"
#include "../Headers/header.h"

void cell_prim2cons_gr( double * prim , double * cons , double *pos , double dV ,struct Sim * theSim)
{
    int i,j;

    struct Metric *g;
    double a, b[3], sqrtg;
    double u0, u[4], U[4];
    double rho, Pp, v[3];
    double GAMMALAW, rhoh;
    double r, phi, z;
    r = pos[R_DIR];
    phi = pos[P_DIR];
    z = pos[Z_DIR];

    //Get hydro primitives
    rho = prim[RHO];
    Pp  = prim[PPP];
    v[0]  = prim[URR];
    v[1]  = prim[UPP];
    v[2]  = prim[UZZ];

    //Get needed metric values
    g = metric_create(time_global, r, phi, z, theSim);
    a = metric_lapse(g);
    for(i=0; i<3; i++)
        b[i] = metric_shift_u(g, i);
    sqrtg = metric_sqrtgamma(g)/r;
    for(i=0; i<4; i++)
        U[i] = metric_frame_U_u(g, i, theSim);

    //Calculate 4-velocity, u0 = u^0, u[i] = u_i
    u0 = 1.0 / sqrt(-metric_g_dd(g,0,0) - 2*metric_dot3_u(g,b,v) - metric_square3_u(g,v));
    u[0] = metric_g_dd(g,0,0) * u0 + u0*metric_dot3_u(g,b,v);
    for(i=1; i<4; i++)
    {
        u[i] = 0;
        for(j=0; j<3; j++)
            u[i] += metric_gamma_dd(g,i-1,j) * (v[j]+b[j]);
        u[i] *= u0;
    }

    GAMMALAW = sim_GAMMALAW(theSim);
    rhoh = rho + GAMMALAW*Pp/(GAMMALAW - 1.);

    if(u0 < 0)
        printf("whoa, u0 < 0! (r=%.12f)\n", r);
    if(a < 0)
        printf("whoa, a < 0! (r=%.12f)\n", r);
    if(sqrtg < 0)
        printf("whoa, sqrtg < 0! (r=%.12f)\n", r);
    if(dV < 0)
        printf("whoa, dV < 0! (r=%.12f)\n", r);
    
    cons[DDD] = a*sqrtg*u0 * rho * dV;
    cons[SRR] = a*sqrtg*rhoh*u0 * u[1] * dV;
    cons[LLL] = a*sqrtg*rhoh*u0 * u[2] * dV;
    cons[SZZ] = a*sqrtg*rhoh*u0 * u[3] * dV;
    cons[TAU] = a*sqrtg*(u0*(-rhoh*(U[0]*u[0]+U[1]*u[1]+U[2]*u[2]+U[3]*u[3]) - rho) - U[0]*Pp) * dV;
    //cons[TAU] = sqrtg * (a*u0*(a*u0*rhoh - rho) - Pp) * dV;

    int q;
    for( q=sim_NUM_C(theSim) ; q<sim_NUM_Q(theSim) ; ++q ){
        cons[q] = prim[q]*cons[DDD];
    }

    metric_destroy(g);
}

void cell_cons2prim_gr(double *cons, double *prim, double *pos, double dV, struct Sim *theSim)
{
    int i,j;
    struct Metric *g;
    double a, b[3], sqrtg, U[4];
    double GAMMALAW, rhoh, u0, hmo;
    double rho, v[3], Pp;
    double rhostar, S[3], tau;
    double w, wmo, wmo0, wmo1; //w = lapse * u^0, wmo = w-1
    double e, s2, gam, Us; //e = tau/rhostar, s2 = S_i*S^i/rhostar^2, gam = (GAMMALAW-1)/GAMMALAW, Us = (U0*b^i+U^i)S_i
    double c[5];
    double eps = 1.0e-14;
    double CS_FLOOR, CS_CAP, RHO_FLOOR;
    double r, phi, z;

    r = pos[R_DIR];
    phi = pos[P_DIR];
    z = pos[Z_DIR];

    g = metric_create(time_global, r, phi, z, theSim);

    //Metric quantities needed later
    a = metric_lapse(g);
    for(i=0; i<3; i++)
        b[i] = metric_shift_u(g,i);
    sqrtg = metric_sqrtgamma(g)/r;
    for(i=0; i<4; i++)
        U[i] = metric_frame_U_u(g,i,theSim);

    //Conserved Quantities
    rhostar = cons[DDD]/dV;
    S[0] = cons[SRR]/dV;
    S[1] = cons[LLL]/dV;
    S[2] = cons[SZZ]/dV;
    tau = cons[TAU]/dV;
    GAMMALAW = sim_GAMMALAW(theSim);

    //Dimensionless conserved variables for w recovery
    e = tau/rhostar;
    s2 = metric_square3_d(g,S)/(rhostar*rhostar);
    gam = (GAMMALAW-1.0)/GAMMALAW;
    Us = 0.0;
    for(i=0; i<3; i++)
        Us += (U[0]*b[i]+U[i+1])*S[i];
    Us /= rhostar;
    double E = (1.0 + e + Us) / (a*U[0]);

    //Coefficients of polynomial to solve for w
    //TODO: expand E and arrange to minimize truncation error.
    c[0] = -s2*(gam-1)*(gam-1);
    c[1] = 2*((E-gam)*(E-gam)+2*(gam-1)*s2);
    c[2] = (5*E-gam)*(E-gam)-2*(3-gam)*s2;
    c[3] = 4*(E*E-s2) - 2*gam*E;
    c[4] = E*E-s2;
    if(c[4] <= 0)
    {
        printf("ERROR: E2 <= s2!: e^2 = %lg, s^2 = %lg\n",E*E,s2);
        printf("r = %lg, dV = %lg\n", r, dV);
        printf("rho = %lg, P = %lg, vr = %lg, vp = %lg, vz = %lg\n",prim[RHO],prim[PPP],prim[URR],prim[UPP],prim[UZZ]);
        printf("rhostar = %lg, tau = %lg, Sr = %lg, Sp = %lg, Sz = %lg\n",cons[DDD],cons[TAU],cons[SRR],cons[LLL],cons[SZZ]);
        //exit(0);
    }

    //Inital guess: previous value of prim
    v[0] = prim[URR];
    v[1] = prim[UPP];
    v[2] = prim[UZZ];
    wmo0 = a / sqrt(-metric_g_dd(g,0,0) - 2*metric_dot3_u(g,b,v) - metric_square3_u(g,v)) - 1.0;
    //Newton-Raphson to find w.
    wmo1 = wmo0;
    i = 0;
    do
    {
        wmo = wmo1;
        wmo1 = wmo - (c[0] + c[1]*wmo + c[2]*wmo*wmo + c[3]*wmo*wmo*wmo + c[4]*wmo*wmo*wmo*wmo)
                    / (c[1] + 2*c[2]*wmo + 3*c[3]*wmo*wmo + 4*c[4]*wmo*wmo*wmo);
        i++;
    }
    while(fabs(wmo-wmo1) > eps && i < 10000);
    if(i == 10000)
    {
        printf("NR failed to converge: wmo0 = %lg, wmo = %lg, wmo1 = %lg\n", wmo0,wmo,wmo1);
        printf("Poly coeffs: c[0]=%lg, c[1]=%lg, c[2]=%lg, c[3]=%lg, c[4]=%lg\n",c[0],c[1],c[2],c[3],c[4]);
    }
    wmo = wmo1;

    if(wmo < 0.0)
    {
        printf("ERROR: w-1 < 0 in cons2prim_gr. (r = %lg, wmo = %lg)\n", r, wmo);
        wmo = 0.0;
    }
    
    //Prim recovery
    w = wmo + 1.0;
    u0 = w/a;
    hmo = w*(E-w)/(w*w-gam);
    if(hmo < -1.0)
        printf("ERROR: h < 0 in cons2prim_gr (r = %lg, h-1 = %lg)\n", r, hmo);
    else if(hmo < 0.0)
    {
        printf("ERROR: h < 1 in cons2prim_gr (r = %.12g, h-1 = %.12g)\n", r, hmo);
        printf("       e^2 = %.12g, s^2 = %.12g, w =%.12g\n", e*e, s2, w);
    }

    RHO_FLOOR = sim_RHO_FLOOR(theSim);
    CS_FLOOR = sim_CS_FLOOR(theSim);
    CS_CAP = sim_CS_CAP(theSim);

    rho = rhostar/(sqrtg*w);
    if(rho < RHO_FLOOR)
    {
        rho = RHO_FLOOR;
        printf("Whoa, not enough density pal! (r=%.12g)\n", r);
    }
    Pp = gam * rho * hmo;
    if(Pp < CS_FLOOR*CS_FLOOR*rho*(hmo+1)/GAMMALAW)
    {
        Pp = CS_FLOOR*CS_FLOOR*rho*(hmo+1)/GAMMALAW;
        printf("Whoa, that's a pretty low pressure bub! (r=%.12g)\n", r);
    }
    if(Pp > CS_CAP*CS_CAP*rho*(hmo+1)/GAMMALAW)
    {
        Pp = CS_CAP*CS_CAP*rho*(hmo+1)/GAMMALAW;
        printf("Whoa, that's a really high pressure chum! (r=%.12g)\n", r);
    }

    for(i=0; i<3; i++)
    {
        v[i] = 0.0;
        for(j=0; j<3; j++)
            v[i] += metric_gamma_uu(g,i,j) * S[j];
        v[i] /= u0*rhostar*(hmo+1.0);
        v[i] -= b[i];
    }

    prim[RHO] = rho;
    prim[URR] = v[0];
    prim[UPP] = v[1];
    prim[UZZ] = v[2];
    prim[PPP] = Pp;
    
    for(i = sim_NUM_C(theSim); i < sim_NUM_Q(theSim); i++)
        prim[i] = cons[i]/cons[DDD];

    metric_destroy(g);

    if(PRINTTOOMUCH)
    {
        double cons2[sim_NUM_Q(theSim)];
        cell_prim2cons(prim,cons2,pos,dV,theSim);
        printf("cons2prim: r=%.12g, dV = %.12g\n", r, dV);
        printf("cons: rhostar=%.12g, Sr=%.12g, Sp=%.12g, Sz=%.12g, tau=%.12g\n", cons[DDD],cons[SRR],cons[LLL],cons[SZZ],cons[TAU]);
        printf("prim: rho=%.12g, vr=%.12g, vp=%.12g, vz=%.12g, P=%.12g\n", prim[RHO],prim[URR],prim[UPP],prim[UZZ],prim[PPP]);
        printf("cons2: rhostar=%.12g, Sr=%.12g, Sp=%.12g, Sz=%.12g, tau=%.12g\n", cons2[DDD],cons2[SRR],cons2[LLL],cons2[SZZ],cons2[TAU]);
    }
}
