#define RIEMANN_PRIVATE_DEFS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../Headers/Riemann.h"
#include "../Headers/Sim.h"
#include "../Headers/Cell.h"
#include "../Headers/Face.h"
#include "../Headers/Metric.h"
#include "../Headers/header.h"


// ********************************************************************************************
// WE REALLY SHOULD IMPROVE THE COMMENTING OF ALL OF THESE ROUTINES. 
// THERE IS SOME COMPLICATED STUFF HERE. 
// LETS CHOOSE A REFERENCE SUCH AS TORO AND IDENTIFY LINES OF CODE WITH EQUATIONS IN THE BOOK.
// ********************************************************************************************

// this routine is only called by riemann_set_vel.
// It is used to find various L/R quantities. 
// TODO: remove GAMMALAW
void LR_speed_gr(double *prim, int *n, double GAMMALAW, double *p_vn,
                double *p_cf2, double *m, double *Fm, double *p_e, 
                double *p_Fe, double *p_w, struct Metric *g)
{
    double P   = prim[PPP];
    double rho = prim[RHO];
    double v[3]  = {prim[URR], prim[UPP], prim[UZZ]};
    double vn  = v[0]*n[0] + v[1]*n[1] + v[2]*n[2];
    double rhoh = rho + GAMMALAW/(GAMMALAW-1.0)*P;
    double cf2  = GAMMALAW*P/rhoh;

    int i,j;
    double al = metric_lapse(g);
    double be[3];
    for(i=0; i<3; i++)
        be[i] = metric_shift_u(g, i);
    double bn = be[0]*n[0] + be[1]*n[1] + be[2]*n[2];

    double u0 = 1.0 / sqrt(-metric_g_dd(g,0,0) - 2.0*metric_dot3_u(g,be,v)
                    - metric_square3_u(g,v));
    double w = al * u0;
    double u[4] = {u0, u0*v[0], u0*v[1], u0*v[2]};
    double l[3];
    for(i=0; i<3; i++)
    {
        l[i] = 0.0;
        for(j=0; j<4; j++)
            l[i] += metric_g_dd(g,i+1,j)*u[j];
    }

    *p_vn = vn;
    *p_cf2 = cf2;
    *p_w = w;

    m[0] = rhoh*w*l[0];
    m[1] = rhoh*w*l[1];
    m[2] = rhoh*w*l[2];
    Fm[0] = m[0]*vn + n[0]*al*P;
    Fm[1] = m[1]*vn + n[1]*al*P;
    Fm[2] = m[2]*vn + n[2]*al*P;

    double e = rhoh*w*w-P;
    double Fe = e*vn + P*(vn+bn);
    
    *p_e = e;
    *p_Fe = Fe;
}

// Find velocities needed for the Riemann problem
void riemann_set_vel_gr(struct Riemann *theRiemann, struct Sim *theSim, double r, double GAMMALAW)
{
    int i, j, dir;
    double Sl, Sr, Sl1, Sr1, Sl2, Sr2, Ss;
    double al, be[3], bn, sig, ign, w, v[3], dv;
    struct Metric *g;

    g = metric_create(time_global, theRiemann->pos[R_DIR], theRiemann->pos[P_DIR], theRiemann->pos[Z_DIR], theSim);

    dir = -1;
    for(i=0; i<3; i++)
    {
        be[i] = metric_shift_u(g,i);
        if(theRiemann->n[i] != 0)
            dir = i;
    }
    al = metric_lapse(g);
    bn = be[dir];
    ign = metric_gamma_uu(g, dir, dir);

    double vnL, cf2L, mL[3], FmL[3], eL, FeL;
    LR_speed_gr(theRiemann->primL, theRiemann->n, GAMMALAW, &vnL, &cf2L, 
                mL, FmL, &eL, &FeL, &w, g);

    sig = cf2L/(w*w*(1.0-cf2L));
    dv = sqrt(sig*(1.0+sig)*al*al*ign - sig*(vnL+bn)*(vnL+bn));
    
    Sl1 = (vnL - sig*bn - dv) / (1.0+sig);
    Sr1 = (vnL - sig*bn + dv) / (1.0+sig);

    double vnR, cf2R, mR[3], FmR[3], eR, FeR;
    LR_speed_gr(theRiemann->primR, theRiemann->n, GAMMALAW, &vnR, &cf2R, 
                mR, FmR, &eR, &FeR, &w, g);

    sig = cf2R/(w*w*(1.0-cf2R));
    dv = sqrt(sig*(1.0+sig)*al*al*ign - sig*(vnR+bn)*(vnR+bn));

    Sl2 = (vnR - sig*bn - dv) / (1.0+sig);
    Sr2 = (vnR - sig*bn + dv) / (1.0+sig);

    if(Sl1 > Sl2)
        Sl = Sl2;
    else
        Sl = Sl1;
    if(Sr1 < Sr2)
        Sr = Sr2;
    else
        Sr = Sr1;

    //TODO: RUSANOV.  REMOVE THIS.
    /*
    if(fabs(Sl) > Sr)
    {
        Sl = -fabs(Sl);
        Sr = fabs(Sl);
    }
    else
    {
        Sl = -fabs(Sr);
        Sr = fabs(Sr);
    }
    */

    // Contact Wave Speed
    double UE = (Sr*eR - Sl*eL - FeR + FeL) / (Sr - Sl);
    double FE = ((Sr*FeL - Sl*FeR + Sl*Sr*(eR-eL)) / (Sr - Sl) - bn*UE) / al;

    double UM_hll[3], FM_hll[3];
    for(i=0; i<3; i++)
    {
        UM_hll[i] = (Sr*mR[i]-Sl*mL[i]-FmR[i]+FmL[i]) / (Sr-Sl);
        FM_hll[i] = ((Sr*FmL[i] - Sl*FmR[i] + Sl*Sr*(mR[i]-mL[i])) / (Sr - Sl)
                         - bn*UM_hll[i]) / al;
    }
    double UM = 0.0;
    double FM = 0.0;
    for(i=0; i<3; i++)
    {
        double igi = theRiemann->n[0] * metric_gamma_uu(g,i,0)
                    + theRiemann->n[1] * metric_gamma_uu(g,i,1)
                    + theRiemann->n[2] * metric_gamma_uu(g,i,2);
        UM += igi * UM_hll[i];
        FM += igi * FM_hll[i];
    }

    double A = FE;
    double B =-FM-ign*UE;
    double C = ign*UM;

    double SsS;

    if(fabs(4*A*C/(B*B)) < 1.0e-7)
        SsS = -C/B * (1.0 + A*C/(B*B) + 2*A*A*C*C/(B*B*B*B));
    else
        SsS = (-B - sqrt(B*B-4*A*C)) / (2*A);

    Ss = al*SsS-bn;

    //Fluxes in orthonormal basis
    if(dir == PDIRECTION)
    {
        Sl *= r;
        Sr *= r;
        Ss *= r;
    }

  theRiemann->Sl = Sl;
  theRiemann->Sr = Sr;
  theRiemann->Ss = Ss;

  metric_destroy(g);
}

//THIS FUNCTION ASSUMES theRiemann->n EQUALS SOME PERMUTATION OF (0,0,1).
//IT WILL NOT WORK FOR ARBITRARY n.
void riemann_set_flux_gr(struct Riemann *theRiemann, struct Sim *theSim, double GAMMALAW, int SetState)
{
    double r = theRiemann->pos[R_DIR];
    double *prim;
    double *F;
    
    if (SetState==LEFT)
    {
        prim = theRiemann->primL;
        F = theRiemann->FL;
    }
    else if (SetState==RIGHT)
    {
        prim = theRiemann->primR;
        F = theRiemann->FR;
    } 
    else
    {
        printf("ERROR: riemann_set_flux given unrecognized state.\n");
        exit(0);
    }

    int i,j;
    struct Metric *g;
    double a, b[3], sqrtg, U[4];
    double u0, u[4]; //u0 = u^0, u[i] = u_i
    double rho, Pp, v[3];
    double rhoh, vn, bn, hn, Un;

    //Get hydro primitives
    rho = prim[RHO];
    Pp  = prim[PPP];
    v[0]  = prim[URR];
    v[1]  = prim[UPP];
    v[2]  = prim[UZZ];

    //Get needed metric values
    g = metric_create(time_global, theRiemann->pos[R_DIR], theRiemann->pos[P_DIR], theRiemann->pos[Z_DIR], theSim);
    a = metric_lapse(g);
    for(i=0; i<3; i++)
        b[i] = metric_shift_u(g, i);
    sqrtg = metric_sqrtgamma(g)/r;
    for(i=0; i<4; i++)
        U[i] = metric_frame_U_u(g,i,theSim);

    //Check if interpolated velocity is superluminal
    if(-metric_g_dd(g,0,0) - 2*metric_dot3_u(g,b,v) - metric_square3_u(g,v) < 0)
    {
        printf("ERROR: Velocity too high in flux. r=%.12g, vr=%.12g, vp=%.12g, vz=%.12g\n", r, v[0], v[1], v[2]);

        //If velocity is superluminal, reduce to Lorentz factor 5, keeping
        //direction same in rest frame.
        
        double MAXW = 5;
        double V[3], V2, corr;
        //Calculate Eulerian velocity
        for(i=0; i<3; i++)
            V[i] = (v[i]+b[i])/a;
        V2 = metric_square3_u(g, V);
        //Correction factor.
        corr = sqrt((MAXW*MAXW-1.0)/(MAXW*MAXW*V2));
        //Reset velocity
        for(i=0; i<3; i++)
            v[i] = corr*v[i] - (1.0-corr)*b[i];

        for(i=0; i<3; i++)
            V[i] = (v[i]+b[i])/a;
        double newV2 = metric_square3_u(g, V);
        printf("   fix: badV2 = %.12g corr = %.12g, newV2 = %.12g\n", 
                V2, corr, newV2);
    }

    //Calculate 4-velocity
    u0 = 1.0 / sqrt(-metric_g_dd(g,0,0) - 2*metric_dot3_u(g,b,v) - metric_square3_u(g,v));
    u[0] = metric_g_dd(g,0,0) * u0 + u0*metric_dot3_u(g,b,v);
    for(i=1; i<4; i++)
    {
        u[i] = 0;
        for(j=0; j<3; j++)
            u[i] += metric_gamma_dd(g,i-1,j) * (v[j]+b[j]);
        u[i] *= u0;
    }

    //Calculate beta & v normal to face.
    vn = v[0]*theRiemann->n[0] + v[1]*theRiemann->n[1] + v[2]*theRiemann->n[2];
    bn = b[0]*theRiemann->n[0] + b[1]*theRiemann->n[1] + b[2]*theRiemann->n[2];
    Un = U[1]*theRiemann->n[0] + U[2]*theRiemann->n[1] + U[3]*theRiemann->n[2];
    if(theRiemann->n[1] == 1)
        hn = r;
    else
        hn = 1.0;
   
    rhoh = rho + GAMMALAW/(GAMMALAW-1.0)*Pp;
    
    //Fluxes
    F[DDD] = hn*sqrtg*a*u0 * rho*vn;
    F[SRR] = hn*sqrtg*a*(u0*rhoh * u[1]*vn + Pp*theRiemann->n[0]);
    F[LLL] = hn*sqrtg*a*(u0*rhoh * u[2]*vn + Pp*theRiemann->n[1]);
    F[SZZ] = hn*sqrtg*a*(u0*rhoh * u[3]*vn + Pp*theRiemann->n[2]);
    F[TAU] = hn*sqrtg*a*(u0*(-rhoh*(U[0]*u[0]+U[1]*u[1]+U[2]*u[2]+U[3]*u[3])
                            - rho)*vn - Un*Pp);
    //F[TAU] = hn*sqrtg*(a*u0*(a*u0*rhoh - rho)*vn + Pp*bn);

    //HLL Viscous Flux
    /*
    double *Fvisc = (double *) malloc(sim_NUM_Q(theSim) * sizeof(double));
    riemann_visc_flux_LR(theRiemann, theSim, SetState, Fvisc);
    F[DDD] += Fvisc[DDD];
    F[SRR] += Fvisc[SRR];
    F[LLL] += Fvisc[LLL];
    F[SZZ] += Fvisc[SZZ];
    F[TAU] += Fvisc[TAU];
    free(Fvisc);
    */

    //Passive Fluxes
    int q;
    for( q=sim_NUM_C(theSim) ; q<sim_NUM_Q(theSim) ; ++q )
        F[q] = prim[q]*F[DDD];

    for(q=0; q<sim_NUM_Q(theSim); q++)
        if(F[q] != F[q])
            printf("ERROR: r=%.12g Flux[%d] is NaN.\n", r, q);

    metric_destroy(g);
}

void riemann_set_star_hllc_gr(struct Riemann *theRiemann, struct Sim *theSim,
                                double GAMMALAW)
{
    int i,j;
    double r = theRiemann->pos[R_DIR];
    int *n = theRiemann->n;
    double *prim;
    double Sk;
    double *Uk;
    double *Fk;
    if (theRiemann->state==LEFTSTAR)
    {
        prim = theRiemann->primL;
        Sk = theRiemann->Sl;
        Uk = theRiemann->UL;
        Fk = theRiemann->FL;
    }
    else
    {
        prim = theRiemann->primR;
        Sk = theRiemann->Sr;
        Uk = theRiemann->UR;
        Fk = theRiemann->FR;
    }
    double Ss=theRiemann->Ss;
    double rho = prim[RHO];
    double v[3]  = {prim[URR], prim[UPP], prim[UZZ]};
    double Pp  = prim[PPP];

    struct Metric *g = metric_create(time_global, theRiemann->pos[R_DIR], 
                                    theRiemann->pos[P_DIR], 
                                    theRiemann->pos[Z_DIR], theSim);
    double al, be[3], sqrtg, U[4];
    al = metric_lapse(g);
    for(i=0; i<3; i++)
        be[i] = metric_shift_u(g, i);
    sqrtg = metric_sqrtgamma(g)/r;
    for(i=0; i<4; i++)
        U[i] = metric_frame_U_u(g,i,theSim);
    
    double u0 = 1.0 / sqrt(-metric_g_dd(g,0,0) - 2*metric_dot3_u(g, be, v)
                             - metric_square3_u(g, v));
    double u[4] = {u0, u0*v[0], u0*v[1], u0*v[2]};
    double uU = metric_dot4_u(g, u, U);
    double w = al*u0;

    double l[3]; // l[i] = u_i
    for(i=0; i<3; i++)
    {
        l[i] = 0.0;
        for(j=0; j<4; j++)
            l[i] += metric_g_dd(g,i+1,j)*u[j];
    }

    int dir = -1;
    for(i=0; i<3; i++)
        if(n[i] == 1)
            dir = i;
    
    double hn = 1.0*n[0] + r*n[1] + 1.0*n[2];
    double bn = be[dir];
    double vn = v[dir];
    double Un = U[dir+1];
    double ign = metric_gamma_uu(g, dir, dir);
    double uSn = u0*(vn + bn);

    double rhoh = rho + GAMMALAW/(GAMMALAW-1.0) * Pp;
    double mn = rhoh*w*uSn;

    double ss = Ss/hn;
    double sk = Sk/hn;

    // Q == F - s * U
    double qE = rhoh*w*w*(vn-sk) + Pp*(sk+bn);
    double qMn = mn*(vn-sk) + al*Pp*ign;

    double ssS = (ss+bn)/al;
    double skS = (sk+bn)/al;

    // P star!
    double Pstar = (qMn - ssS*qE) / (al*(ign - ssS*skS));

    double kappa = (vn - sk) / (ss - sk);
    double alpha1 = (Pp - Pstar) / (ss - sk);
    double alpha2 = (vn*Pp - ss*Pstar) / (ss - sk);

    double rhoe = Pp / (GAMMALAW - 1.0);
    double tau = -rhoe*uU*u0 - Pp*(u0*uU+U[0]) - rho*(uU+1.0)*u0;

    theRiemann->Ustar[DDD] = al*sqrtg * rho*u0 * kappa;
    theRiemann->Ustar[SRR] = al*sqrtg * (rhoh*u0*l[0] * kappa + alpha1 * n[0]);
    theRiemann->Ustar[LLL] = al*sqrtg * (rhoh*u0*l[1] * kappa + alpha1 * n[1]);
    theRiemann->Ustar[SZZ] = al*sqrtg * (rhoh*u0*l[2] * kappa + alpha1 * n[2]);
    theRiemann->Ustar[TAU] = al*sqrtg * (tau * kappa - Un * alpha1
                                            + U[0] * alpha2);

    int q;
    for(q = sim_NUM_C(theSim); q < sim_NUM_Q(theSim); q++)
        theRiemann->Ustar[q] = prim[q] * theRiemann->Ustar[DDD];

    //Now set Fstar
    for(q = 0; q < sim_NUM_Q(theSim); q++)
        theRiemann->Fstar[q] = Fk[q] + Sk * (theRiemann->Ustar[q] - Uk[q]);

    metric_destroy(g);
}
