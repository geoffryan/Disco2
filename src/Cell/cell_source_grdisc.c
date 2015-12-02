#define CELL_PRIVATE_DEFS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../Headers/Cell.h"
#include "../Headers/EOS.h"
#include "../Headers/Sim.h"
#include "../Headers/Face.h"
#include "../Headers/GravMass.h"
#include "../Headers/Metric.h"
#include "../Headers/header.h"

// Local Functions
void cell_cool_integrateT_grdisc_num(double *prim, double *dcons, double dt, 
                                double u0, double pos[], double M,
                                struct Sim *theSim);
double logT_prime(double logT, double p[], double r, double M, double u0,
                    double sigma, struct Sim *theSim);
int rho_solve(double p[], double sigma, double r, double M, double u0, 
                    struct Sim *theSim);

//Add source terms to theCells.
void cell_add_src_grdisc( struct Cell ***theCells, struct Sim *theSim, 
                        struct GravMass *theGravMasses, double dt)
{
    int i,j,k;
  
    FILE *sourcefile;
    if(PRINTTOOMUCH)
        sourcefile = fopen("source.out","w");
  
    for(k = 0; k < sim_N(theSim,Z_DIR); k++)
    {
        double zm = sim_FacePos(theSim,k-1,Z_DIR);
        double zp = sim_FacePos(theSim,k,Z_DIR);
    
        for(i = 0; i < sim_N(theSim,R_DIR); i++)
        {
            double rm = sim_FacePos(theSim,i-1,R_DIR);
            double rp = sim_FacePos(theSim,i,R_DIR);

            for(j = 0; j < sim_N_p(theSim,i); j++)
            {
                struct Cell * c = &(theCells[k][i][j]);
                double phi = c->tiph-.5*c->dphi;
                double dphi = c->dphi;

                double r   = 0.5*(rp+rm);
                double dz = zp-zm;
                double z  = 0.5*(zp+zm);
                double dV = dphi*0.5*(rp*rp-rm*rm)*dz;

       
                int mu, nu, la;
                double a, b[3], sqrtg, u[4], u_d[4], U[4], s, sk;
                double Pp, eps, rhoh, cs2, M, H;
                double rho, v[3];
                struct Metric *g;
                double dv[12];
                double qdot, visc[16], viscm[16], alpha;
                double T[16], Tm[16];
                
                g = metric_create(time_global, r, phi, z, theSim);
                a = metric_lapse(g);
                for(mu=0; mu<3; mu++)
                    b[mu] = metric_shift_u(g,mu);
                sqrtg = metric_sqrtgamma(g) / r;
                for(mu=0; mu<4; mu++)
                    U[mu] = metric_frame_U_u(g, mu, theSim);

                rho = c->prim[RHO];
                v[0] = c->prim[URR];
                v[1] = c->prim[UPP];
                v[2] = c->prim[UZZ];
                //Contravariant Four-Velocity u[i] = u^i
                u[0] = 1.0 / sqrt(-metric_g_dd(g,0,0) - 2*metric_dot3_u(g,b,v)
                                - metric_square3_u(g,v));
                u[1] = u[0] * c->prim[URR];
                u[2] = u[0] * c->prim[UPP];
                u[3] = u[0] * c->prim[UZZ];
                //Covariant Four-Velocity u_d[i] = u_i
                for(mu=0; mu<4; mu++)
                {
                    u_d[mu] = 0.0;
                    for(nu=0; nu<4; nu++)
                        u_d[mu] += metric_g_dd(g,mu,nu) * u[nu];
                }

                //Thermal Quantities
                Pp = eos_ppp(c->prim, theSim);
                eps = eos_eps(c->prim, theSim);
                cs2 = eos_cs2(c->prim, theSim);
                rhoh = rho + rho*eps + Pp;
                M = sim_GravM(theSim);

                if(sim_Metric(theSim) == KERR_KS)
                {
                    double A = M*sim_GravA(theSim);

                    H = r*r * sqrt(Pp / (rhoh * (u_d[2]*u_d[2]
                                    -A*A * (u_d[0]*u_d[0]-1.0) ) ) );
                }
                else
                    H = sqrt(r*r*r*Pp / (rhoh*M)) / u[0];
               
                //Viscous Terms
                alpha = sim_AlphaVisc(theSim);
                for(mu=0; mu<16; mu++)
                {
                    visc[mu] = 0.0;
                    viscm[mu] = 0.0;
                }
                
                if (alpha > 0)
                    alpha *= sqrt(cs2) * H * rho;
                else
                    alpha = -alpha * rho;

                //Velocity gradients
                dv[0] = 0.0;  dv[1] = 0.0;  dv[2] = 0.0;
                dv[3] = cell_gradr(c, URR);  
                dv[4] = cell_gradr(c, UPP);  
                dv[5] = cell_gradr(c, UZZ);
                dv[6] = cell_gradp(c, URR);  
                dv[7] = cell_gradp(c, UPP);  
                dv[8] = cell_gradp(c, UZZ);
                dv[9] = cell_gradz(c, URR);  
                dv[10] = cell_gradz(c, UPP);  
                dv[11] = cell_gradz(c, UZZ);
               
                //Shear Tensor!
                metric_shear_uu(g, v, dv, visc, theSim);

                //If 2-D, no z-shear
                if(sim_N(theSim,Z_DIR)==1)
                    for(mu=0; mu<4; mu++)
                    {
                        visc[4*mu+3] = 0.0;
                        visc[4*3+mu] = 0.0;
                    }
            
                for(mu=0; mu<16; mu++)
                    visc[mu] *= -alpha;
                
                //viscm[4*mu+nu] = visc^mu_nu
                for(mu=0; mu<4; mu++)
                    for(nu=0; nu<4; nu++)
                        for(la=0; la<4; la++)
                            viscm[4*mu+nu] += metric_g_dd(g,nu,la)*visc[4*mu+la];

                // Stress-Energy Tensor T.
                // (2,0) T[4*mu+nu] = T^{mu nu}
                // (1,1) Tm[4*mu+nu] = T^mu_nu
                for(mu=0; mu<4; mu++)
                {
                    for(nu=0; nu<4; nu++)
                    {
                        T[4*mu+nu] = rhoh*u[mu]*u[nu]
                                    + Pp*metric_g_uu(g,mu,nu)
                                    + visc[4*mu+nu];
                        Tm[4*mu+nu] = rhoh*u[mu]*u_d[nu]
                                    + viscm[4*mu+nu];
                    }
                    Tm[4*mu+mu] += Pp;
                }

                //Momentum sources and contribution to energy source
                s = 0;
                int max_dim = 4;
                if(sim_N(theSim,Z_DIR)==1)
                    max_dim = 3;
       
                for(la=0; la<max_dim; la++)
                    if(!metric_killcoord(g,la))
                    {
                        sk = 0;
                        for(mu=0; mu<max_dim; mu++)
                            for(nu=0; nu<max_dim; nu++)
                                sk += 0.5*T[4*mu+nu]*metric_dg_dd(g,la,mu,nu);

                        if(la == 1)
                        {
                            c->cons[SRR] += dt*dV*sqrtg*a * sk * H;
                            if(PRINTTOOMUCH)
                            {
                                printf("SRR source: (%d,%d,%d): r=%.12g, dV=%.12g, s = %.12g, S = %.12g\n",i,j,k,r,dV,a*sqrtg*sk,dt*dV*sqrtg*a * sk);
                                //fprintf(sourcefile, "(%d,%d,%d): r=%.12g, Fg=%.12g, Fc=%.12g, P/r=%.12g, F=%.12g\n", 
                                //        i,j,k,r,0.5*rhoh*u[0]*u[0]*metric_dg_dd(g,1,0,0), 0.5*rhoh*u[2]*u[2]*metric_dg_dd(g,1,2,2), 
                                //        0.5*metric_g_uu(g,2,2)*Pp*metric_dg_dd(g,1,2,2), sk);
                            }
                        }
                        else if(la == 2)
                        {
                            c->cons[LLL] += dt*dV*sqrtg*a * sk * H;
                        }
                        else if(la == 3)
                        {
                            c->cons[SZZ] += dt*dV*sqrtg*a * sk * H;
                            if(PRINTTOOMUCH)
                            {
                                printf("SZZ source: (%d,%d,%d): r=%.12f, dV=%.12f, s = %.12f, S = %.12f\n",i,j,k,r,dV,a*sqrtg*sk,dt*dV*sqrtg*a * sk);
                            }
                        }
                        s -= U[la]*sk;
                    }

                //Remaining energy sources
                for(mu=0; mu<max_dim; mu++)
                    for(nu=0; nu<max_dim; nu++)
                        s -= Tm[4*mu+nu] * metric_frame_dU_du(g,mu,nu,theSim);
                
                if(PRINTTOOMUCH)
                {
                    printf("TAU source: (%d,%d,%d): r=%.12g, dV=%.12g, s = %.12g, S = %.12g\n",i,j,k,r,dV,a*sqrtg*s,dt*dV*sqrtg*a * s);
                }

                c->cons[TAU] += dt*dV*sqrtg*a * s * H;

                //Cooling
                //
                //Cooling should never (?) change the sign of the momenta,
                //if it will, instead reduce the momenta by 0.9.  These lines
                //should be irrelevant because of the luminosity-limited
                //timestep.
                //
                //It is assumed eos_cool() returns the column-integrated
                //cooling rate, hence these don't include explicit factors
                //of H.
                
                double pos[3];
                pos[R_DIR] = r;
                pos[P_DIR] = phi;
                pos[Z_DIR] = z;
                int NUMQ = sim_NUM_Q(theSim);
                double dcons_cool[NUMQ];

                cell_cool_integrateT_grdisc_num(c->prim, dcons_cool, dt, u[0], 
                                                pos, M, theSim);

                c->cons[SRR] += dcons_cool[SRR] * dV;
                c->cons[LLL] += dcons_cool[LLL] * dV;
                c->cons[SZZ] += dcons_cool[SZZ] * dV;
                c->cons[TAU] += dcons_cool[TAU] * dV;

                /*
                qdot = eos_cool(c->prim, H, theSim);

                if(fabs(c->cons[SRR]) < fabs(dt*dV*sqrtg*a* qdot*u_d[1]))
                    c->cons[SRR] /= 1.1;
                else
                    c->cons[SRR] -= dt*dV*sqrtg*a* qdot*u_d[1];
                if(fabs(c->cons[LLL]) <  fabs(dt*dV*sqrtg*a* qdot*u_d[2]))
                    c->cons[LLL] /= 1.1;
                else
                    c->cons[LLL] -= dt*dV*sqrtg*a* qdot*u_d[2];
                if( fabs(c->cons[SZZ]) < fabs(dt*dV*sqrtg*a* qdot*u_d[3]))
                    c->cons[SZZ] /= 1.1;
                else
                    c->cons[SZZ] -= dt*dV*sqrtg*a* qdot*u_d[3];
                if(fabs(c->cons[TAU]) < fabs(dt*dV*sqrtg*a* qdot
                        * (u_d[0]*U[0]+u_d[1]*U[1]+u_d[2]*U[2]+u_d[3]*U[3])))
                {
                    c->cons[TAU] /= 1.1;
                    printf("That's pretty cool!\n");
                }
                else
                    c->cons[TAU] += dt*dV*sqrtg*a* qdot * (u_d[0]*U[0]+u_d[1]*U[1]+u_d[2]*U[2]+u_d[3]*U[3]); 
                */
                if(PRINTTOOMUCH)
                {
                    printf("SRR cooling: (%d,%d,%d): r=%.12g, dV=%.12g, q = %.12g, Q = %.12g\n",i,j,k,r,dV,-a*sqrtg* qdot*u_d[1],-dt*dV*sqrtg*a *qdot*u_d[1]);
                    printf("LLL cooling: (%d,%d,%d): r=%.12g, dV=%.12g, q = %.12g, Q = %.12g\n",i,j,k,r,dV,-a*sqrtg* qdot*u_d[2],-dt*dV*sqrtg*a *qdot*u_d[2]);
                    printf("TAU cooling: (%d,%d,%d): r=%.12g, dV=%.12g, q = %.12g, Q = %.12g\n",i,j,k,r,dV,a*sqrtg* qdot * (u_d[0]*U[0]+u_d[1]*U[1]+u_d[2]*U[2]+u_d[3]*U[3]),dt*dV*sqrtg*a *qdot * (u_d[0]*U[0]+u_d[1]*U[1]+u_d[2]*U[2]+u_d[3]*U[3]));
                }

                metric_destroy(g);
            }
        }
    }

    if(PRINTTOOMUCH)
        fclose(sourcefile);
}

void cell_cool_integrateT_grdisc_num(double *prim, double *dcons, double dt, 
                                double u0, double pos[], double M, 
                                struct Sim *theSim)
{
    // Add source terms by integrating temperature over timestep subject only
    // to cooling.  Assume velocities and surface density are constant.

    int i;
    int NUMQ = sim_NUM_Q(theSim);
    double res = 1.0e-3;

    double r = pos[R_DIR];

    double p[NUMQ];
    p[RHO] = prim[RHO];
    p[TTT] = prim[TTT];
    p[URR] = prim[URR];
    p[UPP] = prim[UPP];
    p[UZZ] = prim[UZZ];
    double logT = log(p[TTT]);
    double rho = eos_rho(p, theSim);
    double eps = eos_eps(p, theSim);
    double P   = eos_ppp(p, theSim);
    double H = sqrt(r*r*r*P / (M*(rho+rho*eps+P))) / u0;

    double sigma = rho * H;

    if(sim_CoolingType(theSim) == COOL_NONE) {}
    /*else if(sim_CoolingType(theSim) == COOL_ISOTHERM)
    {
        logT = log(sim_CoolPar1(theSim));
    }
    else if(sim_CoolingType(theSim) == COOL_BB_ES)
    {
        double GAM = sim_GAMMALAW(theSim);
        double qdot = eos_cool(prim, H, theSim);

        double T0 = prim[TTT];
        double T1;

        T1 = pow(1.0 + 3*(GAM-1)*qdot/(prim[RHO]*u0*H*T0)*dt, -1.0/3.0) * T0;

        logT = log(T1);
    }*/
    else
    {
        double t = 0;

        // Cool using adaptive Forward Euler or RK4.
        // TODO: Verify convergence, etc.
        //
        i = 0;
        while(t < dt)
        {
            double logTprime = logT_prime(logT, p, r, M, u0, sigma, theSim);

            double step = res / logTprime;
            step = step < dt-t ? step : dt-t;
            
            //FE
            //logT += -logTprime * step;
            
            //RK4
            
            double logT1 = logT - 0.5*step*logTprime;
            double logTp2 = logT_prime(logT1, p, r, M, u0, sigma, theSim);
            double logT2 = logT - 0.5*step*logTp2;
            double logTp3 = logT_prime(logT2, p, r, M, u0, sigma, theSim);
            double logT3 = logT - step*logTp3;
            double logTp4 = logT_prime(logT3, p, r, M, u0, sigma, theSim);
            logT += -step*(logTprime + 2*logTp2 + 2*logTp3 + logTp4)/6.0;
            

            t += step;
            i++;
        }
        
        if(i>100)
            printf("   Cell at (%.12lg, %.12lg, %.12lg) cooled in %d steps.\n",
                    pos[R_DIR], pos[P_DIR], pos[Z_DIR], i);
    }

    /*
    FILE *Tfile = fopen("cool.txt", "a");
    fprintf(Tfile, "%.16lg %.16lg %.16lg\n", pos[R_DIR], prim[TTT], exp(logT));
    fclose(Tfile);
    */

    double T = exp(logT);
    p[TTT] = T;
    int err = rho_solve(p, sigma, r, M, u0, theSim);

    if((T-prim[TTT])/prim[TTT] > 1.0e-10)
        printf("WAT: x %.12lg T0 %.24lg T1 %.24lg\n", (T-prim[TTT])/prim[TTT], prim[TTT], T);

    //printf("%.12lg %.12lg\n", prim[PPP]/prim[RHO], T);

    double *cons0 = (double *)malloc(NUMQ * sizeof(double));
    double *cons1 = (double *)malloc(NUMQ * sizeof(double));
    cell_prim2cons(prim, cons0, pos, 1.0, theSim);
    cell_prim2cons(p,    cons1, pos, 1.0, theSim);

    //printf("%.12lg\n", cons1[DDD]-cons0[DDD]);

    dcons[DDD] = 0.0;
    dcons[SRR] = cons1[SRR]-cons0[SRR];
    dcons[LLL] = cons1[LLL]-cons0[LLL];
    dcons[SZZ] = cons1[SZZ]-cons0[SZZ];
    dcons[TAU] = cons1[TAU]-cons0[TAU];
    for(i=5; i<NUMQ; i++)
        dcons[i] = 0.0;

    if(dcons[TAU]/cons0[TAU] > 1.0e-10)
        printf("WAT WAT %.12lg\n", dcons[TAU]/cons0[TAU]);

    free(cons0);
    free(cons1);
}

double logT_prime(double logT, double p[], double r, double M, double u0,
                    double sigma, struct Sim *theSim)
{
        double T = exp(logT);
        p[TTT] = T;
        
        double rho, P, eps, dPdp, dPdT, dedp, dedT, dedT_sig;
        int err = rho_solve(p, sigma, r, M, u0, theSim);

        rho = p[RHO];
        P = eos_ppp(p, theSim);
        eps = eos_eps(p, theSim);
        dPdp = eos_dpppdrho(p, theSim);
        dPdT = eos_dpppdttt(p, theSim);
        dedp = eos_depsdrho(p, theSim);
        dedT = eos_depsdttt(p, theSim);
        double rhoh = rho + rho*eps + P;
        double h = rhoh / rho;
        double height = sqrt(P*r*r*r/(rhoh*M))/u0;

        dedT_sig = dedT - 0.5*dedp * ((1.0+eps)*rho/P*dPdT - rho*dedT)/h
                         / (1 + 0.5*((1+eps)*(-1+rho/P*dPdp) - rho*dedp)/h);

        double qdot = eos_cool(p, height, theSim);
        double logTprime = qdot / (rho * height * u0 * dedT_sig * T);

        return logTprime;
}

int rho_solve(double p[], double sigma, double r, double M, double u0, 
                struct Sim *theSim)
{
    /*
     * A Newton-Raphson solver to find the central density 'rho' for a
     * given surface density 'sigma' and temperature (given in prim array 'p').
     * The value of p[RHO] is updated. Returns non-zero if an error occurs.
     */

    double rho0, rho1, rho;
    double rhoH, drhoH;
    double c = sqrt(r*r*r/M)/u0;
    int err = 0;
    double tol = 1.0e-12;

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

    p[RHO] = rho1;
    
    return err;
}
