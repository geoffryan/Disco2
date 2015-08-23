#define METRIC_PRIVATE_DEFS
#include <stdio.h>
#include <math.h>
#include "../Headers/Metric.h"
#include "../Headers/Sim.h"
#include "../Headers/header.h"


double metric_square3_u(struct Metric *g, double *vec)
{
    int i,j;
    double v2 = 0;
    for(i=0; i<3; i++)
        for(j=0; j<3; j++)
            v2 += metric_gamma_dd(g,i,j)*vec[i]*vec[j];
    return v2;
}

double metric_square3_d(struct Metric *g, double *vec)
{
    int i,j;
    double v2 = 0.0;
    for(i=0; i<3; i++)
        for(j=0; j<3; j++)
            v2 += metric_gamma_uu(g,i,j)*vec[i]*vec[j];
    return v2;
}

double metric_square4_u(struct Metric *g, double *vec)
{
    int i,j;
    double v2 = 0;
    for(i=0; i<4; i++)
        for(j=0; j<4; j++)
            v2 += metric_g_dd(g,i,j)*vec[i]*vec[j];
    return v2;
}

double metric_square4_d(struct Metric *g, double *vec)
{
    int i,j;
    double v2 = 0;
    for(i=0; i<4; i++)
        for(j=0; j<4; j++)
            v2 += metric_g_uu(g,i,j)*vec[i]*vec[j];
    return v2;
}

double metric_dot3_u(struct Metric *g, double *a, double *b)
{
    int i,j;
    double v2 = 0;
    for(i=0; i<3; i++)
        for(j=0; j<3; j++)
            v2 += metric_gamma_dd(g,i,j)*a[i]*b[j];
    return v2;
}

double metric_dot3_d(struct Metric *g, double *a, double *b)
{
    int i,j;
    double v2 = 0;
    for(i=0; i<3; i++)
        for(j=0; j<3; j++)
            v2 += metric_gamma_uu(g,i,j)*a[i]*b[j];
    return v2;
}

double metric_dot4_u(struct Metric *g, double *a, double *b)
{
    int i,j;
    double v2 = 0;
    for(i=0; i<4; i++)
        for(j=0; j<4; j++)
            v2 += metric_g_dd(g,i,j)*a[i]*b[j];
    return v2;
}

double metric_dot4_d(struct Metric *g, double *a, double *b)
{
    int i,j;
    double v2 = 0;
    for(i=0; i<4; i++)
        for(j=0; j<4; j++)
            v2 += metric_g_uu(g,i,j)*a[i]*b[j];
    return v2;
}

double metric_frame_U_u_euler(struct Metric *g, int mu, struct Sim *theSim)
{
    if(mu == 0)
        return 1.0/metric_lapse(g);
    else if (mu > 0 && mu < 4)
        return -metric_shift_u(g,mu-1) / metric_lapse(g);
    return 0.0;
}

double metric_frame_dU_du_euler(struct Metric *g, int mu, int nu, struct Sim *theSim)
{
    if(nu == 0)
    {
        double a = metric_lapse(g);
        return -metric_dlapse(g,mu) / (a*a);
    }
    else if (nu > 0 && nu < 4)
    {
        double a = metric_lapse(g);
        return (metric_dlapse(g,mu)*metric_shift_u(g,nu-1) - a*metric_dshift_u(g,mu,nu-1)) / (a*a);
    }
    return 0.0;
}

double metric_frame_U_u_kep(struct Metric *g, int mu, struct Sim *theSim)
{
    double M = sim_GravM(theSim);
    double r = g->x[1];

    double u0, vr, vp;

    if(sim_Metric(theSim) == SCHWARZSCHILD_SC)
    {
        vr = 0.0;

        if(r > 3.0*M)
            vp = exp(-0.5/(r/M-3.0)) * sqrt(M/(r*r*r));
        else
            vp = 0.0;

        u0 = 1.0/sqrt(1-2*M/r-vr*vr/(1-2*M/r)-r*r*vp*vp);
        if(mu == 0)
            return u0;
        else if(mu == 2)
            return u0*vp;
        return 0.0;
    }
    else if(sim_Metric(theSim) == SCHWARZSCHILD_KS)
    {
        vr = -2*M/(r + 2*M);

        if(r > 3.0*M)
            vp = exp(-0.5/(r/M-3.0)) * sqrt(M/(r*r*r));
        else
            vp = 0.0;

        u0 = 1.0/sqrt(1.0-2*M/r - 4*M/r*vr - (1.0+2*M/r)*vr*vr - r*r*vp*vp);
        if(mu == 0)
            return u0;
        else if(mu == 1)
            return u0*vr;
        else if(mu == 2)
            return u0*vp;
        return 0.0;
    }
    
    return 0.0;
}

double metric_frame_dU_du_kep(struct Metric *g, int mu, int nu, struct Sim *theSim)
{
    double M = sim_GravM(theSim);
    double r = g->x[1];

    double u0, du0, vr, dvr, vp, dvp;

    if(mu != 1)
        return 0.0;

    if(sim_Metric(theSim) == SCHWARZSCHILD_SC)
    {
        vr = 0.0;

        if(r > 3.0*M)
        {
            vp = exp(-0.5/(r/M-3.0)) * sqrt(M/(r*r*r));
            dvp = -vp * (27*M*M-19*M*r+3*r*r) / (2*(r-3*M)*(r-3*M));
        }
        else
        {
            vp = 0.0;
            dvp = 0.0;
        }

        u0 = 1.0/sqrt(1-2*M/r-r*r*vp*vp);
        du0 = -0.5*u0*u0*u0*(2*M/(r*r) - 2*r*vp*vp - 2*r*r*vp*dvp);

        if(nu == 0)
            return du0;
        else if(nu == 2)
            return du0*vp + u0*dvp;
        return 0.0;
    }
    else if(sim_Metric(theSim) == SCHWARZSCHILD_KS)
    {
        vr = -2*M/(r + 2*M);
        dvr = 2*M/((r+2*M)*(r+2*M));

        if(r > 3.0*M)
        {
            vp = exp(-0.5/(r/M-3.0)) * sqrt(M/(r*r*r));
            dvp = -vp * (27*M*M-19*M*r+3*r*r) / (2*(r-3*M)*(r-3*M));
        }
        else
        {
            vp = 0.0;
            dvp = 0.0;
        }

        u0 = 1.0/sqrt(1.0-2*M/r - 4*M/r*vr - (1.0+2*M/r)*vr*vr - r*r*vp*vp);
        du0 = -0.5*u0*u0*u0*(2*M/(r*r) + 4*M*vr/(r*r) - 4*M/r*dvr + 2*M*vr*vr/(r*r) - 4*M/r*vr*dvr - 2*r*vp*vp - 2*r*r*vp*dvp);

        if(nu == 0)
            return du0;
        else if(nu == 1)
            return du0*vr + u0*dvr;
        else if(nu == 2)
            return du0*vp + u0*dvp;
        return 0.0;
    }

    return 0.0;
}

double metric_frame_U_u_acc(struct Metric *g, int mu, struct Sim *theSim)
{
    // This frame has purely Keplerian velocity in the Kerr-Schild frame.
    // Since Kerr-Schild coordinates have a shift, this induces an (inwards)
    // radial velocity in the coordinate basis as well.

    double M = sim_GravM(theSim);
    double A = M*sim_GravA(theSim);
    double r = g->x[1];

    if(sim_Metric(theSim) == SCHWARZSCHILD_KS)
    {
        /*
        if(mu == 1)
            return -2*M/sqrt((r+2*M)*(r-M));
        else if(mu == 2)
            return sqrt(M/(r*r*(r-M)));
        else if (mu == 0)
            return sqrt((r+2*M)/(r-M));
        */
        if(mu == 1)
            return -2*M/r * sqrt((r+M)/(r+2*M));
        else if(mu == 2)
            return sqrt(M/(r*r*r));
        else if (mu == 0)
            return sqrt((1+2*M/r)*(1+M/r));
    }
    else if(sim_Metric(theSim) == SCHWARZSCHILD_SC)
    {
        if(mu == 1)
            return -2*M/sqrt((r+2*M)*(r-M));
        else if(mu == 2)
            return sqrt(M/(r*r*(r-M)));
        else if (mu == 0)
            return r*r / sqrt((r-2*M)*(r-2*M)*(r*r+M*r-2*M*M));
    }
    else if(sim_Metric(theSim) == KERR_KS)
    {
        double omk = sqrt(M/(r*r*r));
        double Wp = 1.0 / sqrt((1.0+(A+r)*omk) * (1.0+(A-r)*omk));

        if(mu == 1)
            return (-2*M/r * (1+A*omk) / sqrt(1+2*M/r) + A*omk) * Wp;
        else if(mu == 2)
            return omk * Wp;
        else if (mu == 0)
            return (1.0+A*omk) * sqrt(1.0+2*M/r) * Wp;
    }
    else if(sim_Metric(theSim) == SCHWARZSCHILD_KS_ADM)
    {
        if(mu == 1)
            return -2*M/sqrt((r+2*M)*(r-M));
        else if(mu == 2)
            return sqrt(M/(r*r*(r-M)));
        else if (mu == 0)
            return sqrt((r+2*M)/(r-M));
    }
    return 0.0;
}

double metric_frame_dU_du_acc(struct Metric *g, int mu, int nu, struct Sim *theSim)
{
    double M = sim_GravM(theSim);
    double A = M*sim_GravA(theSim);
    double r = g->x[1];

    if(mu != 1)
        return 0.0;
    
    if(sim_Metric(theSim) == SCHWARZSCHILD_KS)
    {
        
        // ur = -2*M/r * sqrt((r+M)/(r+2*M));
        // up = sqrt(M/(r*r*r))
        // u0 = sqrt((1+2*M/r)*(1+M/r))

        if(nu == 1)
        {
            double ur = -2*M/r * sqrt((r+M)/(r+2*M));
            return ur * ( -1.0/r + 0.5/(r+M) - 0.5/(r+2*M));
        }
        else if(nu == 2)
        {
            return -1.5 * sqrt(M/(r*r*r*r*r));
        }
        else if(nu == 0)
        {
            double  u0 = sqrt((1+2*M/r)*(1+M/r));
            return 0.5/u0 * (-3*M/(r*r) - 4*M*M/(r*r*r));
        }
    }
    else if(sim_Metric(theSim) == SCHWARZSCHILD_SC)
    {
        if(nu == 1)
        {
            double ur = -2*M/sqrt((r+2*M)*(r-M));
            return ur*ur*ur/(-8*M*M) * (2*r+M);
        }
        else if(nu == 2)
        {
            double up = sqrt(M/(r*r*(r-M)));
            return -0.5*up*up*up/M * (3*r*r-2*M*r);
        }
        else if(nu == 0)
        {
            double x = r/M;
            double y = sqrt(x*x + x - 2);
            return -0.5 * x * (-16 + 10*x + 3*x*x) / (M * (x-2)*(x-2) * y*y*y);
        }
    }
    else if(sim_Metric(theSim) == KERR_KS)
    {
        double omk = sqrt(M/(r*r*r));
        double domk = -1.5 * omk/r;
        double Wp = 1.0 / sqrt((1.0+(A+r)*omk) * (1.0+(A-r)*omk));
        double dWp = -0.5 * (2*(1.0+A*omk)*A*domk + M/(r*r)) * Wp*Wp*Wp;

        if(nu == 1)
        {
            return (2*M/(r*r) * (1+A*omk) / sqrt(1+2*M/r)
                    - 2*M/r * A*domk / sqrt(1+2*M/r)
                    - 2*M/r * (1+A*omk) * M/(r*r*pow(1+2*M/r,1.5))
                    + A*domk) * Wp
                    + (-2*M/r * (1+A*omk) / sqrt(1+2*M/r) + A*omk) * dWp;
        }
        else if(nu == 2)
            return domk * Wp + omk * dWp;
        else if (nu == 0)
            return A*domk * sqrt(1.0+2*M/r) * Wp
                    + (1.0+A*omk) * (-M/(r*r*sqrt(1.0+2*M/r))) * Wp
                    + (1.0+A*omk) * sqrt(1.0+2*M/r) * dWp;
    }
    if(sim_Metric(theSim) == SCHWARZSCHILD_KS_ADM)
    {
        if(nu == 1)
        {
            double ur = -2*M/sqrt((r+2*M)*(r-M));
            return ur*ur*ur/(-8*M*M) * (2*r+M);
        }
        else if(nu == 2)
        {
            double up = sqrt(M/(r*r*(r-M)));
            return -0.5*up*up*up/M * (3*r*r-2*M*r);
        }
        else if(nu == 0)
        {
            return -1.5*M*sqrt((r-M)/(r+2*M)) / ((r-M)*(r-M));
        }
    }

    return 0.0;
}

double metric_frame_U_u_geo(struct Metric *g, int mu, struct Sim *theSim)
{
    double M = sim_GravM(theSim);
    double r = g->x[1];

    if(sim_Metric(theSim) == SCHWARZSCHILD_KS)
    {
        double Risco = 6*M;
        if(mu == 0)
        {
            if(r > Risco)
                return sqrt(r/(r-3*M));
            else
            {
                double x = Risco/r - 1.0;
                return 2.0*(sqrt(2.0)*r - M*sqrt(x*x*x)) / (3.0*(r-2.0*M));
            }
        }
        if(mu == 1)
        {
            if(r > Risco)
                return 0.0;
            else
            {
                double x = Risco/r - 1.0;
                return -sqrt(x*x*x) / 3.0;
            }
        }
        if(mu == 2)
        {
            if(r > Risco)
                return sqrt(M/(r*r*(r-3*M)));
            else
                return 2.0*sqrt(3.0)*M/(r*r);
        }
    }
    else if(sim_Metric(theSim) == KERR_KS)
    {
        double a = sim_GravA(theSim);
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
            if(mu == 0)
                return u0;
            else if(mu == 1)
                return 0.0;
            else if(mu == 2)
            {
                double omk = sqrt(M/(r*r*r));
                return u0 * omk / (1.0 + a*M*omk);
            }
            else
                return 0.0;
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
            double A = metric_g_uu(g,1,1);
            double hB = metric_g_uu(g,0,1)*eps + metric_g_uu(g,1,2)*lll;
            double C = metric_g_uu(g,0,0)*eps*eps
                        + 2*metric_g_uu(g,0,2)*eps*lll
                        + metric_g_uu(g,2,2)*lll*lll + 1.0;
            double urd = (-hB - sqrt(hB*hB - A*C)) / A;

            if(mu == 0)
                return metric_g_uu(g,0,0)*eps + metric_g_uu(g,0,1)*urd
                        + metric_g_uu(g,0,2)*lll;
            else if(mu == 1)
                return metric_g_uu(g,1,0)*eps + metric_g_uu(g,1,1)*urd
                        + metric_g_uu(g,1,2)*lll;
            else if(mu == 2)
                return metric_g_uu(g,2,0)*eps + metric_g_uu(g,2,1)*urd
                        + metric_g_uu(g,2,2)*lll;
            else
                return 0.0;
        }
    }

    return 0.0;
}

double metric_frame_dU_du_geo(struct Metric *g, int mu, int nu, 
                                struct Sim *theSim)
{
    double M = sim_GravM(theSim);
    double r = g->x[1];

    if(sim_Metric(theSim) == SCHWARZSCHILD_KS)
    {
        double Risco = 6*M;

        if(mu != 1)
            return 0.0;

        if(nu == 0)
        {
            if(r > Risco)
            {
                double x = 1.0 - 3.0*M/r;
                return -1.5*M / (sqrt(x*x*x) * r*r);
            }
            else
            {
                double x = sqrt(Risco/r - 1.0);
                double y = M/r;
                return -2.0*M * (x*(18.0*y*y-15.0*y+1.0) + 2.0*sqrt(2.0))
                        / (3.0*(r-2.0*M)*(r-2.0*M));
            }
        }
        if(nu == 1)
        {
            if(r > Risco)
                return 0.0;
            else
            {
                double x = Risco/r - 1.0;
                double dx = -Risco/(r*r);
                return -0.5 * sqrt(x) * dx;
            }
        }
        if(nu == 2)
        {
            if(r > Risco)
            {
                double x = r-3.0*M;
                return -1.5 * (r-2.0*M) * sqrt(M/(x*x*x)) / (r*r);
            }
            else
                return -4.0*sqrt(3.0) * M / (r*r*r);
        }
    }
    else if(sim_Metric(theSim) == KERR_KS)
    {
        if(mu != 1)
            return 0.0;

        double a = sim_GravA(theSim);
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
            double omk = sqrt(M/(r*r*r));
            double u0 = (1.0 + a*M*omk) / sqrt(1.0 - 3.0*M/r + 2*a*M*omk);
            double du0 = -1.5*M/(r*r) * pow(1.0 - 3.0*M/r + 2*a*M*omk, -1.5) * 
                            (1.0 - 2*a*M*omk + a*a*M*M/(r*r));

            if(nu == 0)
                return du0;
            else if(nu == 1)
                return 0.0;
            else if(nu == 2)
            {
                double domk = -1.5 * omk / r;
                return du0 * omk / (1.0+a*M*omk) + u0 * domk / (1.0+a*M*omk)
                         - u0 * omk * a*M*domk / ((1.0+a*M*omk)*(1.0+a*M*omk));
            }
            else
                return 0.0;
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
            double A = metric_g_uu(g,1,1);
            double hB = metric_g_uu(g,0,1)*eps + metric_g_uu(g,1,2)*lll;
            double C = metric_g_uu(g,0,0)*eps*eps
                        + 2*metric_g_uu(g,0,2)*eps*lll
                        + metric_g_uu(g,2,2)*lll*lll + 1.0;
            double dA = metric_dg_uu(g,1,1,1);
            double dhB = metric_dg_uu(g,1,0,1)*eps + metric_dg_uu(g,1,1,2)*lll;
            double dC = metric_dg_uu(g,1,0,0)*eps*eps
                        + 2*metric_dg_uu(g,1,0,2)*eps*lll
                        + metric_dg_uu(g,1,2,2)*lll*lll;
            double urd = (-hB - sqrt(hB*hB - A*C)) / A;
            double durd = -(dhB*A-hB*dA)/(A*A) - (A*hB*dhB-hB*hB*dA+0.5*A*C*dA
                            -0.5*A*A*dC) / (sqrt(hB*hB-A*C)*A*A);

            if(nu == 0)
                return metric_dg_uu(g,1,0,0)*eps + metric_dg_uu(g,1,0,2)*lll
                    + metric_dg_uu(g,1,0,1)*urd + metric_g_uu(g,0,1)*durd;
            else if(nu == 1)
                return metric_dg_uu(g,1,1,0)*eps + metric_dg_uu(g,1,1,2)*lll
                    + metric_dg_uu(g,1,1,1)*urd + metric_g_uu(g,1,1)*durd;
            else if(nu == 2)
                return metric_dg_uu(g,1,2,0)*eps + metric_dg_uu(g,1,2,2)*lll
                    + metric_dg_uu(g,1,2,1)*urd + metric_g_uu(g,2,1)*durd;
            else
                return 0.0;
        }
    }

    return 0.0;
}

int metric_fixV(struct Metric *g, double *v, double maxW)
{
    //If velocity is superluminal, reduce to Lorentz factor 5, keeping
    //direction same in rest frame.

    int i, err;
    double a, b[3], V[3], V2, corr;

    a = metric_lapse(g);
    b[0] = metric_shift_u(g, 0);
    b[1] = metric_shift_u(g, 1);
    b[2] = metric_shift_u(g, 2);

    //Calculate Eulerian velocity.
    V[0] = (v[0]+b[0])/a;
    V[1] = (v[1]+b[1])/a;
    V[2] = (v[2]+b[2])/a;
    V2 = metric_square3_u(g,V);

    if(V2 < 1.0)
        err = 0;
    else
    {
        //Correction factor.
        corr = sqrt((maxW*maxW-1.0)/(maxW*maxW*V2));
        //Reset velocity
        for(i=0; i<3; i++)
            v[i] = corr*v[i] - (1.0-corr)*b[i];

        err = 1;
    }

    return err;
}
