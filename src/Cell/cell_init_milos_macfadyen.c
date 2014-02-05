#define CELL_PRIVATE_DEFS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../Headers/Cell.h"
#include "../Headers/Sim.h"
#include "../Headers/Face.h"
#include "../Headers/GravMass.h"
#include "../Headers/header.h"

void cell_single_init_milos_macfadyen(struct Cell *theCell, struct Sim *theSim,int i,int j,int k){
  printf("ERROR. cell_single_init_mm isnt set up right now\n");
  exit(0);
}

void cell_init_milos_macfadyen(struct Cell ***theCells,struct Sim *theSim,struct MPIsetup * theMPIsetup) {
  double rs = 10.0;

  double rho0   = 1.0;
  double Mach   = 10.0;

  double delta_exp   = 3.0;
  double xi_exp = 2.0;

  double Gam = sim_GAMMALAW(theSim);
  double cp = 1./Mach;
  double fac = 0.0001;

  //printf("rbh1: %e, rbh2: %e, mbh1: %e, mbh2: %e\n",gravMass_r(theGravMasses,0),gravMass_r(theGravMasses,1),gravMass_M(theGravMasses,0),gravMass_M(theGravMasses,1));

  double Mtotal = 1.0;
  double sep = 1.0;
  double massratio = sim_MassRatio(theSim);
  double M0 = Mtotal/(1.+massratio);
  double M1 = Mtotal/(1.+1./massratio);
  double r0 = M1/Mtotal*sep;
  double r1 = M0/Mtotal*sep;

  double eps0 = sim_G_EPS(theSim)*r0;
  double eps1 = sim_G_EPS(theSim)*r1;
  
  double DISK_ALPHA = sim_EXPLICIT_VISCOSITY(theSim);

  int i, j,k;
  for (k = 0; k < sim_N(theSim,Z_DIR); k++) {
    for (i = 0; i < sim_N(theSim,R_DIR); i++) {
      double rm = sim_FacePos(theSim,i-1,R_DIR);
      double rp = sim_FacePos(theSim,i,R_DIR);
      double r = 0.5*(rm+rp);

      double cs;
      if (r<1.){
        cs = 1./Mach;
      }else{
        cs = sqrt(1./r)/Mach;
      }
      double xbh0 = r0;
      double xbh1 = -r1;
      double ybh0 = 0.0;
      double ybh1 = 0.0;

      for (j = 0; j < sim_N_p(theSim,i); j++) {

        double phi = theCells[k][i][j].tiph - 0.5*theCells[k][i][j].dphi;
        double xpos = r*cos(phi);
        double ypos = r*sin(phi);

        double dist_bh1 = sqrt((xpos-xbh0)*(xpos-xbh0)+(ypos-ybh0)*(ypos-ybh0));
        double dist_bh2 = sqrt((xpos-xbh1)*(xpos-xbh1)+(ypos-ybh1)*(ypos-ybh1));

        double n = sim_PHI_ORDER(theSim);
        double Pot1 = M0/pow( pow(dist_bh1,n) + pow(eps0,n) , 1./n );
        double Pot2 = M1/pow( pow(dist_bh2,n) + pow(eps1,n) , 1./n );

		double rho = rho0*( pow(rs/r,delta_exp) )*exp(-pow(rs/r,xi_exp) );
        double omega = 1./pow(r,1.5);
        omega *= 1.+3./4.*(sep*sep/r/r*(massratio/((1.+massratio)*(1.+massratio))));
        double O2 = omega*omega + cs*cs/r/r*( 2.*rs*rs/r/r - 3. ); // add derivative of Pressure term
        double vr;
        if (r<1.){
          omega = sqrt(O2)*pow(r,2.);
          vr = 0.0;
          //vr = (-3.0*DISK_ALPHA*(1.0/Mach)*(1.0/Mach)*(1.0-delta_exp+xi_exp*pow(1./rs,-xi_exp)))*pow(r,2.);
        }else{
          omega = sqrt(O2);
          //vr = 0.0;
          vr = -3.0/sqrt(r)*DISK_ALPHA*(1.0/Mach)*(1.0/Mach)*(1.0-delta_exp+xi_exp*pow(r/rs,-xi_exp));
        }
        /* 
        if (r<3.){
          omega = 0.0;//pow(r,-1.5);
        }
        */
        if( rho<sim_RHO_FLOOR(theSim) ) rho = sim_RHO_FLOOR(theSim);
        
        double Pp = cs*cs*rho/Gam;

        theCells[k][i][j].prim[RHO] = rho*exp(fac*(Pot1+Pot2)/cp/cp);
        theCells[k][i][j].prim[PPP] = Pp*exp(fac*(Pot1+Pot2)/cp/cp);
        theCells[k][i][j].prim[URR] = vr;
        theCells[k][i][j].prim[UPP] = omega;
        theCells[k][i][j].prim[UZZ] = 0.0;
        theCells[k][i][j].wiph = 0.0;
        theCells[k][i][j].divB = 0.0;
        theCells[k][i][j].GradPsi[0] = 0.0;
        theCells[k][i][j].GradPsi[1] = 0.0;
        theCells[k][i][j].GradPsi[2] = 0.0;
      }
    }
  }

}