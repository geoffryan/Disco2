#define CELL_PRIVATE_DEFS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../../Headers/Cell.h"
#include "../../Headers/Sim.h"
#include "../../Headers/Face.h"
#include "../../Headers/GravMass.h"
#include "../../Headers/header.h"

double blah = 0.0;

void cell_single_init_rad_dom(struct Cell *theCell, struct Sim *theSim,struct GravMass * theGravMasses,int i,int j,int k){
  
  double rm = sim_FacePos(theSim,i-1,R_DIR);
  double rp = sim_FacePos(theSim,i,R_DIR);
  double r = 0.5*(rm+rp);

  double xi = 0.005;
  double a_o_M = 100;
  double alpha = sim_EXPLICIT_VISCOSITY(theSim);
  double Gam = sim_GAMMALAW(theSim);

  double rho0 = 2./(9.*xi*alpha*Gam)*sqrt(a_o_M);
  double   P0 = 8.*xi/(9.*alpha*Gam)*sqrt(a_o_M);

  double rho    = rho0*pow(r,1.5);
  double P      = P0*pow(r,-1.5);
  double o2  = 1./r/r/r - 3./2.*P/rho/r/r;// - 20.*alpha*Gam*xi*xi*xi*pow(r,-4.5);//6.*xi*xi*pow(r,-5.);
  double omega = sqrt(o2);
  double vr     = -1.5*alpha*Gam*(P/rho)*sqrt(r) - blah*pow(r,-4.5) ;

  theCell->prim[RHO] = rho;
  theCell->prim[PPP] = P;
  theCell->prim[URR] = vr;
  theCell->prim[UPP] = omega-1./pow(r,1.5);
  theCell->prim[UZZ] = 0.0;
  theCell->wiph = pow(r,-.5);
  //printf("ERROR. cell_single_init_rad_dom isnt set up right now\n");
  //exit(0);
}

void cell_init_rad_dom(struct Cell ***theCells,struct Sim *theSim,struct GravMass * theGravMasses,struct MPIsetup * theMPIsetup) {

  double xi = 0.005;
  double a_o_M = 100;
  double alpha = sim_EXPLICIT_VISCOSITY(theSim);
  double Gam = sim_GAMMALAW(theSim);

  double rho0 = 2./(9.*xi*alpha*Gam)*sqrt(a_o_M);
  double   P0 = 8.*xi/(9.*alpha*Gam)*sqrt(a_o_M); 

  int i, j,k;
  for (k = 0; k < sim_N(theSim,Z_DIR); k++) {
    for (i = 0; i < sim_N(theSim,R_DIR); i++) {
      double rm = sim_FacePos(theSim,i-1,R_DIR);
      double rp = sim_FacePos(theSim,i,R_DIR);
      double r = 0.5*(rm+rp);

      for (j = 0; j < sim_N_p(theSim,i); j++) {

        double rho    = rho0*pow(r,1.5);
        double P      = P0*pow(r,-1.5);
        double o2  = 1./r/r/r - 3./2.*P/rho/r/r;// - 20.*alpha*Gam*xi*xi*xi*pow(r,-4.5);//6.*xi*xi*pow(r,-5.);
        double omega = sqrt(o2);
        double vr     = -1.5*alpha*Gam*(P/rho)*sqrt(r) - blah*pow(r,-4.5);

        theCells[k][i][j].prim[RHO] = rho;
        theCells[k][i][j].prim[PPP] = P;
        theCells[k][i][j].prim[URR] = vr;
        theCells[k][i][j].prim[UPP] = omega-sqrt(1.)/pow(r,1.5);
        theCells[k][i][j].prim[UZZ] = 0.0;
        theCells[k][i][j].wiph = pow(r,-1.5);
      }
    }
  }
}
