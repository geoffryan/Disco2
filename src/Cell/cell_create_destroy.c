#define CELL_PRIVATE_DEFS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../Headers/Cell.h"
#include "../Headers/Sim.h"
#include "../Headers/Face.h"
#include "../Headers/GravMass.h"
#include "../Headers/MPIsetup.h"
#include "../Headers/header.h"

struct Cell * cell_single_create(struct Sim * theSim){
  struct Cell *theCell = malloc(sizeof(struct Cell));
  theCell->prim = malloc(sim_NUM_Q(theSim)*sizeof(double));
  return(theCell);
}

void cell_single_destroy(struct Cell * theCell){
  free(theCell->prim);
  free(theCell);
}

struct Cell ***cell_create(struct Sim *theSim,struct MPIsetup * theMPIsetup){
  int NUM_Q = sim_NUM_Q(theSim);
  //count up the total number of cells
  int Ncells=0;
  int i,j,k,q;
  for (k=0; k<sim_N(theSim,Z_DIR); k++) {
    for (i=0; i<sim_N(theSim,R_DIR);i++){
      for(j = 0; j < sim_N_p(theSim,i); j++){
        ++Ncells;
      }
    }
  }

  //I wanted to allocate memory contiguously. It's not quite there yet.
  struct Cell ***theCells = (struct Cell ***)malloc(sizeof(struct Cell **)*sim_N(theSim,Z_DIR));
  theCells[0] = (struct Cell **)malloc(sizeof(struct Cell *)*sim_N(theSim,R_DIR)*sim_N(theSim,Z_DIR));
  theCells[0][0] = (struct Cell *)malloc(sizeof(struct Cell)*Ncells);

  int position = 0;
  for (k=0; k<sim_N(theSim,Z_DIR); k++) {
    theCells[k] = theCells[0]+k*sim_N(theSim,R_DIR);
    for (i=0; i<sim_N(theSim,R_DIR);i++){
      theCells[k][i] = theCells[0][0]+position;//sim_N_p(theSim,i)*(k*sim_N(theSim,R_DIR)+i);
      position += sim_N_p(theSim,i);
    }
  }
  Ncells=0;
  for (k=0; k<sim_N(theSim,Z_DIR); k++) {
    for (i=0; i<sim_N(theSim,R_DIR);i++){
      for(j = 0; j < sim_N_p(theSim,i); j++){
        theCells[k][i][j].prim = malloc(NUM_Q*sizeof(double));
        theCells[k][i][j].cons = malloc(NUM_Q*sizeof(double));
        theCells[k][i][j].RKcons = malloc(NUM_Q*sizeof(double));
        theCells[k][i][j].gradr = malloc(NUM_Q*sizeof(double));
        theCells[k][i][j].gradp = malloc(NUM_Q*sizeof(double));
        theCells[k][i][j].gradz = malloc(NUM_Q*sizeof(double));
        ++Ncells;
      }
    }
  }

  //  initialize
  srand(666+mpisetup_MyProc(theMPIsetup));

  // ghost indices
  int iNmg = sim_N(theSim,R_DIR)-sim_Nghost_max(theSim,R_DIR);
  int ig = sim_Nghost_min(theSim,R_DIR);
  int kNmg = sim_N(theSim,Z_DIR)-sim_Nghost_max(theSim,Z_DIR);
  int kg = sim_Nghost_min(theSim,Z_DIR);

  for(k = 0; k < sim_N(theSim,Z_DIR); k++){
    for(i = 0; i < sim_N(theSim,R_DIR); i++){
      //double tiph_0 = 0.0;
      double tiph_0 = 2.*M_PI*(double)rand()/(double)RAND_MAX;
      double dphi = 2.*M_PI/(double)sim_N_p(theSim,i);
      for(j = 0; j < sim_N_p(theSim,i); j++){
        double tiph = tiph_0 + (double)j*dphi;
        for(q=0; q < NUM_Q; q++)
        {
            theCells[k][i][j].prim[q] = 0.0;
            theCells[k][i][j].cons[q] = 0.0;
            theCells[k][i][j].RKcons[q] = 0.0;
            theCells[k][i][j].gradr[q] = 0.0;
            theCells[k][i][j].gradp[q] = 0.0;
            theCells[k][i][j].gradz[q] = 0.0;
        }
        theCells[k][i][j].wiph = 0.0;
        theCells[k][i][j].tiph = tiph;
        theCells[k][i][j].RKtiph = tiph;
        theCells[k][i][j].dphi = dphi;
        //Is this a ghost zone
        if((i<ig && !(mpisetup_dim_MyProc(theMPIsetup,0)==0
                        && sim_NoInnerBC(theSim)==1))
                || i>=iNmg 
                || k<kg || k>=kNmg)
            theCells[k][i][j].real = 0;
        else
            theCells[k][i][j].real = 1;
      }
    }
  }

  return theCells;
}

void cell_destroy(struct Cell ***theCells,struct Sim *theSim){
  int i,j,k;
  for (k=0; k<sim_N(theSim,Z_DIR); k++) {
    for (i=0; i<sim_N(theSim,R_DIR);i++){
      for(j = 0; j < sim_N_p(theSim,i); j++){
        free(theCells[k][i][j].gradr);
        free(theCells[k][i][j].gradp);
        free(theCells[k][i][j].gradz);
        free(theCells[k][i][j].RKcons);
        free(theCells[k][i][j].cons);
        free(theCells[k][i][j].prim);
      }
    }
  }

  free(theCells[0][0]);
  free(theCells[0]);
  free(theCells);
}

