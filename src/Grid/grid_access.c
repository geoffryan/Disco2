#define GRID_PRIVATE_DEFS
#include <stdlib.h>
#include <stdio.h>
#include "../Headers/Grid.h"
#include "../Headers/header.h"


int grid_N_p(struct Grid *theGrid,int i){
  return(theGrid->N_p[i]);
}
double grid_r_faces(struct Grid *theGrid,int i){
  return(theGrid->r_faces[i+1]);
}
double grid_z_faces(struct Grid *theGrid,int k){
  return(theGrid->z_faces[k+1]);
}
int grid_N_r(struct Grid *theGrid){
  return(theGrid->N_r);
}
int grid_N_z(struct Grid *theGrid){
  return(theGrid->N_z);
}
int grid_Ncells(struct Grid *theGrid){
  return(theGrid->Ncells);
}
int grid_Restart(struct Grid *theGrid){
  return(theGrid->Restart);
}
int grid_Ncells_global(struct Grid *theGrid){
  return(theGrid->Ncells_global);
}
int grid_offset(struct Grid *theGrid){
  return(theGrid->offset);
}
int grid_Nghost_rmin(struct Grid *theGrid){
  return(theGrid->Nghost_rmin);
}
int grid_Nghost_rmax(struct Grid *theGrid){
  return(theGrid->Nghost_rmax);
}
int grid_Nghost_zmin(struct Grid *theGrid){
  return(theGrid->Nghost_zmin);
}
int grid_Nghost_zmax(struct Grid *theGrid){
  return(theGrid->Nghost_zmax);
}
int grid_ng(struct Grid *theGrid){
  return(theGrid->ng);
}
int grid_N_z_global(struct Grid *theGrid){
  return(theGrid->N_z_global);
}
int grid_NUM_Q(struct Grid *theGrid){
  return(theGrid->NUM_Q);
}
int grid_MOVE_CELLS(struct Grid *theGrid){
  return(theGrid->MOVE_CELLS);
}
int grid_NumGravMass(struct Grid *theGrid){
  return(theGrid->NumGravMass);
}
double grid_GAMMALAW(struct Grid *theGrid){
  return(theGrid->GAMMALAW);
}
int grid_INCLUDE_VISCOSITY(struct Grid *theGrid) {
  return(theGrid->INCLUDE_VISCOSITY);
}
double grid_EXPLICIT_VISCOSITY(struct Grid *theGrid){
  return(theGrid->EXPLICIT_VISCOSITY);
}
double grid_DIVB_CH(struct Grid *theGrid){
  return(theGrid->DIVB_CH);
}
double grid_DIVB_L(struct Grid *theGrid) {
  return(theGrid->DIVB_L);
}
double grid_CFL(struct Grid *theGrid){
  return(theGrid->CFL);
}
double grid_PLM(struct Grid *theGrid){
  return(theGrid->PLM);
}
int grid_POWELL(struct Grid *theGrid){
  return(theGrid->POWELL);
}
int grid_GRAV2D(struct Grid *theGrid){
  return(theGrid->GRAV2D);
}
double grid_G_EPS(struct Grid *theGrid){
  return(theGrid->G_EPS);
}
double grid_PHI_ORDER(struct Grid *theGrid){
  return(theGrid->PHI_ORDER);
}
double grid_RHO_FLOOR(struct Grid *theGrid) {
  return(theGrid->RHO_FLOOR);
}
double grid_CS_FLOOR(struct Grid *theGrid){
  return(theGrid->CS_FLOOR);
}
double grid_CS_CAP(struct Grid *theGrid){
  return(theGrid->CS_CAP);
}
double grid_VEL_CAP(struct Grid *theGrid) {
  return(theGrid->VEL_CAP);
}
double grid_get_T_MAX(struct Grid * theGrid){
  return(theGrid->T_MAX);
}
double grid_NUM_CHECKPOINTS(struct Grid * theGrid){
  return(theGrid->NUM_CHECKPOINTS);
}


