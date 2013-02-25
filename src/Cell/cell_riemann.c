#define CELL_PRIVATE_DEFS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../Headers/Cell.h"
#include "../Headers/Grid.h"
#include "../Headers/Face.h"
#include "../Headers/GravMass.h"
#include "../Headers/header.h"





void vel( double * , double * , double * , double * , double * , double * , double , double * );
void prim2cons_local( double * , double * , double , double );
void getUstar( double * , double * , double , double , double , double * , double * );
void flux( double * , double * , double , double * );

void cell_riemann_p( struct Cell * cL , struct Cell * cR, double dA , double dt , double r ){

  double primL[NUM_Q];
  double primR[NUM_Q];

  int q;
  for( q=0 ; q<NUM_Q ; ++q ){
    primL[q] = cell_single_get_prims(cL)[q] 
      + cell_single_get_gradp(cL)[q]*cell_single_dphi(cL);
    primR[q] = cell_single_get_prims(cR)[q] 
      + cell_single_get_gradp(cR)[q]*cell_single_dphi(cR);
  }


  double Sl,Sr,Ss;
  double n[3];
  n[0] = 0.0;   n[1] = 1.0;   n[2] = 0.0;

  double Bpack[6];
  vel( primL , primR , &Sl , &Sr , &Ss , n , r , Bpack );

  double Fk[NUM_Q];
  double Uk[NUM_Q];

  double Flux[NUM_Q];

  double Bp_face = 0.5*(primL[BPP]+primR[BPP]);
  double Psi_face = 0.5*(primL[PSI]+primR[PSI]);

  if( MOVE_CELLS == C_WRIEMANN ) cell_add_wiph(cL,Ss);
  double w = cell_single_wiph(cL);

  if( w < Sl ){
    flux( primL , Fk , r , n );
    prim2cons_local( primL , Uk , r , 1.0 );
    for( q=0 ; q<NUM_Q ; ++q ){
      Flux[q] = Fk[q] - w*Uk[q];
    }
  }else if( w > Sr ){
    flux( primR , Fk , r , n );
    prim2cons_local( primR , Uk , r , 1.0 );
    for( q=0 ; q<NUM_Q ; ++q ){
      Flux[q] = Fk[q] - w*Uk[q];
    }
  }else{
    double Ustar[NUM_Q];
    if( w < Ss ){
      prim2cons_local( primL , Uk , r , 1.0 );
      getUstar( primL , Ustar , r , Sl , Ss , n , Bpack );
      flux( primL , Fk , r , n );

      for( q=0 ; q<NUM_Q ; ++q ){
        Flux[q] = Fk[q] + Sl*( Ustar[q] - Uk[q] ) - w*Ustar[q];
      }
    }else{
      prim2cons_local( primR , Uk , r , 1.0 );
      getUstar( primR , Ustar , r , Sr , Ss , n , Bpack );
      flux( primR , Fk , r , n );

      for( q=0 ; q<NUM_Q ; ++q ){
        Flux[q] = Fk[q] + Sr*( Ustar[q] - Uk[q] ) - w*Ustar[q];
      }
    }
  }

  /*
     if( INCLUDE_VISCOSITY == 1 ){
     double VFlux[NUM_Q];
     double AvgPrim[NUM_Q];
     double Gprim[NUM_Q];
     for( q=0 ; q<NUM_Q ; ++q ){
     AvgPrim[q] = .5*(primL[q]+primR[q]);
     Gprim[q] = .5*(cL->gradp[q]+cR->gradp[q])/r;
     VFlux[q] = 0.0;
     }
     visc_flux( AvgPrim , Gprim , VFlux , r , n );
     for( q=0 ; q<NUM_Q ; ++q ){
     Flux[q] += VFlux[q];
     }
     }
     */

  for( q=0 ; q<NUM_Q ; ++q ){
    cell_add_cons(cL,q,-dt*dA*Flux[q]);
    cell_add_cons(cR,q,dt*dA*Flux[q]);
  }
  cell_add_divB(cL,dA*Bp_face);
  cell_add_divB(cR,-dA*Bp_face);
  cell_add_GradPsi(cL,1,Psi_face/r);
  cell_add_GradPsi(cR,1,-Psi_face/r);

}


