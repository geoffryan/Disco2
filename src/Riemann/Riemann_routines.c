#define RIEMANN_PRIVATE_DEFS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../Headers/Riemann.h"
#include "../Headers/Grid.h"
#include "../Headers/Cell.h"
#include "../Headers/Face.h"
#include "../Headers/header.h"

void riemann_set_vel(struct Riemann * theRiemann,double *n,double r,double *Bpack,double GAMMALAW,double DIVB_CH){
  double wp = 0.0*n[1];
  double ch = DIVB_CH;

  double L_Mins, L_Plus, L_Star;

  double * prim1 = theRiemann->primL;
  double * prim2 = theRiemann->primR;

  double P1   = prim1[PPP];
  double rho1 = prim1[RHO];

  double vr1  =   prim1[URR];
  double vp1  = r*prim1[UPP];
  double vz1  =   prim1[UZZ];

  double vn1  = vr1*n[0] + vp1*n[1] + vz1*n[2];

  double cs1  = sqrt( GAMMALAW*(P1/rho1) );

  double Br1 = prim1[BRR];
  double Bp1 = prim1[BPP];
  double Bz1 = prim1[BZZ];

  double Bn1 =  Br1*n[0] + Bp1*n[1] + Bz1*n[2];
  double B21 = (Br1*Br1  + Bp1*Bp1  + Bz1*Bz1);
  double b21 = B21/rho1;
  double psiL = prim1[PSI];
  double FpsL = wp*psiL + pow(DIVB_CH,2.)*Bn1;

  double FrL = vn1*Br1 - Bn1*vr1 + psiL*n[0];
  double FpL = vn1*Bp1 - Bn1*vp1 + psiL*n[1];
  double FzL = vn1*Bz1 - Bn1*vz1 + psiL*n[2];

  double mrL = rho1*vr1;
  double mpL = rho1*vp1;
  double mzL = rho1*vz1;

  double FmrL = rho1*vr1*vn1 + (P1+.5*B21)*n[0] - Br1*Bn1;
  double FmpL = rho1*vp1*vn1 + (P1+.5*B21)*n[1] - Bp1*Bn1;
  double FmzL = rho1*vz1*vn1 + (P1+.5*B21)*n[2] - Bz1*Bn1;

  double cf21 = .5*( cs1*cs1 + b21 + sqrt(fabs(  (cs1*cs1+b21)*(cs1*cs1+b21) - 4.0*cs1*cs1*Bn1*Bn1/rho1 )) );

  L_Mins = vn1 - sqrt( cf21 );
  L_Plus = vn1 + sqrt( cf21 );

  double P2   = prim2[PPP];
  double rho2 = prim2[RHO];

  double vr2  =   prim2[URR];
  double vp2  = r*prim2[UPP];
  double vz2  =   prim2[UZZ];

  double vn2  = vr2*n[0] + vp2*n[1] + vz2*n[2];

  double cs2  = sqrt( GAMMALAW*(P2/rho2) );

  double Br2 = prim2[BRR];
  double Bp2 = prim2[BPP];
  double Bz2 = prim2[BZZ];

  double Bn2 =  Br2*n[0] + Bp2*n[1] + Bz2*n[2];
  double B22 = (Br2*Br2  + Bp2*Bp2  + Bz2*Bz2);
  double b22 = B22/rho2;
  double psiR = prim2[PSI];
  double FpsR = wp*psiR + pow(DIVB_CH,2.)*Bn2;

  double FrR = vn2*Br2 - Bn2*vr2 + psiR*n[0];
  double FpR = vn2*Bp2 - Bn2*vp2 + psiR*n[1];
  double FzR = vn2*Bz2 - Bn2*vz2 + psiR*n[2];

  double mrR = rho2*vr2;
  double mpR = rho2*vp2;
  double mzR = rho2*vz2;

  double FmrR = rho2*vr2*vn2 + (P2+.5*B22)*n[0] - Br2*Bn2;
  double FmpR = rho2*vp2*vn2 + (P2+.5*B22)*n[1] - Bp2*Bn2;
  double FmzR = rho2*vz2*vn2 + (P2+.5*B22)*n[2] - Bz2*Bn2;

  double cf22 = .5*( cs2*cs2 + b22 + sqrt(fabs(  (cs2*cs2+b22)*(cs2*cs2+b22) - 4.0*cs2*cs2*Bn2*Bn2/rho2 )) );

  if( L_Mins > vn2 - sqrt( cf22 ) ) L_Mins = vn2 - sqrt( cf22 );
  if( L_Plus < vn2 + sqrt( cf22 ) ) L_Plus = vn2 + sqrt( cf22 );
  if( L_Mins > -ch+wp ) L_Mins = -ch+wp;
  if( L_Plus <  ch+wp ) L_Plus =  ch+wp;

  double aL = L_Plus;
  double aR = -L_Mins;

  double Br = ( aR*Br1 + aL*Br2 + FrL - FrR )/( aL + aR );
  double Bp = ( aR*Bp1 + aL*Bp2 + FpL - FpR )/( aL + aR );
  double Bz = ( aR*Bz1 + aL*Bz2 + FzL - FzR )/( aL + aR );
  double Bn = Br*n[0] + Bp*n[1] + Bz*n[2];

  double mr = ( aR*mrL + aL*mrR + FmrL - FmrR )/( aL + aR );
  double mp = ( aR*mpL + aL*mpR + FmpL - FmpR )/( aL + aR );
  double mz = ( aR*mzL + aL*mzR + FmzL - FmzR )/( aL + aR );

  double mnL = mrL*n[0]+mpL*n[1]+mzL*n[2];
  double mnR = mrR*n[0]+mpR*n[1]+mzR*n[2];
  double rho = ( aR*rho1 + aL*rho2 + mnL - mnR )/( aL + aR );
  double psi = ( aR*psiL + aL*psiR + FpsL - FpsR )/( aL + aR );

  L_Star = ( rho2*vn2*(L_Plus-vn2) - rho1*vn1*(L_Mins-vn1) + (P1+.5*B21-Bn1*Bn1) - (P2+.5*B22-Bn2*Bn2) )/( rho2*(L_Plus-vn2) - rho1*(L_Mins-vn1) );

  double vr = mr/rho;
  double vp = mp/rho;
  double vz = mz/rho;
  double vdotB = vr*Br + vp*Bp + vz*Bz;

  Bpack[0] = Bn;
  Bpack[1] = Br;
  Bpack[2] = Bp;
  Bpack[3] = Bz;
  Bpack[4] = vdotB;
  Bpack[5] = psi;

  theRiemann->Sl = L_Mins;
  theRiemann->Sr = L_Plus;
  theRiemann->Ss = L_Star;

}

void riemann_set_state(struct Riemann * theRiemann,int w ){
  if (w < theRiemann->Sl){
    theRiemann->state=LEFT;
  }else if( w > theRiemann->Sr ){
    theRiemann->state=RIGHT;
  }else{
    if( w < theRiemann->Ss ){
      theRiemann->state=LEFTSTAR;
    }else{
      theRiemann->state=RIGHTSTAR;
    }
  }
}


void riemann_set_Ustar(struct Riemann * theRiemann,double *n,double r,double *Bpack,double GAMMALAW){
  double *prim;
  double Sk;
  if (theRiemann->state==LEFTSTAR){
    prim = theRiemann->primL;
    Sk = theRiemann->Sl;
  }else{
    prim = theRiemann->primR;
    Sk = theRiemann->Sr;
  }
  double Ss=theRiemann->Ss;

  double Bsn = Bpack[0];
  double Bsr = Bpack[1];
  double Bsp = Bpack[2];
  double Bsz = Bpack[3];
  double vBs = Bpack[4];
  double psi = Bpack[5];

  double rho = prim[RHO];
  double vr  = prim[URR];
  double vp  = prim[UPP]*r;
  double vz  = prim[UZZ];
  double Pp  = prim[PPP];

  double Br  = prim[BRR];
  double Bp  = prim[BPP];
  double Bz  = prim[BZZ];

  double v2 = vr*vr+vp*vp+vz*vz;
  double B2 = Br*Br+Bp*Bp+Bz*Bz;

  double vn = vr*n[0] + vp*n[1] + vz*n[2];
  double Bn = Br*n[0] + Bp*n[1] + Bz*n[2];
  double vB = vr*Br   + vp*Bp   + vz*Bz;

  double rhoe = Pp/(GAMMALAW-1.);

  double D  = rho;
  double mr = rho*vr;
  double mp = rho*vp;
  double mz = rho*vz;
  double E  = .5*rho*v2 + rhoe + .5*B2;

  double Bs2 = Bsr*Bsr+Bsp*Bsp+Bsz*Bsz;
  double Ps  = rho*( Sk - vn )*( Ss - vn ) + (Pp+.5*B2-Bn*Bn) - .5*Bs2 + Bsn*Bsn;

  double Dstar = ( Sk - vn )*D/( Sk - Ss );
  double Msr   = ( ( Sk - vn )*mr + ( Br*Bn - Bsr*Bsn ) ) / ( Sk - Ss );
  double Msp   = ( ( Sk - vn )*mp + ( Bp*Bn - Bsp*Bsn ) ) / ( Sk - Ss );
  double Msz   = ( ( Sk - vn )*mz + ( Bz*Bn - Bsz*Bsn ) ) / ( Sk - Ss );
  double Estar = ( ( Sk - vn )*E + (Ps+.5*Bs2)*Ss - (Pp+.5*B2)*vn - vBs*Bsn + vB*Bn ) / ( Sk - Ss );

  double Msn = Dstar*Ss;
  double mn  = Msr*n[0] + Msp*n[1] + Msz*n[2];

  Msr += n[0]*( Msn - mn );
  Msp += n[1]*( Msn - mn );
  Msz += n[2]*( Msn - mn );

  theRiemann->Ustar[DDD] = Dstar;
  theRiemann->Ustar[SRR] = Msr;
  theRiemann->Ustar[LLL] = r*Msp;
  theRiemann->Ustar[SZZ] = Msz;
  theRiemann->Ustar[TAU] = Estar;

  theRiemann->Ustar[BRR] = Bsr/r;
  theRiemann->Ustar[BPP] = Bsp/r;
  theRiemann->Ustar[BZZ] = Bsz;

  theRiemann->Ustar[PSI] = psi;

}

void riemann_set_flux(struct Riemann * theRiemann, double r , double * n ,double GAMMALAW,double DIVB_CH){
  double *prim;
  if ((theRiemann->state==LEFT)||(theRiemann->state==LEFTSTAR)){
    prim = theRiemann->primL;
  }else{
    prim = theRiemann->primR;
  }

  double rho = prim[RHO];
  double Pp  = prim[PPP];
  double vr  = prim[URR];
  double vp  = prim[UPP]*r;
  double vz  = prim[UZZ];

  double Br  = prim[BRR];
  double Bp  = prim[BPP];
  double Bz  = prim[BZZ];

  double vn = vr*n[0] + vp*n[1] + vz*n[2];
  double Bn = Br*n[0] + Bp*n[1] + Bz*n[2];
  double vB = vr*Br + vp*Bp + vz*Bz;

  double rhoe = Pp/(GAMMALAW-1.);
  double v2 = vr*vr + vp*vp + vz*vz;
  double B2 = Br*Br + Bp*Bp + Bz*Bz;

  theRiemann->F[DDD] = rho*vn;
  theRiemann->F[SRR] =     rho*vr*vn + (Pp+.5*B2)*n[0] - Br*Bn;
  theRiemann->F[LLL] = r*( rho*vp*vn + (Pp+.5*B2)*n[1] - Bp*Bn );
  theRiemann->F[SZZ] =     rho*vz*vn + (Pp+.5*B2)*n[2] - Bz*Bn;
  theRiemann->F[TAU] = ( .5*rho*v2 + rhoe + Pp + B2 )*vn - vB*Bn;

  double psi = prim[PSI];
  theRiemann->F[BRR] =(Br*vn - vr*Bn + psi*n[0])/r;
  theRiemann->F[BPP] =(Bp*vn - vp*Bn + psi*n[1])/r;
  theRiemann->F[BZZ] = Bz*vn - vz*Bn + psi*n[2];

  double wp = 0.0;
  theRiemann->F[PSI] = wp*psi*n[1] + pow(DIVB_CH,2.)*Bn;

}

void riemann_addto_flux_general(struct Riemann * theRiemann,double w,int NUM_Q){
  int q;
  for (q=0;q<NUM_Q;++q){
    if ((theRiemann->state==LEFT)||(theRiemann->state==RIGHT)){
      theRiemann->F[q] -= w*theRiemann->Uk[q];
    }else if(theRiemann->state==LEFTSTAR){
      theRiemann->F[q] += theRiemann->Sl*( theRiemann->Ustar[q] - theRiemann->Uk[q] ) - w*theRiemann->Ustar[q];
    }else{
      theRiemann->F[q] += theRiemann->Sr*( theRiemann->Ustar[q] - theRiemann->Uk[q] ) - w*theRiemann->Ustar[q]; 
    }
  }
}

void riemann_setup_rz(struct Riemann * theRiemann,struct Face * theFaces,struct Grid * theGrid,int n){
  int NUM_Q = grid_NUM_Q(theGrid);
  double deltaL = face_deltaL(face_pointer(theFaces,n));
  double deltaR = face_deltaR(face_pointer(theFaces,n));
  theRiemann->cL = face_L_pointer(theFaces,n);
  theRiemann->cR = face_R_pointer(theFaces,n);
  double pL = cell_tiph(theRiemann->cL) - .5*cell_dphi(theRiemann->cL);
  double pR = cell_tiph(theRiemann->cR) - .5*cell_dphi(theRiemann->cR);   
  double dpL =  face_cm(face_pointer(theFaces,n)) - pL;
  double dpR = -face_cm(face_pointer(theFaces,n)) + pR;
  while( dpL >  M_PI ) dpL -= 2.*M_PI;
  while( dpL < -M_PI ) dpL += 2.*M_PI;
  while( dpR >  M_PI ) dpR -= 2.*M_PI;
  while( dpR < -M_PI ) dpR += 2.*M_PI;
  dpL = dpL;
  dpR = dpR;
  theRiemann->r = face_r(face_pointer(theFaces,n));
  theRiemann->dA = face_dA(face_pointer(theFaces,n));

  int q;
  for (q=0;q<NUM_Q;++q){
    theRiemann->primL[q] = cell_prims(theRiemann->cL)[q] + cell_grad(theRiemann->cL)[q]*deltaL + cell_gradp(theRiemann->cL)[q]*dpL;
    theRiemann->primR[q] = cell_prims(theRiemann->cR)[q] + cell_grad(theRiemann->cR)[q]*deltaR + cell_gradp(theRiemann->cR)[q]*dpR;
  }
}

void riemann_setup_p(struct Riemann * theRiemann,struct Cell *** theCells,struct Grid * theGrid,int i,int j_low,int j_hi,int k){
  int NUM_Q = grid_NUM_Q(theGrid);
  theRiemann->cL = cell_single(theCells,i,j_low,k);
  theRiemann->cR = cell_single(theCells,i,j_hi ,k);
  double dpL = cell_dphi(theRiemann->cL);
  double dpR = cell_dphi(theRiemann->cR);
  double zm = grid_z_faces(theGrid,k-1);
  double zp = grid_z_faces(theGrid,k);
  double dz = zp-zm;
  double rm = grid_r_faces(theGrid,i-1);
  double rp = grid_r_faces(theGrid,i);
  double dr = rp-rm;
  double r = .5*(rp+rm);
  theRiemann->dA = dr*dz;
  theRiemann->r = r; 
  int q;
  for (q=0;q<NUM_Q;++q){
    theRiemann->primL[q] = cell_prims(theRiemann->cL)[q] + cell_gradp(theRiemann->cL)[q]*dpL;
    theRiemann->primR[q] = cell_prims(theRiemann->cR)[q] + cell_gradp(theRiemann->cR)[q]*dpR;
  }

}


//void riemann_hllc(struct Riemann * theRiemann, struct Cell * cL , struct Cell * cR, struct Grid *theGrid, double dA,double dt, double r,double deltaL,double deltaR, double dpL, double dpR , int direction ){
void riemann_hllc(struct Riemann * theRiemann, struct Grid *theGrid,double dt, int direction ){
  int NUM_Q = grid_NUM_Q(theGrid);
  double GAMMALAW = grid_GAMMALAW(theGrid);
  double DIVB_CH = grid_DIVB_CH(theGrid);

/*
  int q;
  for (q=0;q<NUM_Q;++q){
    theRiemann->primL[q] = cell_prims(cL)[q] + cell_grad(cL)[q]*deltaL + cell_gradp(cL)[q]*dpL;
    theRiemann->primR[q] = cell_prims(cR)[q] + cell_grad(cR)[q]*deltaR + cell_gradp(cR)[q]*dpR;
  }
*/
  //initialize
  double n[3];int i;
  for (i=0;i<3;++i){
    n[i]=0;
  }
  //set
  n[direction]=1.0;

  double Bpack[6];
  riemann_set_vel(theRiemann,n,theRiemann->r,Bpack,GAMMALAW,DIVB_CH);

  double Bk_face;
  if (direction==0){
    Bk_face = 0.5*(theRiemann->primL[BRR]+theRiemann->primR[BRR]);  
  } else if (direction==1){
    Bk_face = 0.5*(theRiemann->primL[BPP]+theRiemann->primR[BPP]);
  } else if (direction==2){
    Bk_face = 0.5*(theRiemann->primL[BZZ]+theRiemann->primR[BZZ]);
  }
  double Psi_face = 0.5*(theRiemann->primL[PSI]+theRiemann->primR[PSI]);

  double w;
  if (direction==1){
    if( grid_MOVE_CELLS(theGrid) == C_WRIEMANN ) cell_add_wiph(theRiemann->cL,theRiemann->Ss);
    w = cell_wiph(theRiemann->cL);
  } else{
    w = 0.0;
  }

  riemann_set_state(theRiemann,w);

  riemann_set_flux( theRiemann , theRiemann->r , n,GAMMALAW,DIVB_CH );
  if (theRiemann->state==LEFTSTAR){
    cell_prim2cons( theRiemann->primL , theRiemann->Uk , theRiemann->r , 1.0 ,GAMMALAW);
    riemann_set_Ustar(theRiemann,n,theRiemann->r,Bpack,GAMMALAW);
  } else if(theRiemann->state==RIGHTSTAR){
    cell_prim2cons( theRiemann->primR , theRiemann->Uk , theRiemann->r , 1.0 ,GAMMALAW);
    riemann_set_Ustar(theRiemann,n,theRiemann->r,Bpack,GAMMALAW);
  }
  riemann_addto_flux_general(theRiemann,w,grid_NUM_Q(theGrid));

int q;
for( q=0 ; q<NUM_Q ; ++q ){
    cell_add_cons(theRiemann->cL,q,-dt*theRiemann->dA*theRiemann->F[q]);
    cell_add_cons(theRiemann->cR,q,dt*theRiemann->dA*theRiemann->F[q]);
  }

  cell_add_divB(theRiemann->cL,theRiemann->dA*Bk_face);
  cell_add_divB(theRiemann->cR,-theRiemann->dA*Bk_face);
  cell_add_GradPsi(theRiemann->cL,direction,Psi_face);
  cell_add_GradPsi(theRiemann->cR,direction,-Psi_face);

}


