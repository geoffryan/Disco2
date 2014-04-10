#define RIEMANN_PRIVATE_DEFS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../Headers/Riemann.h"
#include "../Headers/GravMass.h"
#include "../Headers/Sim.h"
#include "../Headers/Cell.h"
#include "../Headers/Face.h"
#include "../Headers/header.h"


// ********************************************************************************************
// WE REALLY SHOULD IMPROVE THE COMMENTING OF ALL OF THESE ROUTINES. 
// THERE IS SOME COMPLICATED STUFF HERE. 
// LETS CHOOSE A REFERENCE SUCH AS TORO AND IDENTIFY LINES OF CODE WITH EQUATIONS IN THE BOOK.
// ********************************************************************************************

// this routine is only called by riemann_set_vel.
// It is used to find various L/R quantities. 
void LR_speed(double *prim,double r,int * n,double GAMMALAW,double * p_vn,double * p_cf2,double *Fm,double * p_mn){
  double P   = prim[PPP];
  double rho = prim[RHO];
  double vr  =   prim[URR];
  double vp  = r*prim[UPP];
  double vz  =   prim[UZZ];
  double vn  = vr*n[0] + vp*n[1] + vz*n[2];
  double cf2  = GAMMALAW*(P/rho);
  double mr = rho*vr;
  double mp = rho*vp;
  double mz = rho*vz;
  double mn = mr*n[0]+mp*n[1]+mz*n[2];

  Fm[0] = rho*vr*vn + P*n[0];
  Fm[1] = rho*vp*vn + P*n[1];
  Fm[2] = rho*vz*vn + P*n[2];

  *p_vn = vn;
  *p_cf2 = cf2;
  *p_mn = mn;
}

// this routine is only called by riemann_set_vel.
// It is used to find various L/R quantities. 
void LR_speed_mhd(double *prim,double r,int * n,double ch, double * p_cf2,double *F,double *Fm,double * p_Bn,double * p_B2){
  double rho  =   prim[RHO];
  double vr  =   prim[URR];
  double vp  = r*prim[UPP];
  double vz  =   prim[UZZ];
  double vn  = vr*n[0] + vp*n[1] + vz*n[2];

  double Br = prim[BRR];
  double Bp = prim[BPP];
  double Bz = prim[BZZ];
  double psi = prim[PSI];

  double Bn =  Br*n[0] + Bp*n[1] + Bz*n[2];
  double B2 = (Br*Br  + Bp*Bp  + Bz*Bz);
  double Fps = pow(ch,2.)*Bn;

  F[0] = vn*Br - Bn*vr + psi*n[0];
  F[1] = vn*Bp - Bn*vp + psi*n[1];
  F[2] = vn*Bz - Bn*vz + psi*n[2];

  Fm[0] +=  .5*B2*n[0] - Br*Bn;
  Fm[1] +=  .5*B2*n[1] - Bp*Bn;
  Fm[2] +=  .5*B2*n[2] - Bz*Bn;

  *p_cf2 = .5*( *p_cf2 + B2/rho + sqrt(fabs(  (*p_cf2+B2/rho)*(*p_cf2+B2/rho) - 4.0*(*p_cf2)*Bn*Bn/rho )) );

  *p_Bn = Bn;
  *p_B2 = B2;
}

// Find velocities needed for the Riemann problem
void riemann_set_vel(struct Riemann * theRiemann,struct Sim * theSim,struct GravMass * theGravMasses, double r,double *Bpack,double GAMMALAW,double DIVB_CH){
  double Sl, Sr, Ss;

  double vnL,cf21,mnL,BnL,B2L;
  double FL[3], FmL[3];
  LR_speed(theRiemann->primL,r,theRiemann->n,GAMMALAW,&vnL,&cf21,FmL,&mnL);
  if (sim_runtype(theSim)==MHD){
    LR_speed_mhd(theRiemann->primL,r,theRiemann->n,DIVB_CH,&cf21,FL,FmL,&BnL,&B2L);
  }
  Sl = vnL - sqrt( cf21 );
  Sr = vnL + sqrt( cf21 );

  double vnR,cf22,mnR,BnR,B2R;
  double FR[3],FmR[3];
  LR_speed(theRiemann->primR,r,theRiemann->n,GAMMALAW,&vnR,&cf22,FmR,&mnR);
  if (sim_runtype(theSim)==MHD){
    LR_speed_mhd(theRiemann->primR,r,theRiemann->n,DIVB_CH,&cf22,FR,FmR,&BnR,&B2R);
  }
  if( Sl > vnR - sqrt( cf22 ) ) Sl = vnR - sqrt( cf22 );
  if( Sr < vnR + sqrt( cf22 ) ) Sr = vnR + sqrt( cf22 );

  double a = gravMass_r(theGravMasses,0) + gravMass_r(theGravMasses,1);
  double wp_a = theRiemann->n[1]*sim_rOm_a(theSim,r,a);

  if (sim_runtype(theSim)==MHD && DIVB_CH>0.00000001){
    if( Sl - wp_a > -DIVB_CH ) Sl = -DIVB_CH + wp_a;
    if( Sr - wp_a <  DIVB_CH ) Sr =  DIVB_CH + wp_a;
  }

  double  mr = ( -Sl*theRiemann->primL[RHO]*theRiemann->primL[URR] + Sr*theRiemann->primR[RHO]*theRiemann->primR[URR] + FmL[0] - FmR[0] )/( Sr - Sl );
  double  mp = ( -Sl*theRiemann->primL[RHO]*theRiemann->primL[UPP]*r + Sr*theRiemann->primR[RHO]*theRiemann->primR[UPP]*r + FmL[1] - FmR[1] )/( Sr - Sl );
  double  mz = ( -Sl*theRiemann->primL[RHO]*theRiemann->primL[UZZ] + Sr*theRiemann->primR[RHO]*theRiemann->primR[UZZ] + FmL[2] - FmR[2] )/( Sr - Sl );
  double rho = ( -Sl*theRiemann->primL[RHO] + Sr*theRiemann->primR[RHO] + mnL - mnR )/( Sr - Sl );

  Ss = (theRiemann->primR[RHO]*vnR*(Sr-vnR)-theRiemann->primL[RHO]*vnL*(Sl-vnL)+theRiemann->primL[PPP]-theRiemann->primR[PPP])
    /(theRiemann->primR[RHO]*(Sr-vnR)-theRiemann->primL[RHO]*(Sl-vnL));

  //add mhd terms
  if (sim_runtype(theSim)==MHD){  
    double Br = ( -Sl*theRiemann->primL[BRR] + Sr*theRiemann->primR[BRR] + FL[0] - FR[0] )/( Sr - Sl );
    double Bp = ( -Sl*theRiemann->primL[BPP] + Sr*theRiemann->primR[BPP] + FL[1] - FR[1] )/( Sr - Sl );
    double Bz = ( -Sl*theRiemann->primL[BZZ] + Sr*theRiemann->primR[BZZ] + FL[2] - FR[2] )/( Sr - Sl );
    double psi = ( -Sl*theRiemann->primL[PSI] + Sr*theRiemann->primR[PSI] + pow(DIVB_CH,2.)*BnL + wp_a*theRiemann->primL[PSI] - pow(DIVB_CH,2.)*BnR - wp_a*theRiemann->primR[PSI] )/( Sr - Sl );
    Bpack[0] = Br*theRiemann->n[0] + Bp*theRiemann->n[1] + Bz*theRiemann->n[2]; // Bn
    Bpack[1] = Br;
    Bpack[2] = Bp;
    Bpack[3] = Bz;
    Bpack[4] = (mr*Br + mp*Bp + mz*Bz)/rho; // v dot B
    Bpack[5] = psi;

    //Ss += ((.5*B2L-BnL*BnL)-.5*B2R+BnR*BnR) / (theRiemann->primR[RHO]*(Sr-vnR)-theRiemann->primL[RHO]*(Sl-vnL));
    Ss += (.5*B2L-.5*B2R) / (theRiemann->primR[RHO]*(Sr-vnR)-theRiemann->primL[RHO]*(Sl-vnL));
  }
  theRiemann->Sl = Sl;
  theRiemann->Sr = Sr;
  theRiemann->Ss = Ss;

}

// Which state of the riemann problem are we in?
void riemann_set_state(struct Riemann * theRiemann,double w ){
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

void riemann_set_star_hll(struct Riemann * theRiemann,struct Sim * theSim){
  double Sr =  theRiemann->Sr;
  double Sl = theRiemann->Sl;
  int q;
  for( q=0 ; q<sim_NUM_Q(theSim) ; ++q ){
    theRiemann->Ustar[q] = ( Sr*theRiemann->UR[q] - Sl*theRiemann->UL[q] + theRiemann->FL[q] - theRiemann->FR[q] )/( Sr - Sl );
    theRiemann->Fstar[q] = ( Sr*theRiemann->FL[q] - Sl*theRiemann->FR[q] - Sr*Sl*( theRiemann->UL[q] - theRiemann->UR[q] ) )/( Sr - Sl );
  }
}


void riemann_set_star_hllc(struct Riemann * theRiemann,struct Sim * theSim,double *Bpack,double GAMMALAW){
  double r = theRiemann->r;
  double *prim;
  double Sk;
  double *Uk;
  double *Fk;
  if (theRiemann->state==LEFTSTAR){
    prim = theRiemann->primL;
    Sk = theRiemann->Sl;
    Uk = theRiemann->UL;
    Fk = theRiemann->FL;
  }else{
    prim = theRiemann->primR;
    Sk = theRiemann->Sr;
    Uk = theRiemann->UR;
    Fk = theRiemann->FR;
  }
  double Ss=theRiemann->Ss;

  double rho = prim[RHO];
  double vr  = prim[URR];
  double vp  = prim[UPP]*r;
  double vz  = prim[UZZ];
  double Pp  = prim[PPP];
  double v2 = vr*vr+vp*vp+vz*vz;
  double vn = vr*theRiemann->n[0] + vp*theRiemann->n[1] + vz*theRiemann->n[2];
  double rhoe = Pp/(GAMMALAW-1.);
  double D  = rho;
  double mr = rho*vr;
  double mp = rho*vp;
  double mz = rho*vz;
  double E_hydro  = .5*rho*v2 + rhoe;
  double Ps  = rho*( Sk - vn )*( Ss - vn ) + Pp;
  double Dstar = ( Sk - vn )*D/( Sk - Ss );
  double Msn = Dstar*Ss;
  double Msr   = ( Sk - vn )*mr / ( Sk - Ss );
  double Msp   = ( Sk - vn )*mp / ( Sk - Ss );
  double Msz   = ( Sk - vn )*mz / ( Sk - Ss );
  double Estar = ( ( Sk - vn )*E_hydro + Ps*Ss - Pp*vn ) / ( Sk - Ss );

  if (sim_runtype(theSim)==MHD){
    double Bsn = Bpack[0];
    double Bsr = Bpack[1];
    double Bsp = Bpack[2];
    double Bsz = Bpack[3];
    double vBs = Bpack[4];
    double psi = Bpack[5];

    double Br  = prim[BRR];
    double Bp  = prim[BPP];
    double Bz  = prim[BZZ];
    double B2 = Br*Br+Bp*Bp+Bz*Bz;
    double Bn = Br*theRiemann->n[0] + Bp*theRiemann->n[1] + Bz*theRiemann->n[2];
    double vB = vr*Br   + vp*Bp   + vz*Bz;
    double Bs2 = Bsr*Bsr+Bsp*Bsp+Bsz*Bsz;
    //double Ps_mag = (.5*B2-Bn*Bn) - .5*Bs2 + Bsn*Bsn;
    Msr   += ( Br*Bn - Bsr*Bsn ) / ( Sk - Ss );
    Msp   += ( Bp*Bn - Bsp*Bsn ) / ( Sk - Ss );
    Msz   += ( Bz*Bn - Bsz*Bsn ) / ( Sk - Ss );
    Estar += ( ( Sk - vn )*.5*B2 + (/*Ps_mag+*/.5*Bs2)*Ss - .5*B2*vn - vBs*Bsn + vB*Bn ) / ( Sk - Ss );

    double BnL = theRiemann->primL[BRR]*theRiemann->n[0] 
      + theRiemann->primL[BPP]*theRiemann->n[1] 
      + theRiemann->primL[BZZ]*theRiemann->n[2];

    double BnR = theRiemann->primR[BRR]*theRiemann->n[0] 
      + theRiemann->primR[BPP]*theRiemann->n[1] 
      + theRiemann->primR[BZZ]*theRiemann->n[2];

    double wp_a = sim_rOm_a(theSim,r,1.);
    printf("Need to fix wp_a stuff when MHD is turned on\n");
    exit(1);

    theRiemann->Ustar[BRR] = Bsr;
    theRiemann->Ustar[BPP] = (Bsp + wp_a*(BnL-BnR)/(theRiemann->Sr-theRiemann->Sl))/r;
    theRiemann->Ustar[BZZ] = Bsz;
    theRiemann->Ustar[PSI] = psi;
  }
  double mn  = Msr*theRiemann->n[0] + Msp*theRiemann->n[1] + Msz*theRiemann->n[2];

  Msr += theRiemann->n[0]*( Msn - mn );
  Msp += theRiemann->n[1]*( Msn - mn );
  Msz += theRiemann->n[2]*( Msn - mn );

  theRiemann->Ustar[DDD] = Dstar;
  theRiemann->Ustar[SRR] = Msr;
  theRiemann->Ustar[LLL] = r*Msp;
  theRiemann->Ustar[SZZ] = Msz;
  theRiemann->Ustar[TAU] = Estar;

  int q;
  for( q=sim_NUM_C(theSim) ; q<sim_NUM_Q(theSim) ; ++q ){
    theRiemann->Ustar[q] = prim[q]*theRiemann->Ustar[DDD];
  }

  //Now set Fstar
  for (q=0;q<sim_NUM_Q(theSim);++q){
    theRiemann->Fstar[q] = Fk[q] + Sk*( theRiemann->Ustar[q] - Uk[q] ) ;
  }
}

void riemann_set_flux(struct Riemann * theRiemann, struct Sim * theSim,double GAMMALAW,double DIVB_CH,int SetState){
  double r = theRiemann->r;
  double *prim;
  double *F;
  if (SetState==LEFT){
    prim = theRiemann->primL;
    F = theRiemann->FL;
  }else if (SetState==RIGHT){
    prim = theRiemann->primR;
    F = theRiemann->FR;
  } else{
    printf("ERROR\n");
    exit(0);
  }

  double rho = prim[RHO];
  double Pp  = prim[PPP];
  double vr  = prim[URR];
  double vp  = prim[UPP]*r;
  double vz  = prim[UZZ];
  double vn = vr*theRiemann->n[0] + vp*theRiemann->n[1] + vz*theRiemann->n[2];
  double rhoe = Pp/(GAMMALAW-1.);
  double v2 = vr*vr + vp*vp + vz*vz;
  F[DDD] = rho*vn;
  F[SRR] =     rho*vr*vn + Pp*theRiemann->n[0] ;
  F[LLL] = r*( rho*vp*vn + Pp*theRiemann->n[1] );
  F[SZZ] =     rho*vz*vn + Pp*theRiemann->n[2] ;
  F[TAU] = ( .5*rho*v2 + rhoe + Pp )*vn ;

  if (sim_runtype(theSim)==MHD){ 
    double Br  = prim[BRR];
    double Bp  = prim[BPP];
    double Bz  = prim[BZZ];
    double Bn = Br*theRiemann->n[0] + Bp*theRiemann->n[1] + Bz*theRiemann->n[2];
    double vB = vr*Br + vp*Bp + vz*Bz;
    double B2 = Br*Br + Bp*Bp + Bz*Bz;

    double wp_a = sim_rOm_a(theSim,r,1.);
    printf("Need to fix wp_a stuff when MHD is turned on\n");
    exit(1);

    F[SRR] +=     .5*B2*theRiemann->n[0] - Br*Bn;
    F[LLL] += r*( .5*B2*theRiemann->n[1] - Bp*Bn );
    F[SZZ] +=     .5*B2*theRiemann->n[2] - Bz*Bn;
    F[TAU] += B2*vn - vB*Bn;
    double psi = prim[PSI];
    F[BRR] =(Br*vn - vr*Bn + psi*theRiemann->n[0]);
    F[BPP] =(Bp*vn - vp*Bn + wp_a*Bn + psi*theRiemann->n[1])/r;
    F[BZZ] = Bz*vn - vz*Bn + psi*theRiemann->n[2];
    F[PSI] = pow(DIVB_CH,2.)*Bn + wp_a*psi*theRiemann->n[1];
  }

  int q;
  for( q=sim_NUM_C(theSim) ; q<sim_NUM_Q(theSim) ; ++q ){
    F[q] = prim[q]*F[DDD];
  }

}

/*
   double Gradient_r2RhoNu_o_r2RhoNu(double * AvgPrim , double * Grad_prim, double r, double tiph, int IDtype, int direction){
   if (direction==RDIRECTION){
   if (sim_VISC_CONST(theSim)==1){
   return(2./r + Grad_prim[RHO]/AvgPrim[RHO]);
   } else{
   if (IDtype==SHEAR){
   return(4./r + Grad_prim[PPP]/AvgPrim[PPP]);
   } else{
   return(7./2./r + Grad_prim[PPP]/AvgPrim[PPP]);
   } 
   } 
   } else if (direction==PDIRECTION){
   if (sim_VISC_CONST(theSim)==1){
   return(Grad_prim[RHO]/AvgPrim[RHO]);
   } else{
   if (IDtype==SHEAR){
   return(Grad_prim[PPP]/AvgPrim[PPP] - 2.0/r * tan(tiph));
   } else{
   return(Grad_prim[PPP]/AvgPrim[PPP]);
   } 
   } 
   } else{
   printf("error in Gradient_r2RhoNu_o_r2RhoNu\n");
   exit(1);
   }
   }
   */

void riemann_visc_flux(struct Riemann * theRiemann,struct Sim * theSim,struct GravMass * theGravMasses ){
  int NUM_Q = sim_NUM_Q(theSim);

  double r = theRiemann->r;
  double tiph = theRiemann->cm;

  double Mtotal = 1.0;
  double sep = 1.0;
  double M0 = gravMass_M(theGravMasses,0);
  double M1 = gravMass_M(theGravMasses,1);

  double r0 = gravMass_r(theGravMasses,0);
  double r1 = gravMass_r(theGravMasses,1);

  double a = r0 + r1;

  double phi_bh0 = gravMass_phi(theGravMasses,0);
  double phi_bh1 = gravMass_phi(theGravMasses,1);

  double xbh0 = r0*cos(phi_bh0);
  double ybh0 = r0*sin(phi_bh0);

  double xbh1 = r1*cos(phi_bh1);
  double ybh1 = r1*sin(phi_bh1);

  double xpos = r*cos(tiph);
  double ypos = r*sin(tiph);

  double dist_bh0 = sqrt((xpos-xbh0)*(xpos-xbh0)+(ypos-ybh0)*(ypos-ybh0));
  double dist_bh1 = sqrt((xpos-xbh1)*(xpos-xbh1)+(ypos-ybh1)*(ypos-ybh1));

  double * VFlux = malloc(NUM_Q*sizeof(double));
  double * AvgPrim = malloc(NUM_Q*sizeof(double));
  double * Grad_r_prim = malloc(NUM_Q*sizeof(double));
  double * Grad_ph_prim = malloc(NUM_Q*sizeof(double));
  int q;
  for (q=0;q<NUM_Q;++q){
    AvgPrim[q] = .5*(theRiemann->primL[q]+theRiemann->primR[q]);
    //AvgPrim[q] = 0.5*(cell_prim(theRiemann->cL,q)+cell_prim(theRiemann->cR,q));
    Grad_ph_prim[q] = .5*(cell_gradp(theRiemann->cL,q)+cell_gradp(theRiemann->cR,q))/r;    
    Grad_r_prim[q] = .5*(cell_grad(theRiemann->cL,q)+cell_grad(theRiemann->cR,q));

    if (1==1){
      if (theRiemann->n[0]==1){
        double pL = cell_tiph(theRiemann->cL) - .5*cell_dphi(theRiemann->cL);
        double pR = cell_tiph(theRiemann->cR) - .5*cell_dphi(theRiemann->cR);   

        double dpL =  theRiemann->cm - pL;
        double dpR = -theRiemann->cm + pR;
        while( dpL >  PHIMAX/2. ) dpL -= PHIMAX;
        while( dpL < -PHIMAX/2. ) dpL += PHIMAX;
        while( dpR >  PHIMAX/2. ) dpR -= PHIMAX;
        while( dpR < -PHIMAX/2. ) dpR += PHIMAX;

        double WL = cell_prim(theRiemann->cL,q) + dpL*cell_gradp(theRiemann->cL,q);
        double WR = cell_prim(theRiemann->cR,q) - dpR*cell_gradp(theRiemann->cR,q);

        double deltaL = theRiemann->r-theRiemann->r_cell_L;
        double deltaR = theRiemann->r_cell_R - theRiemann->r;
        AvgPrim[q] = 0.5*(WL+WR);
        Grad_r_prim[q] = (WR-WL)/(deltaR+deltaL);
      }
      if (theRiemann->n[1]==1){
        AvgPrim[q] = 0.5*(cell_prim(theRiemann->cR,q) + cell_prim(theRiemann->cL,q));
        Grad_ph_prim[q] = (cell_prim(theRiemann->cR,q) - cell_prim(theRiemann->cL,q))/ (cell_tiph(theRiemann->cR)-cell_tiph(theRiemann->cL))/r;
      }
    }

    VFlux[q] = 0.0;
  }

  double nu;
  double alpha =  sim_EXPLICIT_VISCOSITY(theSim);
  double Gamma =  sim_GAMMALAW(theSim);
  if (sim_VISC_CONST(theSim)==1){
    nu = alpha;
  } else{
    if (sim_InitialDataType(theSim)==SHEAR){
      double HoR = 0.1;
      //nu = alpha*HoR*HoR*pow(fabs((r*cos(tiph))),1.5);
      nu = alpha*Gamma*AvgPrim[PPP]/AvgPrim[RHO]*pow(fabs(r*cos(tiph)),2.0);    
      if (r*cos(tiph)>20.) nu=0.0;
    } else{
      //nu = alpha*Gamma*AvgPrim[PPP]/AvgPrim[RHO]*pow(r,1.5);
      //nu = alpha*Gamma*AvgPrim[PPP]/AvgPrim[RHO]*(sqrt(M0)+sqrt(M1))/(sqrt(M0)*pow(dist_bh0,-1.5)+sqrt(M1)*pow(dist_bh1,-1.5));
      double eps = sim_G_EPS(theSim);
      nu = alpha*AvgPrim[PPP]/AvgPrim[RHO]/sqrt(pow(dist_bh0*dist_bh0+eps*eps,-1.5)*M0+pow(dist_bh1*dist_bh1+eps*eps,-1.5)*M1);
    }
  }

  double rho = AvgPrim[RHO];
  double vr  = AvgPrim[URR];
  double om  = AvgPrim[UPP];
  double vz  = AvgPrim[UZZ];

  if (fabs(vz)>1.e-12){
    printf("you should not be using viscosity and having a nonzero component of velocity in z direction. Viscosity is only for 2d planar motion right now\n");
    exit(0);
  }

  double Gr_vr = Grad_r_prim[URR];
  double Gr_om = Grad_r_prim[UPP];
  double Gr_vz = Grad_r_prim[UZZ];

  double Gp_vr = Grad_ph_prim[URR];
  double Gp_om = Grad_ph_prim[UPP];
  double Gp_vz = Grad_ph_prim[UZZ];

  double om_cell = sim_rOm_a(theSim,r,a)/r;
  double rdr_om_cell = sim_rdrOm_a(theSim,r,a);

  /*
     if (sim_MOVE_CELLS(theSim)==C_FIXED){
     om_cell = 0.0;
     dr_om_cell = 0.0;
     } else if (sim_MOVE_CELLS(theSim)==C_MILOS){
     om_cell = (1.-exp(-pow(r,1.5)))/pow(r,1.5);
     dr_om_cell = 1.5*pow(r,-2.5)*(-1.+(1.+pow(r,1.5))*exp(-pow(r,1.5)));
     }else if (sim_MOVE_CELLS(theSim)==C_KEPLER){
     om_cell = pow(r,-1.5);
     dr_om_cell = -1.5*pow(r,-2.5);
     }else{
     printf("Problem with setting cell speed in riemann solver\n");
     exit(1);
     }
     */

  //r direction
  if (theRiemann->n[0] ==1){
    //    double Gp_r2RhoNu_o_r2RhoNu = Gradient_r2RhoNu_o_r2RhoNu(AvgPrim,Grad_ph_prim,r,tiph,sim_InitialDataType(theSim),PDIRECTION);

    /*
       VFlux[SRR] = -nu*rho*( 
       r*Gr_vr - vr + (om+om_cell)*r*r*Gp_r2RhoNu_o_r2RhoNu
       );
       VFlux[LLL] = -nu*rho*( 
       r*r*Gr_om + r*r*dr_om_cell - vr*r*Gp_r2RhoNu_o_r2RhoNu
       );
       */
    VFlux[SRR] = -nu*rho*(
        r*Gr_vr - vr - r*r*Gp_om
        );
    VFlux[LLL] = -nu*rho*(
        r*(r*Gr_om+rdr_om_cell) + r*Gp_vr
        );

    VFlux[SZZ] = 0.0; //deal with this later

    VFlux[TAU] = - nu * rho * (
        vr*( Gr_vr - vr/r - r*Gp_om ) + r*om*( Gp_vr + (rdr_om_cell + r*Gr_om) )
        );


  }
  //phi direction
  if (theRiemann->n[1] ==1){
    //    double Gr_r2RhoNu_o_r2RhoNu = Gradient_r2RhoNu_o_r2RhoNu(AvgPrim,Grad_r_prim,r,tiph,sim_InitialDataType(theSim),RDIRECTION);

    /*
       VFlux[SRR] = -nu*rho*( 
       -(om+om_cell)*r*r*Gr_r2RhoNu_o_r2RhoNu + r*Gp_vr 
       );
       VFlux[LLL] = -nu*rho*( 
       r*r*Gp_om + vr*r*Gr_r2RhoNu_o_r2RhoNu
       );
       */
    VFlux[SRR] = -nu*rho*(
        r*(r*Gr_om+rdr_om_cell) + r*Gp_vr
        );
    VFlux[LLL] = -nu*rho*(
        -r*Gr_vr + vr + r*r*Gp_om
        );

    VFlux[SZZ] = 0.0; //deal with this later

    VFlux[TAU] = -nu * rho * (
        -r*om*( Gr_vr - vr/r - r*Gp_om ) + vr*(Gp_vr + (rdr_om_cell + r*Gr_om ) )
        );
  }

  for (q=0;q<NUM_Q;++q){
    theRiemann->Fvisc[q] = VFlux[q];
  }
  free(VFlux);
  free(Grad_r_prim);
  free(Grad_ph_prim);
  free(AvgPrim);
}

void riemann_visc_flux_old(struct Riemann * theRiemann,struct Sim * theSim ){
  double nu = sim_EXPLICIT_VISCOSITY(theSim);
  int NUM_Q = sim_NUM_Q(theSim);

  double r = theRiemann->r;

  double * VFlux = malloc(NUM_Q*sizeof(double));
  double * AvgPrim = malloc(NUM_Q*sizeof(double));
  double * Gprim = malloc(NUM_Q*sizeof(double));
  int q;
  for (q=0;q<NUM_Q;++q){
    AvgPrim[q] = .5*(theRiemann->primL[q]+theRiemann->primR[q]);
    if (theRiemann->n[1]==1){ 
      Gprim[q] = .5*(cell_gradp(theRiemann->cL,q)+cell_gradp(theRiemann->cR,q));    
    } else{
      Gprim[q] = .5*(cell_grad(theRiemann->cL,q)+cell_grad(theRiemann->cR,q));
    }
    VFlux[q] = 0.0;
  }

  double rho = AvgPrim[RHO];
  double vr  = AvgPrim[URR];
  double om  = AvgPrim[UPP];
  double vz  = AvgPrim[UZZ];

  double dnvr = Gprim[URR];
  double dnom = Gprim[UPP];
  double dnvz = Gprim[UZZ];

  VFlux[SRR] = -nu*rho*( dnvr - theRiemann->n[1]*2.*om );
  VFlux[LLL] = -nu*rho*( r*r*dnom + theRiemann->n[0]*2.*vr );
  VFlux[SZZ] = -nu*rho*dnvz;
  VFlux[TAU] = -nu*rho*(vr*dnvr+r*r*om*dnom+vz*dnvz);  

  for (q=0;q<NUM_Q;++q){
    theRiemann->Fvisc[q] = VFlux[q];
  }
  free(VFlux);
  free(Gprim);
  free(AvgPrim);
}


void riemann_setup_rz(struct Riemann * theRiemann,struct Face * theFaces,struct Sim * theSim,int FaceNumber,int direction){
  theRiemann->n[direction]=1; // set
  int NUM_Q = sim_NUM_Q(theSim);
  double deltaL = face_deltaL(theFaces,FaceNumber);
  double deltaR = face_deltaR(theFaces,FaceNumber);
  theRiemann->cL = face_L_pointer(theFaces,FaceNumber);
  theRiemann->cR = face_R_pointer(theFaces,FaceNumber);
  double pL = cell_tiph(theRiemann->cL) - .5*cell_dphi(theRiemann->cL);
  double pR = cell_tiph(theRiemann->cR) - .5*cell_dphi(theRiemann->cR);   
  double dpL =  face_cm(theFaces,FaceNumber) - pL;
  double dpR = -face_cm(theFaces,FaceNumber) + pR;
  while( dpL >  PHIMAX/2. ) dpL -= PHIMAX;
  while( dpL < -PHIMAX/2. ) dpL += PHIMAX;
  while( dpR >  PHIMAX/2. ) dpR -= PHIMAX;
  while( dpR < -PHIMAX/2. ) dpR += PHIMAX;
  dpL = dpL;
  dpR = dpR;
  theRiemann->r = face_r(theFaces,FaceNumber);
  theRiemann->dA = face_dA(theFaces,FaceNumber);
  theRiemann->cm = face_cm(theFaces,FaceNumber);
  if (direction==0){
    theRiemann->r_cell_L = theRiemann->r-deltaL;
    theRiemann->r_cell_R = theRiemann->r+deltaR;
  } else{
    theRiemann->r_cell_L = theRiemann->r;
    theRiemann->r_cell_R = theRiemann->r;
  }

  int q;
  for (q=0;q<NUM_Q;++q){
    theRiemann->primL[q] = cell_prim(theRiemann->cL,q) + cell_grad(theRiemann->cL,q)*deltaL + cell_gradp(theRiemann->cL,q)*dpL;
    theRiemann->primR[q] = cell_prim(theRiemann->cR,q) - cell_grad(theRiemann->cR,q)*deltaR - cell_gradp(theRiemann->cR,q)*dpR;
  }
  if ((sim_runtype(theSim)==1)&&(BNORM_AVG==1)){
    if (direction==RDIRECTION){
      theRiemann->primL[BRR] = 0.5*(theRiemann->primL[BRR] + theRiemann->primR[BRR]);
      theRiemann->primR[BRR] = 0.5*(theRiemann->primL[BRR] + theRiemann->primR[BRR]);
    } else if (direction==PDIRECTION){
      theRiemann->primL[BPP] = 0.5*(theRiemann->primL[BPP] + theRiemann->primR[BPP]);
      theRiemann->primR[BPP] = 0.5*(theRiemann->primL[BPP] + theRiemann->primR[BPP]);
    } else if (direction==ZDIRECTION){
      theRiemann->primL[BZZ] = 0.5*(theRiemann->primL[BZZ] + theRiemann->primR[BZZ]);
      theRiemann->primR[BZZ] = 0.5*(theRiemann->primL[BZZ] + theRiemann->primR[BZZ]);
    }
  }
}

void riemann_setup_p(struct Riemann * theRiemann,struct Cell *** theCells,struct Sim * theSim,int i,int j_low,int k,int direction){
  theRiemann->n[direction]=1; // set
  int NUM_Q = sim_NUM_Q(theSim);

  int j_hi;
  if (j_low == sim_N_p(theSim,i)-1){
    j_hi = 0;
  } else{
    j_hi = j_low+1;
  }
  theRiemann->cL = cell_single(theCells,i,j_low,k);
  theRiemann->cR = cell_single(theCells,i,j_hi ,k);
  double dpL = cell_dphi(theRiemann->cL);
  double dpR = cell_dphi(theRiemann->cR);
  double zm = sim_FacePos(theSim,k-1,Z_DIR);
  double zp = sim_FacePos(theSim,k,Z_DIR);
  double dz = zp-zm;
  double rm = sim_FacePos(theSim,i-1,R_DIR);
  double rp = sim_FacePos(theSim,i,R_DIR);
  double dr = rp-rm;
  double r = .5*(rp+rm);
  theRiemann->dA = dr*dz;
  theRiemann->r = r; 
  theRiemann->cm = cell_tiph(theRiemann->cL);

  theRiemann->r_cell_L = r;
  theRiemann->r_cell_R = r;

  int q;
  for (q=0;q<NUM_Q;++q){
    theRiemann->primL[q] = cell_prim(theRiemann->cL,q) + 0.5*cell_gradp(theRiemann->cL,q)*dpL;
    theRiemann->primR[q] = cell_prim(theRiemann->cR,q) - 0.5*cell_gradp(theRiemann->cR,q)*dpR;
  }
  if ((sim_runtype(theSim)==1)&&(BNORM_AVG==1)){
    if (direction==RDIRECTION){
      theRiemann->primL[BRR] = 0.5*(theRiemann->primL[BRR] + theRiemann->primR[BRR]);
      theRiemann->primR[BRR] = 0.5*(theRiemann->primL[BRR] + theRiemann->primR[BRR]);
    } else if (direction==PDIRECTION){
      theRiemann->primL[BPP] = 0.5*(theRiemann->primL[BPP] + theRiemann->primR[BPP]);
      theRiemann->primR[BPP] = 0.5*(theRiemann->primL[BPP] + theRiemann->primR[BPP]);
    } else if (direction==ZDIRECTION){
      theRiemann->primL[BZZ] = 0.5*(theRiemann->primL[BZZ] + theRiemann->primR[BZZ]);
      theRiemann->primR[BZZ] = 0.5*(theRiemann->primL[BZZ] + theRiemann->primR[BZZ]);
    }
  }

}


void riemann_AddFlux(struct Riemann * theRiemann, struct Sim *theSim,struct GravMass *theGravMasses,double dt ){
  int NUM_Q = sim_NUM_Q(theSim);
  double GAMMALAW = sim_GAMMALAW(theSim);
  double DIVB_CH = sim_DIVB_CH(theSim);

  double Bpack[6];
  riemann_set_vel(theRiemann,theSim,theGravMasses,theRiemann->r,Bpack,GAMMALAW,DIVB_CH);

  double Bk_face,Psi_face;
  if (sim_runtype(theSim)==1){
    if (theRiemann->n[RDIRECTION]){
      Bk_face = 0.5*(theRiemann->primL[BRR]+theRiemann->primR[BRR]);  
    } else if (theRiemann->n[PDIRECTION]){
      Bk_face = 0.5*(theRiemann->primL[BPP]+theRiemann->primR[BPP]);
    } else if (theRiemann->n[ZDIRECTION]){
      Bk_face = 0.5*(theRiemann->primL[BZZ]+theRiemann->primR[BZZ]);
    }
    Psi_face = 0.5*(theRiemann->primL[PSI]+theRiemann->primR[PSI]);
  }
  double w;
  if (theRiemann->n[PDIRECTION]){
    //if( sim_MOVE_CELLS(theSim) == C_WRIEMANN ) cell_add_wiph(theRiemann->cL,theRiemann->Ss);
    w = cell_wiph(theRiemann->cL);
    //w = 0.0;
  } else{
    w = 0.0;
  }
  // which state of the riemann problem are we in?
  riemann_set_state(theRiemann,w);


  if (theRiemann->state==LEFT){
    riemann_set_flux( theRiemann , theSim, GAMMALAW,DIVB_CH,LEFT);//in this case, we only need FL
    cell_prim2cons( theRiemann->primL , theRiemann->UL , theRiemann->r , 1.0 ,theSim);
    int q;
    for (q=0;q<sim_NUM_Q(theSim) ; ++q ){
      theRiemann->F[q] = theRiemann->FL[q] - w*theRiemann->UL[q];// w is only nonzero when we are in phi direction
    }
  } else if (theRiemann->state==RIGHT){
    riemann_set_flux( theRiemann , theSim, GAMMALAW,DIVB_CH,RIGHT);//in this case, we only need FR
    cell_prim2cons( theRiemann->primR , theRiemann->UR , theRiemann->r , 1.0 ,theSim);
    int q;
    for (q=0;q<sim_NUM_Q(theSim) ; ++q ){
      theRiemann->F[q] = theRiemann->FR[q] - w*theRiemann->UR[q];// w is only nonzero when we are in phi direction
    }
  } else{
    if (sim_Riemann(theSim)==HLL){
      riemann_set_flux( theRiemann , theSim, GAMMALAW,DIVB_CH,LEFT);  //we need both
      riemann_set_flux( theRiemann , theSim, GAMMALAW,DIVB_CH,RIGHT);    
      cell_prim2cons( theRiemann->primL , theRiemann->UL , theRiemann->r , 1.0 ,theSim);//we need both
      cell_prim2cons( theRiemann->primR , theRiemann->UR , theRiemann->r , 1.0 ,theSim);
      riemann_set_star_hll(theRiemann,theSim);// get Ustar and Fstar
    } else if (sim_Riemann(theSim)==HLLC){
      if (theRiemann->state==LEFTSTAR){
        riemann_set_flux( theRiemann , theSim, GAMMALAW,DIVB_CH,LEFT);//in this case, we only need FL
        cell_prim2cons( theRiemann->primL , theRiemann->UL , theRiemann->r , 1.0 ,theSim);
        riemann_set_star_hllc(theRiemann,theSim,Bpack,GAMMALAW);// get Ustar and Fstar
      } else if (theRiemann->state==RIGHTSTAR){
        riemann_set_flux( theRiemann , theSim, GAMMALAW,DIVB_CH,RIGHT);//in this case, we only need FR      
        cell_prim2cons( theRiemann->primR , theRiemann->UR , theRiemann->r , 1.0 ,theSim);
        riemann_set_star_hllc(theRiemann,theSim,Bpack,GAMMALAW);// get Ustar and Fstar
      }
    } else{
      printf("ERROR\n");
      exit(0);
    }
    int q;
    for (q=0;q<sim_NUM_Q(theSim) ; ++q ){
      theRiemann->F[q] = theRiemann->Fstar[q] - w*theRiemann->Ustar[q];// w is only nonzero when we are in phi direction
    }
  }


  // viscous flux terms
  if (sim_EXPLICIT_VISCOSITY(theSim)>0.0){
    if (VISC_OLD==1){
      riemann_visc_flux_old(theRiemann,theSim );
    } else{
      riemann_visc_flux(theRiemann,theSim,theGravMasses );  
    }
  }

  int q;
  for( q=0 ; q<NUM_Q ; ++q ){
    cell_add_cons(theRiemann->cL,q,-dt*theRiemann->dA*theRiemann->F[q]);
    cell_add_cons(theRiemann->cR,q,dt*theRiemann->dA*theRiemann->F[q]);
  }

  if (VISC_OLD==1){
    cell_add_cons(theRiemann->cL,SRR,-dt*theRiemann->dA*theRiemann->Fvisc[SRR]);
    cell_add_cons(theRiemann->cR,SRR, dt*theRiemann->dA*theRiemann->Fvisc[SRR]);
  }else {
    cell_add_cons(theRiemann->cL,SRR,-dt*theRiemann->dA*theRiemann->Fvisc[SRR]/theRiemann->r_cell_L);
    cell_add_cons(theRiemann->cR,SRR, dt*theRiemann->dA*theRiemann->Fvisc[SRR]/theRiemann->r_cell_R);
  }

  cell_add_cons(theRiemann->cL,LLL,-dt*theRiemann->dA*theRiemann->Fvisc[LLL]);
  cell_add_cons(theRiemann->cR,LLL, dt*theRiemann->dA*theRiemann->Fvisc[LLL]);

  //viscous heating
  cell_add_cons(theRiemann->cL,TAU,-dt*theRiemann->dA*theRiemann->Fvisc[TAU]);
  cell_add_cons(theRiemann->cR,TAU,+dt*theRiemann->dA*theRiemann->Fvisc[TAU]);

  if (sim_runtype(theSim)==1){
    int direction;
    if (theRiemann->n[RDIRECTION]==1){
      direction=RDIRECTION;
    }else if (theRiemann->n[PDIRECTION]==1){
      direction=PDIRECTION;
    } else if (theRiemann->n[ZDIRECTION]==1){
      direction=ZDIRECTION;
    }

    cell_add_divB(theRiemann->cL,theRiemann->dA*Bk_face);
    cell_add_divB(theRiemann->cR,-theRiemann->dA*Bk_face);
    cell_add_GradPsi(theRiemann->cL,direction,Psi_face*theRiemann->dA/theRiemann->r);
    cell_add_GradPsi(theRiemann->cR,direction,-Psi_face*theRiemann->dA/theRiemann->r);
  }

}



