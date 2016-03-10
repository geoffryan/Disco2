#define CELL_PRIVATE_DEFS
#define EOS_PRIVATE_DEFS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../Headers/Cell.h"
#include "../Headers/Sim.h"
#include "../Headers/EOS.h"
#include "../Headers/Face.h"
#include "../Headers/GravMass.h"
#include "../Headers/Metric.h"
#include "../Headers/MPIsetup.h"
#include "../Headers/TimeStep.h"
#include "../Headers/header.h"


void cell_boundary_outflow_r_inner( struct Cell *** theCells , struct Face * theFaces ,struct Sim * theSim,struct MPIsetup * theMPIsetup, struct TimeStep * theTimeStep ){
  int NUM_Q = sim_NUM_Q(theSim);

  int n,q;
  int i,j,k;
  double r_face,r_face_m1,r_cell;

  if( mpisetup_check_rin_bndry(theMPIsetup) ){

    if (sim_NoInnerBC(theSim)!=1){ // if the global inner radius is set negative, we don't apply an inner BC
      //for( i=0; i>=0 ; --i ){
      for( i=sim_Nghost_min(theSim,R_DIR)-1 ; i>=0 ; --i ){
        r_face=sim_FacePos(theSim,i,R_DIR);
        r_face_m1=sim_FacePos(theSim,i-1,R_DIR);
        r_cell = 0.5*(r_face+r_face_m1);
        //printf("r_cell inner: %e\n",r_cell);

        for( n=timestep_n(theTimeStep,i,R_DIR) ; n<timestep_n(theTimeStep,i+1,R_DIR) ; ++n ){
          for( q=0 ; q<NUM_Q ; ++q ){
            face_L_pointer(theFaces,n)->prim[q] = 0.0;
          }
        } 
        for( n=timestep_n(theTimeStep,i,R_DIR) ; n<timestep_n(theTimeStep,i+1,R_DIR) ; ++n ){
          struct Cell * cL = face_L_pointer(theFaces,n);
          struct Cell * cR = face_R_pointer(theFaces,n);
          for( q=0 ; q<NUM_Q ; ++q ){
            cL->prim[q] += cR->prim[q]*face_dA(theFaces,n);
          }
        }

        for( k=0 ; k<sim_N(theSim,Z_DIR) ; ++k ){
          double zp = sim_FacePos(theSim,k,Z_DIR);
          double zm = sim_FacePos(theSim,k-1,Z_DIR);
          double dz = zp-zm;
          for( j=0 ; j<sim_N_p(theSim,i) ; ++j ){
            double dA = dz*r_face*theCells[k][i][j].dphi;
            for( q=0 ; q<NUM_Q ; ++q ){
              theCells[k][i][j].prim[q] /= dA;
            }
            //printf("r: %e, theCells[k][i][j].prim[URR]: %e\n",r_face,theCells[k][i][j].prim[URR]);
            if( theCells[k][i][j].prim[URR] > 0.0 ) theCells[k][i][j].prim[URR] = 0.0;
            if (KEP_BNDRY==1){
              theCells[k][i][j].prim[UPP] = pow(r_cell,-1.5);          
            }
          }
        }
      }
    }
  }
}

void cell_boundary_outflow_r_outer( struct Cell *** theCells , struct Face * theFaces ,struct Sim * theSim,struct MPIsetup * theMPIsetup, struct TimeStep * theTimeStep ){

  int iN = sim_N(theSim,R_DIR);
  int iNmg = sim_N(theSim,R_DIR) - sim_Nghost_max(theSim,R_DIR);
  int Nf = timestep_n(theTimeStep,sim_N(theSim,R_DIR)-1,R_DIR);
  int NUM_Q = sim_NUM_Q(theSim);
  int n1 = timestep_n(theTimeStep,sim_N(theSim,R_DIR)-2,R_DIR);

  int n,q;
  int i,j,k;
  double r_face,r_face_p1,r_cell;

  if( mpisetup_check_rout_bndry(theMPIsetup) ){
    for(i=iNmg-1; i<iN-1; i++){
        //printf("Boundary outflow r outer %d\n", i);
      n1 = timestep_n(theTimeStep,i,R_DIR);
      Nf = timestep_n(theTimeStep,i+1,R_DIR);
      for( n=n1 ; n<Nf ; ++n ){
        for( q=0 ; q<NUM_Q ; ++q ){
          face_R_pointer(theFaces,n)->prim[q] = 0.0;
        }
      }
      for( n=n1 ; n<Nf ; ++n ){
        struct Cell * cL = face_L_pointer(theFaces,n);
        struct Cell * cR = face_R_pointer(theFaces,n);
        for( q=0 ; q<NUM_Q ; ++q ){
          cR->prim[q] += cL->prim[q]*face_dA(theFaces,n);
        }
      }
      //r_face = sim_FacePos(theSim,sim_N(theSim,R_DIR)-2,R_DIR);
      //r_face_p1 = sim_FacePos(theSim,sim_N(theSim,R_DIR)-1,R_DIR);
      r_face = sim_FacePos(theSim,i,R_DIR);
      r_face_p1 = sim_FacePos(theSim,i+1,R_DIR);
      r_cell = 0.5*(r_face+r_face_p1);
      //printf("r_cell outer: %e\n",r_cell);
      for( k=0 ; k<sim_N(theSim,Z_DIR) ; ++k ){
        double zp = sim_FacePos(theSim,k,Z_DIR);
        double zm = sim_FacePos(theSim,k-1,Z_DIR);
        double dz = zp-zm;
        for( j=0 ; j<sim_N_p(theSim,i+1) ; ++j ){
          double dA = dz*r_face*theCells[k][i+1][j].dphi;
          for( q=0 ; q<NUM_Q ; ++q ){
            theCells[k][i+1][j].prim[q] /= dA;
          }
          //theCells[k][sim_N(theSim,R_DIR)-1][j].prim[URR] *= -1.;
          if( theCells[k][i+1][j].prim[URR] < 0.0 )
            theCells[k][i+1][j].prim[URR] = 0.0;
          if (KEP_BNDRY==1){
            theCells[k][i+1][j].prim[UPP] = pow(r_cell,-1.5);
          }
        }
      }
    }
  }
}

void cell_boundary_outflow_z_bot( struct Cell *** theCells , struct Face * theFaces, struct Sim * theSim,struct MPIsetup * theMPIsetup,struct TimeStep * theTimeStep ){
  int NUM_Q = sim_NUM_Q(theSim);

  int j,i;
  int n0 = timestep_n(theTimeStep,1,Z_DIR);

  int n,q;

  if(mpisetup_check_zbot_bndry(theMPIsetup)){
    for( n=0 ; n<n0 ; ++n ){
      for( q=0 ; q<NUM_Q ; ++q ){
        face_L_pointer(theFaces,n)->prim[q] = 0.0;
      }
    }
    for( n=0 ; n<n0 ; ++n ){
      struct Cell * cL = face_L_pointer(theFaces,n);
      struct Cell * cR = face_R_pointer(theFaces,n);
      for( q=0 ; q<NUM_Q ; ++q ){
        cL->prim[q] += cR->prim[q]*face_dA(theFaces,n);
      }
    }

    for( i=0 ; i<sim_N(theSim,R_DIR) ; ++i ){
      double rp = sim_FacePos(theSim,i,R_DIR);
      double rm = sim_FacePos(theSim,i-1,R_DIR);
      for( j=0 ; j<sim_N_p(theSim,i) ; ++j ){
        double dA = .5*(rp*rp-rm*rm)*theCells[0][i][j].dphi;
        for( q=0 ; q<NUM_Q ; ++q ){
          theCells[0][i][j].prim[q] /= dA;
        }
        theCells[0][i][j].prim[UZZ] *= -1.0;
      }
    }
  }
}


void cell_boundary_outflow_z_top( struct Cell *** theCells , struct Face * theFaces, struct Sim * theSim,struct MPIsetup * theMPIsetup,struct TimeStep * theTimeStep ){
  int NUM_Q = sim_NUM_Q(theSim);

  int j,i;
  int Nf = timestep_n(theTimeStep,sim_N(theSim,Z_DIR)-1,Z_DIR);
  int n1 = timestep_n(theTimeStep,sim_N(theSim,Z_DIR)-2,Z_DIR);

  int n,q;

  if(mpisetup_check_ztop_bndry(theMPIsetup)){
    for( n=n1 ; n<Nf ; ++n ){
      for( q=0 ; q<NUM_Q ; ++q ){
        face_R_pointer(theFaces,n)->prim[q] = 0.0;
      }
    }
    for( n=n1 ; n<Nf ; ++n ){
      struct Cell * cL = face_L_pointer(theFaces,n);
      struct Cell * cR = face_R_pointer(theFaces,n);
      for( q=0 ; q<NUM_Q ; ++q ){
        cR->prim[q] += cL->prim[q]*face_dA(theFaces,n);
      }
    }


    for( i=0 ; i<sim_N(theSim,R_DIR) ; ++i ){
      double rp = sim_FacePos(theSim,i,R_DIR);
      double rm = sim_FacePos(theSim,i-1,R_DIR);
      for( j=0 ; j<sim_N_p(theSim,i) ; ++j ){
        double dA = .5*(rp*rp-rm*rm)*theCells[sim_N(theSim,Z_DIR)-1][i][j].dphi;
        for( q=0 ; q<NUM_Q ; ++q ){
          theCells[sim_N(theSim,Z_DIR)-1][i][j].prim[q] /= dA;
        }
        theCells[sim_N(theSim,Z_DIR)-1][i][j].prim[UZZ] *= -1.0;
      }
    }
  }
}

void cell_boundary_fixed_r_inner( struct Cell *** theCells, struct Sim *theSim,struct MPIsetup * theMPIsetup, 
    void (*single_init_ptr)(struct Cell *,struct Sim *,int,int,int) ){
  int i,j,k;
  if (sim_NoInnerBC(theSim)!=1){ // if the global inner radius is set negative, we don't apply an inner BC
    if(mpisetup_check_rin_bndry(theMPIsetup)){
      for( i=0 ; i<sim_Nghost_min(theSim,R_DIR) ; ++i ){
        for( k=0 ; k<sim_N(theSim,Z_DIR) ; ++k ){
          for( j=0 ; j<sim_N_p(theSim,i) ; ++j ){
            (*single_init_ptr)(&(theCells[k][i][j]),theSim,i,j,k);
          }
        }
      }
    }
  }

  //TODO: Find a better way of doing this.
  if(sim_InitialDataType(theSim)==GBONDI && sim_NoInnerBC(theSim))
      if(mpisetup_check_rin_bndry(theMPIsetup))
          for(k=0; k<sim_N(theSim,Z_DIR); k++)
          {
              double zm = sim_FacePos(theSim,k-1,Z_DIR);
              double zp = sim_FacePos(theSim,k,Z_DIR);
              if(zm < 0 && zp > 0)
                for(j=0; j<sim_N_p(theSim,0); j++)
                    (*single_init_ptr)(&(theCells[k][0][j]),theSim,0,j,k);
          }
}

void cell_boundary_fixed_r_outer( struct Cell *** theCells, struct Sim *theSim,struct MPIsetup * theMPIsetup, 
    void (*single_init_ptr)(struct Cell *,struct Sim *,int,int,int) ){
    int i,j,k;
  if( mpisetup_check_rout_bndry(theMPIsetup) ){
    for( i=sim_N(theSim,R_DIR)-1 ; i>sim_N(theSim,R_DIR)-sim_Nghost_max(theSim,R_DIR)-1 ; --i ){
      for( k=0 ; k<sim_N(theSim,Z_DIR) ; ++k ){
        for( j=0 ; j<sim_N_p(theSim,i) ; ++j ){
          (*single_init_ptr)(&(theCells[k][i][j]),theSim,i,j,k);
        }
      }
    }
  }

}

void cell_boundary_fixed_z_bot( struct Cell *** theCells, struct Sim *theSim,struct MPIsetup * theMPIsetup,
    void (*single_init_ptr)(struct Cell *,struct Sim *,int,int,int)  ){
  int i,j,k;
  if(mpisetup_check_zbot_bndry(theMPIsetup)){
    for( i=0 ; i<sim_N(theSim,R_DIR) ; ++i ){
      for( k=0 ; k<sim_Nghost_min(theSim,Z_DIR) ; ++k ){
        for( j=0 ; j<sim_N_p(theSim,i) ; ++j ){
          (*single_init_ptr)(&(theCells[k][i][j]),theSim,i,j,k);
        }
      }
    }
  }
}

void cell_boundary_fixed_z_top( struct Cell *** theCells, struct Sim *theSim,struct MPIsetup * theMPIsetup,
    void (*single_init_ptr)(struct Cell *,struct Sim *,int,int,int)  ){
  int i,j,k;
  if( mpisetup_check_ztop_bndry(theMPIsetup) ){
    for( i=0 ; i<sim_N(theSim,R_DIR) ; ++i ){
      for( k=sim_N(theSim,Z_DIR)-1 ; k>sim_N(theSim,Z_DIR)-sim_Nghost_max(theSim,Z_DIR)-1 ; --k ){
        for( j=0 ; j<sim_N_p(theSim,i) ; ++j ){
          (*single_init_ptr)(&(theCells[k][i][j]),theSim,i,j,k);
        }
      }
    }
  }

}

void cell_boundary_ssprofile_r_inner( struct Cell *** theCells, struct Sim *theSim,struct MPIsetup * theMPIsetup)
{
    int i,j,k,q;
    if (sim_NoInnerBC(theSim)!=1) // if the global inner radius is set negative, we don't apply an inner BC
    {
        if(mpisetup_check_rin_bndry(theMPIsetup))
        {
            int iIn = sim_Nghost_min(theSim, R_DIR);
            double rIn = 0.5*(sim_FacePos(theSim,iIn,R_DIR) + sim_FacePos(theSim,iIn-1,R_DIR));
            double *primIn = theCells[0][iIn][0].prim;
            for( i=0 ; i<iIn ; ++i )
            {
                double rp = sim_FacePos(theSim,i,R_DIR);
                double rm = sim_FacePos(theSim,i-1,R_DIR);
                double r = 0.5*(rp+rm);
                for( k=0 ; k<sim_N(theSim,Z_DIR) ; ++k )
                    for( j=0 ; j<sim_N_p(theSim,i) ; ++j )
                    {
                        if(sim_InitialDataType(theSim) == SSDISC)
                        {
                            if(sim_InitPar0(theSim)==0)
                            {
                                theCells[k][i][j].prim[RHO] = primIn[RHO] * pow(r/rIn, -0.6);
                                theCells[k][i][j].prim[PPP] = primIn[PPP] * pow(r/rIn, -1.5);
                                theCells[k][i][j].prim[URR] = primIn[URR] * pow(r/rIn, -0.4);
                                theCells[k][i][j].prim[UPP] = primIn[UPP] * pow(r/rIn, -1.5);
                                theCells[k][i][j].prim[UZZ] = 0.0;
                                for(q=sim_NUM_C(theSim); q < sim_NUM_Q(theSim); q++)
                                    theCells[k][i][j].prim[q] = primIn[q] * pow(r/rIn, -0.6);
                            }
                            else if(sim_InitPar0(theSim)==1)
                            {
                                theCells[k][i][j].prim[RHO] = primIn[RHO] * pow(r/rIn, -1.5);
                                theCells[k][i][j].prim[PPP] = primIn[PPP] * pow(r/rIn, -1.5);
                                theCells[k][i][j].prim[URR] = primIn[URR] * pow(r/rIn, 0.5);
                                theCells[k][i][j].prim[UPP] = primIn[UPP] * pow(r/rIn, -1.5);
                                theCells[k][i][j].prim[UZZ] = 0.0;
                                for(q=sim_NUM_C(theSim); q < sim_NUM_Q(theSim); q++)
                                    theCells[k][i][j].prim[q] = primIn[q] * pow(r/rIn, -1.5);
                            }
                        }
                        else if(sim_InitialDataType(theSim) == NTDISC)
                        {
                            double M = sim_GravM(theSim);
                            double rs = 6*M;
                            double C = (1.0-3*M/r)/(1.0-3*M/rIn);
                            double D = (1.0-2*M/r)/(1.0-2*M/rIn);
                            double P = (1.0 - sqrt(rs/r) + sqrt(3*M/r)*(atanh(sqrt(3*M/r))-atanh(sqrt(3*M/rs)))) / (1.0 - sqrt(rs/rIn) + sqrt(3*M/rIn)*(atanh(sqrt(3*M/rIn))-atanh(sqrt(3*M/rs))));
                            if(sim_InitPar0(theSim)==0)
                            {
                                theCells[k][i][j].prim[RHO] = primIn[RHO] * pow(r/rIn,-0.6) * pow(C,0.6) * pow(D,-1.6) * pow(P,0.6);
                                theCells[k][i][j].prim[PPP] = primIn[PPP] * pow(r/rIn,-1.5) * sqrt(C) * P / (D*D);
                                theCells[k][i][j].prim[URR] = primIn[URR] * pow(r/rIn,-0.4) * pow(C,-0.1) * pow(D,1.6) * pow(P,-0.6);
                                theCells[k][i][j].prim[UPP] = primIn[UPP] * pow(r/rIn,-1.5);
                                theCells[k][i][j].prim[UZZ] = 0.0;
                                for(q=sim_NUM_C(theSim); q < sim_NUM_Q(theSim); q++)
                                    theCells[k][i][j].prim[q] = primIn[q] * pow(r/rIn,-0.6) * pow(C,0.6) * pow(D,-1.6) * pow(P,0.6);
                            }
                        }
                        else if (sim_InitialDataType(theSim) == ADAF)
                        {
                            theCells[k][i][j].prim[RHO] = primIn[RHO] * pow(r/rIn,-0.5);
                            theCells[k][i][j].prim[PPP] = primIn[PPP] * pow(r/rIn,-1.5);
                            theCells[k][i][j].prim[URR] = primIn[URR] * pow(r/rIn,-0.5);
                            theCells[k][i][j].prim[UPP] = primIn[UPP] * pow(r/rIn,-1.5);
                        }

                    }
            }
        }
    }
}

void cell_boundary_ssprofile_r_outer( struct Cell *** theCells, struct Sim *theSim,struct MPIsetup * theMPIsetup)
{
    int i,j,k,q;
    if(mpisetup_check_rout_bndry(theMPIsetup))
    {
        int iOut = sim_N(theSim,R_DIR)-sim_Nghost_max(theSim,R_DIR)-1;
        double rOut = 0.5*(sim_FacePos(theSim,iOut,R_DIR) + sim_FacePos(theSim,iOut-1,R_DIR));
        double *primOut = theCells[0][iOut][0].prim;
        for(i=iOut+1; i<sim_N(theSim,R_DIR); i++)
        {
            double rp = sim_FacePos(theSim,i,R_DIR);
            double rm = sim_FacePos(theSim,i-1,R_DIR);
            double r = 0.5*(rp+rm);
            for( k=0 ; k<sim_N(theSim,Z_DIR) ; ++k )
                for( j=0 ; j<sim_N_p(theSim,i) ; ++j )
                {
                    if(sim_InitialDataType(theSim) == SSDISC)
                    {
                        if(sim_InitPar0(theSim)==0)
                        {
                            theCells[k][i][j].prim[RHO] = primOut[RHO] * pow(r/rOut, -0.6);
                            theCells[k][i][j].prim[PPP] = primOut[PPP] * pow(r/rOut, -1.5);
                            theCells[k][i][j].prim[URR] = primOut[URR] * pow(r/rOut, -0.4);
                            theCells[k][i][j].prim[UPP] = primOut[UPP] * pow(r/rOut, -1.5);
                            theCells[k][i][j].prim[UZZ] = 0.0;
                            for(q=sim_NUM_C(theSim); q < sim_NUM_Q(theSim); q++)
                                theCells[k][i][j].prim[q] = primOut[q] * pow(r/rOut, -0.6);
                        }
                        else if(sim_InitPar0(theSim)==1)
                        {
                            theCells[k][i][j].prim[RHO] = primOut[RHO] * pow(r/rOut, -1.5);
                            theCells[k][i][j].prim[PPP] = primOut[PPP] * pow(r/rOut, -1.5);
                            theCells[k][i][j].prim[URR] = primOut[URR] * pow(r/rOut, 0.5);
                            theCells[k][i][j].prim[UPP] = primOut[UPP] * pow(r/rOut, -1.5);
                            theCells[k][i][j].prim[UZZ] = 0.0;
                            for(q=sim_NUM_C(theSim); q < sim_NUM_Q(theSim); q++)
                                theCells[k][i][j].prim[q] = primOut[q] * pow(r/rOut, -1.5);
                        }
                    }
                    
                    else if(sim_InitialDataType(theSim) == NTDISC)
                    {
                        double M = sim_GravM(theSim);
                        double rs = 6*M;
                        double C = (1.0-3*M/r)/(1.0-3*M/rOut);
                        double D = (1.0-2*M/r)/(1.0-2*M/rOut);
                        double P = (1.0 - sqrt(rs/r) + sqrt(3*M/r)*(atanh(sqrt(3*M/r))-atanh(sqrt(3*M/rs)))) / (1.0 - sqrt(rs/rOut) + sqrt(3*M/rOut)*(atanh(sqrt(3*M/rOut))-atanh(sqrt(3*M/rs))));
                        if(sim_InitPar0(theSim)==0)
                        {
                            theCells[k][i][j].prim[RHO] = primOut[RHO] * pow(r/rOut,-0.6) * pow(C,0.6) * pow(D,-1.6) * pow(P,0.6);
                            theCells[k][i][j].prim[PPP] = primOut[PPP] * pow(r/rOut,-1.5) * sqrt(C) * P / (D*D);
                            theCells[k][i][j].prim[URR] = primOut[URR] * pow(r/rOut,-0.4) * pow(C,-0.1) * pow(D,1.6) * pow(P,-0.6);
                            theCells[k][i][j].prim[UPP] = primOut[UPP] * pow(r/rOut,-1.5);
                            theCells[k][i][j].prim[UZZ] = 0.0;
                            for(q=sim_NUM_C(theSim); q < sim_NUM_Q(theSim); q++)
                                theCells[k][i][j].prim[q] = primOut[q] * pow(r/rOut,-0.6) * pow(C,0.6) * pow(D,-1.6) * pow(P,0.6);
                        }
                    }
                    else if (sim_InitialDataType(theSim) == ADAF)
                    {
                        theCells[k][i][j].prim[RHO] = primOut[RHO] * pow(r/rOut,-0.5);
                        theCells[k][i][j].prim[PPP] = primOut[PPP] * pow(r/rOut,-1.5);
                        theCells[k][i][j].prim[URR] = primOut[URR] * pow(r/rOut,-0.5);
                        theCells[k][i][j].prim[UPP] = primOut[UPP] * pow(r/rOut,-1.5);
                    }
                }
        }
    }
}

void cell_boundary_linear_r_inner(struct Cell ***theCells, 
                struct Face *theFaces, struct Sim *theSim, 
                struct MPIsetup *theMPIsetup, struct TimeStep * theTimeStep )
{
    int NUM_Q = sim_NUM_Q(theSim);

    int n,q;
    int j,k;

    // if the global inner radius is set negative, we don't apply an inner BC
    if(mpisetup_check_rin_bndry(theMPIsetup) && sim_NoInnerBC(theSim)!=1)
    {
        double r2, r1, r0, r0m, dr;
        r2 = sim_FacePos(theSim, 2, R_DIR); 
        r1 = sim_FacePos(theSim, 1, R_DIR); 
        r0 = sim_FacePos(theSim, 0, R_DIR); 
        r0m = sim_FacePos(theSim, -1, R_DIR); 

        //Clear radial gradients in Annulus 1.
        for(k = 0; k < sim_N(theSim, Z_DIR); k++)
            for(j = 0; j < sim_N_p(theSim,1); j++)
                for(q = 0; q < NUM_Q; q++)
                    theCells[k][1][j].gradr[q] = 0.0;

        //Radial gradients in Annulus 1 updated with area weighted derivatives
        //between Annulus 1 & 2;
        dr = 0.5*(r2-r0);
        for(n = timestep_n(theTimeStep,1,R_DIR); 
                n < timestep_n(theTimeStep,2,R_DIR); n++)
        {
            double dA = face_dA(theFaces,n);
            struct Cell *cL = face_L_pointer(theFaces,n);
            struct Cell *cR = face_R_pointer(theFaces,n);

            for(q = 0; q < NUM_Q; q++)
                cL->gradr[q] += dA*(cR->prim[q]-cL->prim[q])/dr;
        }
        for(k = 0; k < sim_N(theSim, Z_DIR); k++)
        {
            double dz = sim_FacePos(theSim, k, Z_DIR)
                        - sim_FacePos(theSim, k-1, Z_DIR);
            for(j = 0; j < sim_N_p(theSim,1); j++)
            {
                double dA = r1 * theCells[k][1][j].dphi * dz;
                for(q = 0; q < NUM_Q; q++)
                    theCells[k][1][j].gradr[q] /= dA;
            }
        }

        //Clear prims in Annulus 0
        for(k = 0; k < sim_N(theSim, Z_DIR); k++)
            for(j = 0; j < sim_N_p(theSim,0); j++)
                for(q = 0; q < NUM_Q; q++)
                    theCells[k][0][j].prim[q] = 0.0;

        //Annulus 0 updated to area-weighted extrapolation of Annulus 1.
        dr = r1 - r0m;
        for(n = timestep_n(theTimeStep,0,R_DIR); 
                n < timestep_n(theTimeStep,1,R_DIR); n++)
        {
            double dA = face_dA(theFaces,n);
            struct Cell *cL = face_L_pointer(theFaces,n);
            struct Cell *cR = face_R_pointer(theFaces,n);

            for(q = 0; q < NUM_Q; q++)
            {
                double qextrap;
                if(q == URR || q == UPP || q == UZZ)
                    qextrap = cR->prim[q] - cR->gradr[q] * dr;
                else
                    qextrap = cR->prim[q] - 0.0*cR->gradr[q] * dr;
                if((q == RHO || q == PPP) && qextrap < 0.0)
                    qextrap = 0.5*cR->prim[q];
                cL->prim[q] += dA*qextrap;
            }
        }
        double r = 0.5*(r0m+r0);
        for(k = 0; k < sim_N(theSim, Z_DIR); k++)
        {
            double zp = sim_FacePos(theSim, k, Z_DIR);
            double zm = sim_FacePos(theSim, k-1, Z_DIR);
            double dz = zp-zm;
            double z = 0.5*(zm+zp);

            for(j = 0; j < sim_N_p(theSim,0); j++)
            {
                double phi = theCells[k][0][j].tiph
                             - 0.5*theCells[k][0][j].dphi;
                double dA = r0*theCells[k][0][j].dphi * dz;
                for(q = 0; q < NUM_Q; q++)
                    theCells[k][0][j].prim[q] /= dA;

                if(sim_Background(theSim) != NEWTON)
                {
                    double v[3];
                    v[0] = theCells[k][0][j].prim[URR];
                    v[1] = theCells[k][0][j].prim[UPP];
                    v[2] = theCells[k][0][j].prim[UZZ];
                    struct Metric *g = metric_create(time_global, r, phi, z,
                                                        theSim);
                    int err = metric_fixV(g, v, 5.0);
                    if(err)
                    {
                        printf("Speed reset in LinearBC. r=%.12g phi=%.12g z=%.12g\n", r, phi, z);
                        theCells[k][0][j].prim[URR] = v[0];
                        theCells[k][0][j].prim[UPP] = v[1];
                        theCells[k][0][j].prim[UZZ] = v[2];
                    }
                    metric_destroy(g);
                }
            }
        }
    }
}

void cell_boundary_linear_r_outer( struct Cell *** theCells , struct Face * theFaces ,struct Sim * theSim,struct MPIsetup * theMPIsetup, struct TimeStep * theTimeStep ){
  int Nf = timestep_n(theTimeStep,sim_N(theSim,R_DIR)-1,R_DIR);
  int NUM_Q = sim_NUM_Q(theSim);
  int n1 = timestep_n(theTimeStep,sim_N(theSim,R_DIR)-2,R_DIR);

  int n,q;
  int j,k;
  double r_face,r_face_p1,r_cell;

  if( mpisetup_check_rout_bndry(theMPIsetup) ){
    for( n=n1 ; n<Nf ; ++n ){
      for( q=0 ; q<NUM_Q ; ++q ){
        face_R_pointer(theFaces,n)->prim[q] = 0.0;
      }
    }
    for( n=n1 ; n<Nf ; ++n ){
      struct Cell * cL = face_L_pointer(theFaces,n);
      struct Cell * cR = face_R_pointer(theFaces,n);
      for( q=0 ; q<NUM_Q ; ++q ){
        cR->prim[q] += cL->prim[q]*face_dA(theFaces,n);
      }
    }
    r_face = sim_FacePos(theSim,sim_N(theSim,R_DIR)-2,R_DIR);
    r_face_p1 = sim_FacePos(theSim,sim_N(theSim,R_DIR)-1,R_DIR);
    r_cell = 0.5*(r_face+r_face_p1);
    //printf("r_cell outer: %e\n",r_cell);
    for( k=0 ; k<sim_N(theSim,Z_DIR) ; ++k ){
      double zp = sim_FacePos(theSim,k,Z_DIR);
      double zm = sim_FacePos(theSim,k-1,Z_DIR);
      double dz = zp-zm;
      for( j=0 ; j<sim_N_p(theSim,sim_N(theSim,R_DIR)-1) ; ++j ){
        double dA = dz*r_face*theCells[k][sim_N(theSim,R_DIR)-1][j].dphi;
        for( q=0 ; q<NUM_Q ; ++q ){
          theCells[k][sim_N(theSim,R_DIR)-1][j].prim[q] /= dA;
        }
        //theCells[k][sim_N(theSim,R_DIR)-1][j].prim[URR] *= -1.;
        if( theCells[k][sim_N(theSim,R_DIR)-1][j].prim[URR] < 0.0 ) theCells[k][sim_N(theSim,R_DIR)-1][j].prim[URR] = 0.0;
        if (KEP_BNDRY==1){
          theCells[k][sim_N(theSim,R_DIR)-1][j].prim[UPP] = pow(r_cell,-1.5);
        }
      }
    }
  }
}

void cell_boundary_plunge_r_inner(struct Cell ***theCells, 
                struct Face *theFaces, struct Sim *theSim, 
                struct MPIsetup *theMPIsetup, struct TimeStep * theTimeStep )
{
    int i,j,k,mu;

    // if the global inner radius is set negative, we don't apply an inner BC
    if(mpisetup_check_rin_bndry(theMPIsetup) && sim_NoInnerBC(theSim)!=1)
    {
        double GAM = sim_GAMMALAW(theSim);
        double M = sim_GravM(theSim);
        int bg = sim_Background(theSim);
        if(bg == GRDISC)
            GAM = 0;
        
        //Calculate accretion rate & Isentropic constant
        i = sim_Nghost_min(theSim,R_DIR);
        double R = 0.5*(sim_FacePos(theSim,i,R_DIR)
                        + sim_FacePos(theSim,i-1,R_DIR));

        double mdot = 0.0;
        double Sdot = 0.0;

        for(k = 0; k < sim_N(theSim, Z_DIR); k++)
        {
            double zm = sim_FacePos(theSim, k-1, Z_DIR);
            double zp = sim_FacePos(theSim, k, Z_DIR);
            double z = 0.5*(zm+zp);
            double dz = zp-zm;

            for(j = 0; j < sim_N_p(theSim,1); j++)
            {
                double dphi = theCells[k][i][j].dphi;
                double phi = theCells[k][i][j].tiph - 0.5*dphi;

                struct Metric *g = metric_create(time_global, R, phi, z,
                                                        theSim);
                double b[3], v[3], u0;
                v[0] = theCells[k][i][j].prim[URR];
                v[1] = theCells[k][i][j].prim[UPP];
                v[2] = theCells[k][i][j].prim[UZZ];
                for(mu=0; mu<3; mu++)
                    b[mu] = metric_shift_u(g, mu);
                u0 = 1.0 / sqrt(-metric_g_dd(g,0,0) - 2*metric_dot3_u(g,b,v)
                                - metric_square3_u(g,v));
                metric_destroy(g);

                double Sigma, Pi, gam;

                if(bg == GRDISC)
                {
                    double rho = eos_rho(theCells[k][i][j].prim, theSim);
                    double P = eos_ppp(theCells[k][i][j].prim, theSim);
                    double eps = eos_eps(theCells[k][i][j].prim, theSim);
                    double H = sqrt(R*R*R*P / (M*(rho+rho*eps+P))) / u0;

                    /* So the thermodynamics of the GRDISC setup are pretty
                     * screwy. The EOS is written in terms of central
                     * central quantities, but the energy equation conserves
                     * the column-integrated energy.  This means that 
                     * the isentropic relation enforced by the adiabatic energy
                     * equation is for COLUMN INTEGRATED entropy, NOT the
                     * central entropy that would be easy to calculate.
                     *
                     * It's not easy/obvious at the moment how to write a good
                     * isentropic relation for this system, so I'll assume 
                     * a gas pressure dominated system because a) I know the
                     * answer there and b) the gas cools inside the ISCO 
                     * ANYWAYS, so gas pressure should be dominant.
                     */

                    gam = (rho+rho*eps+P)/P * eos_cs2(theCells[k][i][j].prim,
                                                        theSim);
                    GAM += gam*dphi*dz;
                    Sigma = rho*H;
                    Pi = P*H;
                }
                else
                {
                    double rho = theCells[k][i][j].prim[RHO];
                    double P = theCells[k][i][j].prim[PPP];

                    Sigma = rho;
                    Pi = P;
                    gam = GAM;
                }

                double s = log(pow(Pi/Sigma,1.0/(gam-1.0)) / Sigma);

                mdot += -Sigma * u0*v[0] * R*dphi*dz;
                Sdot += -s * Sigma * u0*v[0] * R*dphi*dz;
            }
        }
        double dZ = sim_FacePos(theSim, sim_N(theSim,Z_DIR)-1, Z_DIR)
                     - sim_FacePos(theSim, -1, Z_DIR);
        mdot /= 2*M_PI*dZ;
        Sdot /= 2*M_PI*dZ;

        double s = Sdot / mdot;
        if(bg == GRDISC)
            GAM /= 2*M_PI*dZ;

        //printf("%lg %lg\n", mdot, K);

        //Update ghost zones with plunging solution
        for(k = 0; k < sim_N(theSim, Z_DIR); k++)
        {
            double zm = sim_FacePos(theSim, k-1, Z_DIR);
            double zp = sim_FacePos(theSim, k, Z_DIR);
            double z = 0.5*(zm+zp);

            for(i=0; i<sim_Nghost_min(theSim,R_DIR); i++)
            {
                double rm = sim_FacePos(theSim, i-1, R_DIR);
                double rp = sim_FacePos(theSim, i, R_DIR);
                double r = 0.5*(rm+rp);
                
                for(j = 0; j < sim_N_p(theSim,1); j++)
                {
                    double dphi = theCells[k][i][j].dphi;
                    double phi = theCells[k][i][j].tiph - 0.5*dphi;

                    struct Metric *g = metric_create(time_global, r, phi, z,
                                                        theSim);
                    double U[4];
                    for(mu=0; mu<4; mu++)
                        U[mu] = metric_frame_U_u(g, mu, theSim);
                    metric_destroy(g);

                    double Sigma = -mdot / (r*U[1]);
                    double Pi = pow(Sigma, GAM) * exp((GAM-1.0)*s);
                    double vr = U[1]/U[0];
                    double vp = U[2]/U[0];
                    double vz = U[3]/U[0];

                    if(bg == GRDISC)
                    {
                        // Perform 2D Newton Raphson in (rho,T) to match
                        // the Sigma and Pi/Sigma found from Mdot and S.
                        double tol = 1.0e-12;
                        int maxIter = 100;
                        double res;

                        double S2 = Pi/Sigma;

                        double rho0 = theCells[k][i][j].prim[RHO];
                        double T0 = theCells[k][i][j].prim[TTT];

                        double prim[5];
                        prim[RHO] = rho0;
                        prim[TTT] = T0;
                        prim[URR] = vr;
                        prim[UPP] = vp;
                        prim[UZZ] = vz;

                        double rho = rho0;
                        double T = T0;
                        int iter = 0;

                        do
                        {
                            prim[RHO] = rho;
                            prim[TTT] = T;
                            double P = eos_ppp(prim, theSim);
                            double e = eos_eps(prim, theSim);
                            double dPdr = eos_dpppdrho(prim, theSim);
                            double dPdt = eos_dpppdttt(prim, theSim);
                            double dedr = eos_depsdrho(prim, theSim);
                            double dedt = eos_depsdttt(prim, theSim);

                            double rhoh = rho + rho*e + P;
                            double H = sqrt(r*r*r*P/(M*rhoh)) / U[0];
                            double sig = rho*H;
                            double s2 = P/rho;

                            double dHdr = 0.5*H*(dPdr/P + 
                                            (1+e+rho*dedr+dPdr)/rhoh);
                            double dHdt = 0.5*H*(dPdt/P+(rho*dedt+dPdt)/rhoh);

                            double dsigdrho = H + rho*dHdr;
                            double dsigdttt = rho*dHdt;
                            double ds2drho = (rho*dPdr - P)/(rho*rho);
                            double ds2dttt = dPdt/rho;

                            double detJ = dsigdrho*ds2dttt - dsigdttt*ds2drho;

                            rho -= (ds2dttt*(sig-Sigma)-dsigdttt*(s2-S2))/detJ;
                            T -= (-ds2drho*(sig-Sigma)+dsigdrho*(s2-S2))/detJ;

                            if(rho <= 0.0)
                            {
                                T = 0.5*prim[RHO]/(prim[RHO]-rho)
                                        * (T-prim[TTT]) + prim[TTT];
                                rho = 0.5*prim[RHO];
                                printf("%d: Small rho\n", iter);
                            }
                            if(T <= 0.0)
                            {
                                rho = 0.5*prim[TTT]/(prim[TTT]-T)
                                        * (rho-prim[RHO]) + prim[RHO];
                                T = 0.5*prim[TTT];
                                printf("%d: Small T\n", iter);
                            }

                            res = sqrt((rho-prim[RHO])*(rho-prim[RHO])
                                        /(rho*rho)
                                    + (T-prim[TTT])*(T-prim[TTT])/(T*T));
                            iter++;
                        }
                        while(res > tol && iter < maxIter);

                        if(iter == maxIter)
                        {
                            printf("ERROR: Plunge BC - NR failed to converge after %d iterations.  Res = %.12lg\n",
                                maxIter, res);
                            printf("Mdot = %.12lg S = %.12lg, gam = %.12lg\n", 
                                    mdot, s, GAM);
                            printf("Sig = %.12lg Pi = %.12lg rho = %.12lg T = %.12lg\n", Sigma, Pi, rho, T);
                        }

                        theCells[k][i][j].prim[RHO] = rho;
                        theCells[k][i][j].prim[TTT] = T;
                        theCells[k][i][j].prim[URR] = vr;
                        theCells[k][i][j].prim[UPP] = vp;
                        theCells[k][i][j].prim[UZZ] = vz;
                    }
                    else
                    {
                        double rho = Sigma;
                        double P = Pi;
                        
                        theCells[k][i][j].prim[RHO] = rho;
                        theCells[k][i][j].prim[PPP] = P;
                        theCells[k][i][j].prim[URR] = vr;
                        theCells[k][i][j].prim[UPP] = vp;
                        theCells[k][i][j].prim[UZZ] = vz;
                    }
                }
            }
        }
    }
}

void cell_boundary_nozzle(struct Cell ***theCells, struct Sim *theSim, 
                struct MPIsetup *theMPIsetup)
{
    // A nozzle entering injecting gas from the boundary.
    
    int i,j,k;
    if(mpisetup_check_rout_bndry(theMPIsetup))
    {
        /*
        double V = sim_BoundPar1(theSim);
        double b = sim_BoundPar2(theSim);
        double rho0 = sim_BoundPar3(theSim);
        double T0 = sim_BoundPar4(theSim);
        */

        double phi0 = sim_BoundPar1(theSim);
        double Mdot = sim_BoundPar2(theSim); // in M_solar/year
        double vr0 = sim_BoundPar3(theSim); // in c
        double T = sim_BoundPar4(theSim);
        double dphi = 0.2; // Nozzle extends from phi0-dphi to phi0+dphi
        double f = M_PI / (2*dphi);

        Mdot *= eos_Msolar/eos_year; // Msolar/year --> g/s
        Mdot /= eos_rho_scale * eos_r_scale*eos_r_scale*eos_c; // g/s --> C.U.
        //sig0 /= eos_rho_scale * eos_r_scale; // g/cm^2 --> C.U.

        for(i = sim_N(theSim,R_DIR) - sim_Nghost_max(theSim,R_DIR); 
                i < sim_N(theSim,R_DIR); i++)
        {
            double rp = sim_FacePos(theSim,i,R_DIR);
            double rm = sim_FacePos(theSim,i-1,R_DIR);
            double r = 0.5*(rp+rm);

            double vr = -vr0;
            double sig0 = -Mdot / (vr*r*dphi);

            for( k=0 ; k<sim_N(theSim,Z_DIR) ; ++k )
            {
                double zp = sim_FacePos(theSim,k,Z_DIR);
                double zm = sim_FacePos(theSim,k-1,Z_DIR);
                double z = 0.5*(zp+zm);

                for( j=0 ; j<sim_N_p(theSim,i) ; ++j )
                {
                    double phip = theCells[k][i][j].tiph - phi0;
                    double phim = theCells[k][i][j].tiph
                                    - theCells[k][i][j].dphi - phi0;
                    while(phip < -M_PI)
                    {
                        phip += 2*M_PI;
                        phim += 2*M_PI;
                    }
                    while(phim > M_PI)
                    {
                        phip -= 2*M_PI;
                        phim -= 2*M_PI;
                    }
                    if(phim < dphi && phip > -dphi)
                    {
                        double xp = f * phip;
                        double xm = f * phim;

                        xp = xp>0.5*M_PI ? 0.5*M_PI : xp;
                        xm = xm<-0.5*M_PI ? -0.5*M_PI : xm;
                        // sig(phi) = sig0 * cos^2(pi/(2*dphi) * (phi-phi0))
                        // sig = <sig(phi)> over the cell
                        double sig = 0.5 * sig0
                                        * (1 + cos(xp+xm)*sin(xp-xm)/(xp-xm));
                        double pi = sig*T;

                        theCells[k][i][j].prim[RHO] = sig;
                        theCells[k][i][j].prim[PPP] = pi;
                        theCells[k][i][j].prim[URR] = vr;
                        theCells[k][i][j].prim[UPP] = 0.0;
                        theCells[k][i][j].prim[UZZ] = 0.0;
                        if(sim_Background(theSim) == GRDISC)
                            theCells[k][i][j].prim[TTT] = T;
                        if(sim_NUM_Q(theSim) >= sim_NUM_C(theSim) + 2)
                            theCells[k][i][j].prim[sim_NUM_C(theSim)+1] = 1.0;
                    }
                }
            }
        }
    }

}
