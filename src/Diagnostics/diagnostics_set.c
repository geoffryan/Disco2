#define DIAGNOSTICS_PRIVATE_DEFS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../Headers/Sim.h"
#include "../Headers/Cell.h"
#include "../Headers/GravMass.h"
#include "../Headers/Diagnostics.h"
#include "../Headers/TimeStep.h"
#include "../Headers/MPIsetup.h"
#include "../Headers/header.h"

void diagnostics_set(struct Diagnostics * theDiagnostics,struct Cell *** theCells,struct Sim * theSim,struct TimeStep * theTimeStep,struct MPIsetup * theMPIsetup,struct GravMass * theGravMasses){
  if (timestep_get_t(theTimeStep)>diagnostics_tdiag_measure(theDiagnostics)){
    int num_r_points = sim_N(theSim,R_DIR)-sim_Nghost_min(theSim,R_DIR)-sim_Nghost_max(theSim,R_DIR);
    int num_r_points_global = sim_N_global(theSim,R_DIR);

    int NUM_SCAL = theDiagnostics->NUM_DIAG;
    int NUM_VEC = theDiagnostics->NUM_DIAG+1;
    int NUM_EQ = theDiagnostics->NUM_DIAG+2;

    int i,j,k,n;

    double * EquatDiag_temp = malloc(sizeof(double) * theDiagnostics->N_eq_cells*NUM_EQ);
    double * EquatDiag_reduce = malloc(sizeof(double) * theDiagnostics->N_eq_cells*NUM_EQ);
    double * VectorDiag_temp = malloc(sizeof(double) * num_r_points_global*NUM_VEC);
    double * VectorDiag_reduce = malloc(sizeof(double) * num_r_points_global*NUM_VEC);
    double * ScalarDiag_temp = malloc(sizeof(double)*NUM_SCAL);
    double * ScalarDiag_reduce = malloc(sizeof(double)*NUM_SCAL);

    double mass_near_bh0_r0p1_temp = 0.0;
    double mass_near_bh1_r0p1_temp = 0.0;
    double mass_near_bh0_r0p1_reduce = 0.0;
    double mass_near_bh1_r0p1_reduce = 0.0;

    double mass_near_bh0_r0p2_temp = 0.0;
    double mass_near_bh1_r0p2_temp = 0.0;
    double mass_near_bh0_r0p2_reduce = 0.0;
    double mass_near_bh1_r0p2_reduce = 0.0;

    double mass_near_bh0_r0p4_temp = 0.0;
    double mass_near_bh1_r0p4_temp = 0.0;
    double mass_near_bh0_r0p4_reduce = 0.0;
    double mass_near_bh1_r0p4_reduce = 0.0;


    double dtout = timestep_get_t(theTimeStep)-theDiagnostics->toutprev;

    int imin = sim_Nghost_min(theSim,R_DIR);
    int imax = sim_N(theSim,R_DIR)-sim_Nghost_max(theSim,R_DIR);
    int kmin = sim_Nghost_min(theSim,Z_DIR);
    int kmax = sim_N(theSim,Z_DIR)-sim_Nghost_max(theSim,Z_DIR);

    for (n=0;n<NUM_SCAL;++n){
      ScalarDiag_temp[n]=0.0;
      ScalarDiag_reduce[n]=0.0;
    }
    for (i=0;i<num_r_points_global;++i){
      for (n=0;n<NUM_VEC;++n){
        VectorDiag_temp[i*NUM_VEC+n]=0.0;
      }
    }

    int position=0;
    for (i=0;i<num_r_points_global;++i){
      for(j = 0; j < theDiagnostics->N_p_global[i]; j++){
        for (n=0;n<NUM_EQ;++n){
          EquatDiag_temp[position]=0.0;
          ++position;
        }
      }
    }

    double mass_inside_1_temp=0.0;
    position=0;
    for (k=kmin;k<kmax;++k){
      double zp = sim_FacePos(theSim,k,Z_DIR);
      double zm = sim_FacePos(theSim,k-1,Z_DIR);
      double z = 0.5*(zm+zp);
      double dz = zp-zm;
      for (i=imin;i<imax;++i){
        double rp = sim_FacePos(theSim,i,R_DIR);
        double rm = sim_FacePos(theSim,i-1,R_DIR);
        double r = 0.5*(rm+rp);
        double r_bh0 = gravMass_r(theGravMasses,0);
        double phi_bh0 = gravMass_phi(theGravMasses,0);
        double r_bh1 = gravMass_r(theGravMasses,1);
        double phi_bh1 = gravMass_phi(theGravMasses,1);
        double a = r_bh0 + r_bh1;        
        for (j=0;j<sim_N_p(theSim,i);++j){
          double dphi = cell_dphi(cell_single(theCells,i,j,k));
          double phi = cell_tiph(cell_single(theCells,i,j,k)) - 0.5*dphi;
          double rho = cell_prim(cell_single(theCells,i,j,k),RHO);
          double press = cell_prim(cell_single(theCells,i,j,k),PPP);
          double vr = cell_prim(cell_single(theCells,i,j,k),URR);
          double vp_minus_w = cell_prim(cell_single(theCells,i,j,k),UPP)*r;
          double vp = cell_prim(cell_single(theCells,i,j,k),UPP)*r+sim_rOm_a(theSim,r,a);
          double vz = cell_prim(cell_single(theCells,i,j,k),UZZ);
          double v2   = vr*vr + vp*vp + vz*vz;
          double rhoe = press/(sim_GAMMALAW(theSim)-1.);
          double KE = 0.5*rho*v2;
          double E_hydro = rhoe + KE;
          double Cool = cell_Cool(theCells,i,j,k);
          double dV = 0.5*(rp*rp-rm*rm)*dphi;

          if (rp<=1.0){
            mass_inside_1_temp += rho*dV;
          }
          double passive_scalar = cell_prim(cell_single(theCells,i,j,k),5);
          double Omega = 1.;
          double t = timestep_get_t(theTimeStep);
          double dPhi_dphi = 1/4.*r*sin(phi-Omega*t) * (
              pow(r*r+.25-r*cos(phi-Omega*t),-1.5) -  
              pow(r*r+.25+r*cos(phi-Omega*t),-1.5));


          double dist_bh0 = gravMass_dist(theGravMasses,0,r,phi,0.);
          double dist_bh1 = gravMass_dist(theGravMasses,1,r,phi,0.);

          if (dist_bh0<0.1){
              double dM = rho*dV;
              mass_near_bh0_r0p1_temp +=dM;
          }
          if (dist_bh1<0.1){
              double dM = rho*dV;
              mass_near_bh1_r0p1_temp +=dM;
          }

          if (dist_bh0<0.2){
              double dM = rho*dV;
              mass_near_bh0_r0p2_temp +=dM;
          }
          if (dist_bh1<0.2){
              double dM = rho*dV;
              mass_near_bh1_r0p2_temp +=dM;
          }

          if (dist_bh0<0.4){
              double dM = rho*dV;
              mass_near_bh0_r0p4_temp +=dM;
          }
          if (dist_bh1<0.4){
              double dM = rho*dV;
              mass_near_bh1_r0p4_temp +=dM;
          }



          if ((fabs(zp)<0.0000001)||(fabs(z)<0.0000001)){
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+0] = r;
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+1] = phi;
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+2] = rho;
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+3] = press;
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+4] = vr;
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+5] = vp;
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+6] = vz;            
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+7] = vp_minus_w;            
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+8] = Cool;
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+9] = passive_scalar;            
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+10] = E_hydro;            
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+11] = KE;            
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+12] = rhoe;
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+13] = 0.0;
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+14] = 0.0;
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+15] = 0.0;
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+16] = 0.0;
              EquatDiag_temp[(theDiagnostics->offset_eq+position)*NUM_EQ+17] = 0.0;
              ++position;
          }
          // divide by number of phi cells to get phi average, mult by dz because we are doing a z integration;
          VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+0] += (r/sim_N_p(theSim,i)*dz) ;
          VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+1] += (rho/sim_N_p(theSim,i)*dz) ;
          VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+2] += (press/sim_N_p(theSim,i)*dz) ;
          VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+3] += (vr/sim_N_p(theSim,i)*dz) ;
          VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+4] += (vp/sim_N_p(theSim,i)*dz) ;
          VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+5] += (vz/sim_N_p(theSim,i)*dz) ;
          VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+6] += (vp_minus_w/sim_N_p(theSim,i)*dz) ;
          VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+7] += (Cool/sim_N_p(theSim,i)*dz) ;
          VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+8] += (passive_scalar/sim_N_p(theSim,i)*dz) ;
          VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+9] += (E_hydro/sim_N_p(theSim,i)*dz) ;
          VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+10] += (KE/sim_N_p(theSim,i)*dz) ;
          VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+11] += (rhoe/sim_N_p(theSim,i)*dz) ;
          VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+12] += 0.0 ;
          VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+13] += 0.0 ;
          VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+14] += 0.0 ;
          VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+15] += 0.0 ;
          VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+16] += 0.0 ; 
          // the above are just placeholders. Put the real diagnostics you want here, then adjust NUM_DIAG accordingly.
        }
      }
    }


    for (i=imin;i<imax;++i){
        double rp = sim_FacePos(theSim,i,R_DIR);
        double rm = sim_FacePos(theSim,i-1,R_DIR);
        for (n=0;n<NUM_SCAL;++n){
            // mult by delta r^2 because we are doing an r integration
            ScalarDiag_temp[n] += VectorDiag_temp[(sim_N0(theSim,R_DIR)+i-imin)*NUM_VEC+n+1] * (rp*rp-rm*rm); 
        }
    }

    double mass_inner_wedge_temp = 0.0;
    if (mpisetup_MyProc(theMPIsetup)==0){
        //printf("myproc: %d, rm(0): %e, rp(0): %e\n",mpisetup_MyProc(theMPIsetup),sim_FacePos(theSim,-1,R_DIR),sim_FacePos(theSim,0,R_DIR));
        double rp = sim_FacePos(theSim,0,R_DIR);
        double rm = sim_FacePos(theSim,-1,R_DIR);
        for (j=0;j<sim_N_p(theSim,i);++j){
            double dphi = cell_dphi(cell_single(theCells,0,j,0));
            double dV = 0.5*(rp*rp-rm*rm)*dphi;
            double rho = cell_prim(cell_single(theCells,0,j,0),RHO);
            mass_inner_wedge_temp += rho*dV;
        }
    }
    MPI_Allreduce( ScalarDiag_temp,ScalarDiag_reduce , NUM_SCAL, MPI_DOUBLE, MPI_SUM, sim_comm);
    MPI_Allreduce( VectorDiag_temp,VectorDiag_reduce , num_r_points_global*NUM_VEC, MPI_DOUBLE, MPI_SUM, sim_comm);
    MPI_Allreduce( EquatDiag_temp,EquatDiag_reduce ,theDiagnostics->N_eq_cells*NUM_EQ, MPI_DOUBLE, MPI_SUM, sim_comm);

    MPI_Allreduce( &mass_near_bh0_r0p1_temp,&mass_near_bh0_r0p1_reduce , 1, MPI_DOUBLE, MPI_SUM, sim_comm);
    MPI_Allreduce( &mass_near_bh1_r0p1_temp,&mass_near_bh1_r0p1_reduce , 1, MPI_DOUBLE, MPI_SUM, sim_comm);

    MPI_Allreduce( &mass_near_bh0_r0p2_temp,&mass_near_bh0_r0p2_reduce , 1, MPI_DOUBLE, MPI_SUM, sim_comm);
    MPI_Allreduce( &mass_near_bh1_r0p2_temp,&mass_near_bh1_r0p2_reduce , 1, MPI_DOUBLE, MPI_SUM, sim_comm);

    MPI_Allreduce( &mass_near_bh0_r0p4_temp,&mass_near_bh0_r0p4_reduce , 1, MPI_DOUBLE, MPI_SUM, sim_comm);
    MPI_Allreduce( &mass_near_bh1_r0p4_temp,&mass_near_bh1_r0p4_reduce , 1, MPI_DOUBLE, MPI_SUM, sim_comm);

    double mass_inside_1_reduce;
    MPI_Allreduce( &mass_inside_1_temp,&mass_inside_1_reduce , 1, MPI_DOUBLE, MPI_SUM, sim_comm);

    double mass_inner_wedge_reduce;
    MPI_Allreduce( &mass_inner_wedge_temp,&mass_inner_wedge_reduce , 1, MPI_DOUBLE, MPI_SUM, sim_comm);

    double RMIN = sim_MIN(theSim,R_DIR);
    double RMAX = sim_MAX(theSim,R_DIR);
    double ZMIN = sim_MIN(theSim,Z_DIR);
    double ZMAX = sim_MAX(theSim,Z_DIR);

    for (n=0;n<NUM_SCAL;++n){
        ScalarDiag_reduce[n] *= dtout/((ZMAX-ZMIN)*(RMAX*RMAX-RMIN*RMIN));
    }

    int req1_found = 0;
    double Mdot_near_req1,r_near_req1;
    for (i=0;i<num_r_points_global;++i){
        double r = VectorDiag_reduce[i*NUM_VEC]/(ZMAX-ZMIN);
        if (r>1.0 && req1_found==0){
            Mdot_near_req1 = VectorDiag_reduce[i*NUM_VEC+5]*2.*M_PI*r/(ZMAX-ZMIN);
            r_near_req1 = r;
            req1_found = 1;
        }
        for (n=0;n<NUM_VEC;++n){
            VectorDiag_reduce[i*NUM_VEC+n] *= dtout/(ZMAX-ZMIN);
        }
    }

    if(mpisetup_MyProc(theMPIsetup)==0){
        char DiagMdotFilename[256];
        sprintf(DiagMdotFilename,"DiagMdot.dat");
        FILE * DiagMdotFile = fopen(DiagMdotFilename,"a");
        fprintf(DiagMdotFile,"%e %e %e %e %e %e %e %e %e %e %e %e %e\n",timestep_get_t(theTimeStep), Mdot_near_req1,r_near_req1,mass_inside_1_reduce,mass_inner_wedge_reduce,gravMass_Mdot(theGravMasses,0),gravMass_Mdot(theGravMasses,1),gravMass_Macc(theGravMasses,0),gravMass_Macc(theGravMasses,1),mass_near_bh0_r0p2_reduce,mass_near_bh1_r0p2_reduce,mass_near_bh0_r0p4_reduce,mass_near_bh1_r0p4_reduce);       
        fclose(DiagMdotFile);
    }

    //We are doing time averaged diagnostics, so mult by delta t and add it
    //We will divide by the total delta next time we save to disk;
    for (i=0;i<num_r_points_global;++i){
        for (n=0;n<NUM_VEC;++n){
            theDiagnostics->VectorDiag[i][n] += VectorDiag_reduce[i*NUM_VEC+n] ;
        }
    }
    for (n=0;n<NUM_SCAL;++n){
        theDiagnostics->ScalarDiag[n] += ScalarDiag_reduce[n] ;
    }

    position=0;
    for (i=0;i<num_r_points_global;++i){
        for(j = 0; j < theDiagnostics->N_p_global[i]; j++){
            for (n=0;n<NUM_EQ;++n){
                theDiagnostics->EquatDiag[position][n] = EquatDiag_reduce[position*NUM_EQ+n];
            }
            ++position;
        }
    }

    //update output time;
    theDiagnostics->toutprev = timestep_get_t(theTimeStep);

    free(ScalarDiag_temp);
    free(ScalarDiag_reduce);
    free(VectorDiag_temp);
    free(VectorDiag_reduce);
    free(EquatDiag_temp);
    free(EquatDiag_reduce);

    theDiagnostics->tdiag_measure += theDiagnostics->dtdiag_measure;
  } 
}






