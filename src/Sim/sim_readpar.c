#define SIM_PRIVATE_DEFS
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../Headers/Sim.h"
#include "../Headers/MPIsetup.h"
#include "../Headers/header.h"

int readvar( char * filename , char * varname , int vartype , void * ptr ){
  FILE * inFile = fopen( filename , "r" );
  char s[512];
  char nm[512];
  char s1[512];
  int found = 0;
  while( (fgets(s,512,inFile) != NULL) && found==0 ){
    sscanf(s,"%s ",nm);
    if( strcmp(nm,varname)==0 ){
      strcpy(s1,s);
      found=1;
    }
  }

  fclose( inFile );
  if( found==0 ) {
    printf("cant find %s\n",varname);
    return(1);
  }
  char * s2 = s1+strlen(nm)+strspn(s1+strlen(nm),"\t :=>_");

  double temp;
  char stringval[256];

  sscanf(s2,"%lf",&temp);
  sscanf(s2,"%256s",stringval);

  if( vartype == VAR_INT ){
    *((int *)   ptr) = (int)temp;
  }else if( vartype == VAR_DOUB ){
    *((double *)ptr) = (double)temp;
  }else{
    strcpy( ptr , stringval );
  }

  return(0);
}

int sim_read_par_file(struct Sim * theSim, struct MPIsetup * theMPIsetup, char * inputfilename){

  char * pfile = inputfilename;
  int err=0;  
  int nrank;
  for( nrank=0 ; nrank<mpisetup_NumProcs(theMPIsetup) ; ++nrank ){
    if( mpisetup_MyProc(theMPIsetup)==nrank ){
      err += readvar( pfile , "Restart"              , VAR_INT  , &(theSim->Restart)  );
      err += readvar( pfile , "InitialDataType"      , VAR_INT  , &(theSim->InitialDataType)  );
      err += readvar( pfile , "GravMassType"      , VAR_INT  , &(theSim->GravMassType)  );
      err += readvar( pfile , "BoundTypeRIn"         , VAR_INT  , &(theSim->BoundTypeRIn)  );
      err += readvar( pfile , "BoundTypeROut"         , VAR_INT  , &(theSim->BoundTypeROut)  );
      err += readvar( pfile , "BoundTypeZBot"         , VAR_INT  , &(theSim->BoundTypeZBot)  );
      err += readvar( pfile , "BoundTypeZTop"         , VAR_INT  , &(theSim->BoundTypeZTop)  );
      err += readvar( pfile , "NoInnerBC"         , VAR_INT  , &(theSim->NoInnerBC)  );
      err += readvar( pfile , "Background"         , VAR_INT  , &(theSim->Background)  );
      err += readvar( pfile , "Metric"         , VAR_INT  , &(theSim->Metric)  );
      err += readvar( pfile , "Frame"         , VAR_INT  , &(theSim->Frame)  );
      err += readvar( pfile , "NumR"              , VAR_INT  , &(theSim->N_global[R_DIR]) );
      err += readvar( pfile , "NumZ"              , VAR_INT  , &(theSim->N_global[Z_DIR]) );
      err += readvar( pfile , "ng"              , VAR_INT  , &(theSim->ng));
      err += readvar( pfile , "R_Min"             , VAR_DOUB , &(theSim->MIN[R_DIR])  );
      err += readvar( pfile , "R_Max"             , VAR_DOUB , &(theSim->MAX[R_DIR])  );
      err += readvar( pfile , "Z_Min"             , VAR_DOUB , &(theSim->MIN[Z_DIR])  );
      err += readvar( pfile , "Z_Max"             , VAR_DOUB , &(theSim->MAX[Z_DIR])  );
      err += readvar( pfile , "NP_CONST"             , VAR_INT , &(theSim->NP_CONST)  );
      err += readvar( pfile , "aspect"             , VAR_DOUB , &(theSim->aspect)  );
      err += readvar( pfile , "NUM_C"              , VAR_INT  , &(theSim->NUM_C) );
      err += readvar( pfile , "NUM_N"              , VAR_INT  , &(theSim->NUM_N) );
      err += readvar( pfile , "Time_Max"       , VAR_DOUB , &(theSim->T_MAX)  );
      err += readvar( pfile , "Num_Checkpoints"   , VAR_INT  , &(theSim->NUM_CHECKPOINTS)  );
      err += readvar( pfile , "Num_Diag_Dump"   , VAR_INT  , &(theSim->NUM_DIAG_DUMP)  );
      err += readvar( pfile , "Num_Diag_Measure"   , VAR_INT  , &(theSim->NUM_DIAG_MEASURE)  );
      err += readvar( pfile , "Move_Cells"        , VAR_INT  , &(theSim->MOVE_CELLS)  );
      err += readvar( pfile , "RiemannSolver"   , VAR_INT , &(theSim->Riemann)  );
      err += readvar( pfile , "Adiabatic_Index"   , VAR_DOUB , &(theSim->GAMMALAW)  );
      err += readvar( pfile , "CFL"               , VAR_DOUB , &(theSim->CFL)  );
      err += readvar( pfile , "PLM"               , VAR_DOUB , &(theSim->PLM)  );
      err += readvar( pfile , "Grav_2D"            , VAR_INT  , &(theSim->GRAV2D)  );
      err += readvar( pfile , "GravM"          , VAR_DOUB  , &(theSim->GravM)  );
      err += readvar( pfile , "GravA"          , VAR_DOUB  , &(theSim->GravA)  );
      err += readvar( pfile , "G_EPS"             , VAR_DOUB , &(theSim->G_EPS)  );
      err += readvar( pfile , "PHI_ORDER"             , VAR_DOUB , &(theSim->PHI_ORDER)  );
      err += readvar( pfile , "Rho_Floor"         , VAR_DOUB , &(theSim->RHO_FLOOR)  );
      err += readvar( pfile , "Cs_Floor"          , VAR_DOUB , &(theSim->CS_FLOOR)  );
      err += readvar( pfile , "Cs_Cap"            , VAR_DOUB , &(theSim->CS_CAP)  );
      err += readvar( pfile , "Vel_Cap"           , VAR_DOUB , &(theSim->VEL_CAP)  );
      err += readvar( pfile , "DAMP_TIME"           , VAR_DOUB , &(theSim->DAMP_TIME)  );
      err += readvar( pfile , "RDAMP_INNER"           , VAR_DOUB , &(theSim->RDAMP_INNER)  );
      err += readvar( pfile , "RDAMP_OUTER"           , VAR_DOUB , &(theSim->RDAMP_OUTER)  );
      err += readvar( pfile , "RLogScale"           , VAR_DOUB , &(theSim->RLogScale)  );
      err += readvar( pfile , "ZLogScale"           , VAR_DOUB , &(theSim->ZLogScale)  );
      err += readvar( pfile , "HiResSigma"           , VAR_DOUB , &(theSim->HiResSigma)  );
      err += readvar( pfile , "HiResR0"           , VAR_DOUB , &(theSim->HiResR0)  );
      err += readvar( pfile , "HiResFac"           , VAR_DOUB , &(theSim->HiResFac)  );
      err += readvar( pfile , "InitPar0"           , VAR_INT , &(theSim->InitPar0)  );
      err += readvar( pfile , "InitPar1"           , VAR_DOUB , &(theSim->InitPar1)  );
      err += readvar( pfile , "InitPar2"           , VAR_DOUB , &(theSim->InitPar2)  );
      err += readvar( pfile , "InitPar3"           , VAR_DOUB , &(theSim->InitPar3)  );
      err += readvar( pfile , "InitPar4"           , VAR_DOUB , &(theSim->InitPar4)  );
      err += readvar( pfile , "InitPar5"           , VAR_DOUB , &(theSim->InitPar5)  );
      err += readvar( pfile , "InitPar6"           , VAR_DOUB , &(theSim->InitPar6)  );
      err += readvar( pfile , "AlphaVisc"           , VAR_DOUB , &(theSim->AlphaVisc)  );
      err += readvar( pfile , "EOSType"           , VAR_INT , &(theSim->EOSType)  );
      err += readvar( pfile , "EOSPar1"           , VAR_DOUB , &(theSim->EOSPar1)  );
      err += readvar( pfile , "EOSPar2"           , VAR_DOUB , &(theSim->EOSPar2)  );
      err += readvar( pfile , "EOSPar3"           , VAR_DOUB , &(theSim->EOSPar3)  );
      err += readvar( pfile , "EOSPar4"           , VAR_DOUB , &(theSim->EOSPar4)  );
      err += readvar( pfile , "CoolingType"           , VAR_INT , &(theSim->CoolingType)  );
      err += readvar( pfile , "CoolPar1"           , VAR_DOUB , &(theSim->CoolPar1)  );
      err += readvar( pfile , "CoolPar2"           , VAR_DOUB , &(theSim->CoolPar2)  );
      err += readvar( pfile , "CoolPar3"           , VAR_DOUB , &(theSim->CoolPar3)  );
      err += readvar( pfile , "CoolPar4"           , VAR_DOUB , &(theSim->CoolPar4)  );
      err += readvar( pfile , "BoostType"       , VAR_INT , &(theSim->BoostType)  );
      err += readvar( pfile , "BinA"           , VAR_DOUB , &(theSim->BinA)  );
      err += readvar( pfile , "BinW"           , VAR_DOUB , &(theSim->BinW)  );
      err += readvar( pfile , "BinM"           , VAR_DOUB , &(theSim->BinM)  );
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  int errtot;
  MPI_Allreduce( &err , &errtot , 1 , MPI_INT , MPI_SUM , MPI_COMM_WORLD );

  if( errtot > 0 ){
    printf("Read Failed\n");
    printf("there were %d errors\n",errtot);
    return(1);
  }

  return(0);

}



