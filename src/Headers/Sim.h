#ifndef SIM_H
#define SIM_H
struct Sim;
struct Cell;
struct MPIsetup;

#ifdef SIM_PRIVATE_DEFS
struct Sim {
  double *r_faces;
  double *z_faces;
  int *N_p;
  int NPCAP;
  int N0[2];
  int N_noghost[2];
  int Nghost_min[2];
  int Nghost_max[2];
  int Ncells;
  int Ncells_global;
  int offset;
  //fixed parameters
  int InitialDataType;
  int GravMassType;
  int BoundTypeR;
  int BoundTypeZ;
  int NoInnerBC;
  int Restart;
  int N_global[2];
  int NP_CONST;
  double aspect;
  int ng;
  double T_MAX;
  int NUM_CHECKPOINTS;
  int NUM_DIAG_DUMP;
  int NUM_DIAG_MEASURE;
  double MIN[2];
  double MAX[2];
  int NUM_C;
  int NUM_N;
  int NUM_Q;
  int W_NUMERIC_TYPE;   
  int NumGravMass;
  double MassRatio;
  double OrbShrinkTscale;
  double OrbShrinkT0;
  int Riemann;
  double GAMMALAW;
  int COOLING;
  double EXPLICIT_VISCOSITY;
  int VISC_CONST;
  double PoRho_r1;
  double CFL;
  double PLM;
  int GRAV2D;
  double G_EPS;
  int RhoSinkOn;
  int PHI_ORDER;
  double RHO_FLOOR;
  double CS_FLOOR;
  double CS_CAP;
  double VEL_CAP;
  double DAMP_TIME;
  double RDAMP_INNER;
  double RDAMP_OUTER;
  double RLogScale;
  double ZLogScale;
  double HiResSigma;
  double HiResR0a;
  double HiResR0b;
  double HiResFac;
  int W_ANALYTIC_TYPE;
};
#endif

//create and destroy
struct Sim *sim_create(struct MPIsetup * );
void sim_alloc_arr(struct Sim * , struct MPIsetup * );
void sim_destroy(struct Sim *); 
//access Sim data
double sim_MIN(struct Sim * ,int);
double sim_MAX(struct Sim * ,int);
int sim_N0(struct Sim * ,int );
int sim_N_p(struct Sim *,int);
double sim_FacePos(struct Sim *,int,int);
int sim_N(struct Sim *,int );
int sim_NoInnerBC(struct Sim *);
int sim_Restart(struct Sim *);
int sim_BoundTypeR(struct Sim *);
int sim_BoundTypeZ(struct Sim *);
int sim_ZeroPsiBndry(struct Sim *);
int sim_N_global(struct Sim *,int);
int sim_Ncells(struct Sim *);
int sim_Ncells_global(struct Sim *);
int sim_offset(struct Sim *);
int sim_ng(struct Sim *);
int sim_Nghost_min(struct Sim *,int);
int sim_Nghost_max(struct Sim *,int);
int sim_W_NUMERIC_TYPE(struct Sim *);
int sim_NumGravMass(struct Sim *);
double sim_MassRatio(struct Sim *);
double sim_OrbShrinkTscale(struct Sim *);
double sim_OrbShrinkT0(struct Sim *);
int sim_Riemann(struct Sim *);
double sim_GAMMALAW(struct Sim *);
int sim_COOLING(struct Sim *);
double sim_EXPLICIT_VISCOSITY(struct Sim *);
int sim_VISC_CONST(struct Sim *);
double sim_PoRho_r1(struct Sim *);
double sim_CFL(struct Sim *);
double sim_PLM(struct Sim *);
int sim_W_ANALYTIC_TYPE(struct Sim *);
int sim_GRAV2D(struct Sim *);
double sim_G_EPS(struct Sim *);
int sim_RhoSinkOn(struct Sim *);
int sim_PHI_ORDER(struct Sim *);
double sim_RHO_FLOOR(struct Sim *);
double sim_CS_FLOOR(struct Sim *);
double sim_CS_CAP(struct Sim *);
double sim_VEL_CAP(struct Sim *);
int sim_NUM_C(struct Sim *);
int sim_NUM_Q(struct Sim *);
double sim_get_T_MAX(struct Sim * );
int sim_NUM_CHECKPOINTS(struct Sim * );
int sim_NUM_DIAG_DUMP(struct Sim * );
int sim_NUM_DIAG_MEASURE(struct Sim * );
int sim_InitialDataType(struct Sim * );
double sim_DAMP_TIME(struct Sim *);
double sim_RDAMP_INNER(struct Sim *);
double sim_RDAMP_OUTER(struct Sim *);

int sim_GravMassType(struct Sim * );
//set Grid data
int sim_read_par_file(struct Sim * ,struct MPIsetup *, char * );
void sim_set_N_p(struct Sim *);
void sim_set_rz(struct Sim *,struct MPIsetup *);
void sim_set_misc(struct Sim *,struct MPIsetup *);
// W_A stuff
double sim_rOm_a(struct Sim * ,double,double );
double sim_rdrOm_a(struct Sim * ,double,double );
double sim_dtOm_a(struct Sim * ,double,double );
#endif
