enum{RHO,PPP,URR,UPP,UZZ};
enum{DDD,TAU,SRR,LLL,SZZ};
enum{C_FIXED,C_WCELL,C_RIGID,C_KEPLER,C_OMEGA20,C_MILOS,C_POWERLAW};
enum{A_FIXED,A_KEPLER,A_OMEGA20,A_MILOS,A_POWERLAW};
enum{LEFT,LEFTSTAR,RIGHTSTAR,RIGHT};
enum{VAR_INT,VAR_DOUB,VAR_STR};
enum{BOUND_FIXED,BOUND_OUTFLOW,BOUND_PERIODIC};
enum{SHEAR,VORTEX,TORUS,MILOS_MACFADYEN,RAD_DOM,MIDDLE,SSTEST};
enum{NONE,SINGLE,BINARY};
//unify the 2 below at some point
enum{R_DIR,Z_DIR};
enum{RDIRECTION,PDIRECTION,ZDIRECTION};
enum{HLLC,HLL};
#include "mpi.h"
MPI_Comm sim_comm;
#define farris_mpi_factorization 0
#define NO_W_IN_CFL 0
#define KEP_BNDRY 0
#define BNORM_AVG 0
//#define CHECKPOINTING 
#define TVISC_FAC 1.0
#define PHIMAX 2.0*M_PI
#define diode 0
#define EXTRAP_BC 0
#define SS_BCS 0
#define w_a_milos_index 8
