enum{RHO,PPP,URR,UPP,UZZ,BRR,BPP,BZZ,PSI};
enum{DDD,TAU,SRR,LLL,SZZ};
enum{C_FIXED,C_WCELL,C_WRIEMANN,C_RIGID,C_KEPLER,C_OMEGA20,C_MILOS};
enum{A_NONE,A_OMEGA20,A_KEPLER,A_WEIRD};
enum{LEFT,LEFTSTAR,RIGHTSTAR,RIGHT};
enum{VAR_INT,VAR_DOUB,VAR_STR};
enum{BOUND_FIXED,BOUND_OUTFLOW,BOUND_PERIODIC,BOUND_SS};
enum{VORTEX,TORUS,BONDI,UNIFORM,SHOCK1,SHOCK2,SHOCK3,SHOCK4,ISENTROPE,GBONDI,GBONDI2,EQUIL1,EQUIL2,SSDISC,NTDISC,CNSTDISC,CARTSHEAR,DISCTEST};
enum{NONE,SINGLE,BINARY};
//unify the 2 below at some point
enum{R_DIR,Z_DIR,P_DIR}; //P_DIR has to be last as is.
enum{RDIRECTION,PDIRECTION,ZDIRECTION};
enum{HLLC,HLL};
//Background Types
enum{NEWTON,GR,GRVISC1};
//Metric Types
enum{SR,SCHWARZSCHILD_SC,SCHWARZSCHILD_KS,SR_CART};
//Frames
enum{FR_EULER,FR_KEP};

#include "mpi.h"
MPI_Comm sim_comm;
#define farris_mpi_factorization 0
#define KEP_BNDRY 0
//#define NPCAP 256 
double time_global;

#define PRINTTOOMUCH 0
