enum{RHO,PPP,URR,UPP,UZZ,BRR,BPP,BZZ,PSI};
enum{DDD,TAU,SRR,LLL,SZZ};
enum{C_FIXED,C_WCELL,C_WRIEMANN,C_RIGID,C_KEPLER,C_OMEGA10};
enum{LEFT,LEFTSTAR,RIGHTSTAR,RIGHT};
enum{EULER,MHD};
enum{VAR_INT,VAR_DOUB,VAR_STR};
enum{BOUND_FIXED,BOUND_OUTFLOW,BOUND_PERIODIC};
enum{FLOCK,SHEAR,VORTEX,STONE,FIELDLOOP,PSIGRAD};
enum{NONE,SINGLE,BINARY};
//unify the 2 below at some point
enum{R_DIR,Z_DIR};
enum{RDIRECTION,PDIRECTION,ZDIRECTION};
enum{HLLC,HLL};
#include "mpi.h"
MPI_Comm sim_comm;


