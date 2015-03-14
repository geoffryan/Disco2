enum{RHO,PPP,URR,UPP,UZZ,BRR,BPP,BZZ,PSI};
#define TTT 1
enum{DDD,TAU,SRR,LLL,SZZ};
enum{C_FIXED,C_WCELL,C_WRIEMANN,C_RIGID,C_KEPLER,C_OMEGA20,C_MILOS};
enum{A_NONE,A_OMEGA20,A_KEPLER,A_WEIRD};
enum{LEFT,LEFTSTAR,RIGHTSTAR,RIGHT};
enum{VAR_INT,VAR_DOUB,VAR_STR};
enum{BOUND_FIXED,BOUND_OUTFLOW,BOUND_PERIODIC,BOUND_SS};
enum{VORTEX,TORUS,BONDI,UNIFORM,SHOCK1,SHOCK2,SHOCK3,SHOCK4,ISENTROPE,GBONDI,
  GBONDI2,EQUIL1,EQUIL2,SSDISC,NTDISC,CNSTDISC,CARTSHEAR,DISCTEST,ACCDISC,ADAF,
  ADAF2,ATMO,ORBIT};
enum{NONE,SINGLE,BINARY};
//unify the 2 below at some point
enum{R_DIR,Z_DIR,P_DIR}; //P_DIR has to be last as is.
enum{RDIRECTION,PDIRECTION,ZDIRECTION};
enum{HLLC,HLL};
//Background Types
enum{NEWTON,GR,GRVISC1,GRDISC};
//Metric Types
enum{SR,SCHWARZSCHILD_SC,SCHWARZSCHILD_KS,SR_CART,KERR_KS};
//Frames
enum{FR_EULER,FR_KEP,FR_ACC};
//Equations of State
enum{EOS_GAMMALAW, EOS_GASRAD, EOS_GASRADDEG, EOS_PWF};
//Cooling Schemes
enum{COOL_NONE,COOL_ISOTHERM,COOL_BB_ES,COOL_BB_FF,COOL_NU_APRX,COOL_NU_ITOH};
//Boosting Effects
enum{BOOST_NONE, BOOST_BIN};

#include "mpi.h"
MPI_Comm sim_comm;
#define farris_mpi_factorization 0
#define KEP_BNDRY 0
//#define NPCAP 256 
double time_global;

#define PRINTTOOMUCH 0

#define RHO_REF 1.0e10
#define C_REF 2.99792458e10
#define SOLAR_RG 1.4766250385e5
#define KB 1.3806488e-16
#define MASS_PROTON 1.672621777e-24 
