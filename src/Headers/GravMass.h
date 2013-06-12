#ifndef PLANET_H
#define PLANET_H
struct GravMass;
struct Sim;

#ifdef PLANET_PRIVATE_DEFS
struct GravMass{
   double r;
   double phi;
   double M;
   double omega;
   double RK_r;
   double RK_phi;
   double RK_M;
   double RK_omega;
   double Mdot;
};
#endif

//create and destroy
struct GravMass *gravMass_create(int);
void gravMass_destroy(struct GravMass *);
//initialization
void gravMass_init_none(struct GravMass *,struct Sim *);
void gravMass_init_single(struct GravMass *,struct Sim *);
void gravMass_init_binary(struct GravMass *,struct Sim *);
void (*gravMass_init_ptr(struct Sim * ))(struct GravMass *,struct Sim *);
void gravMass_set_chkpt(struct GravMass * ,int ,double ,double ,double, double );
//access data
double gravMass_r(struct GravMass * ,int);
double gravMass_phi(struct GravMass * ,int);
double gravMass_M(struct GravMass * ,int);
double gravMass_omega(struct GravMass * ,int );
double gravMass_Mdot(struct GravMass *,int);
//miscellaneous
void gravMass_clean_pi(struct GravMass *,struct Sim *);
void gravMass_copy(struct GravMass *,struct Sim *);
void gravMass_move(struct GravMass *,double);
void gravMass_update_RK( struct GravMass * ,struct Sim * , double );
void gravMass_reset_Mdot( struct GravMass * , struct Sim * );
void gravMass_add_Mdot( struct GravMass * , double, int );
#endif 
