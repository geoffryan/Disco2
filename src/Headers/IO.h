#ifndef IO_H
#define IO_H
struct IO;
struct Cell;
struct TimeStep;
struct Sim;
#ifdef IO_PRIVATE_DEFS
struct IO{
  double **primitives;
};
#endif
//create and destroy
struct IO *io_create(struct Sim *);
void io_destroy(struct IO *);
//move data between theCells and IO buffer
void io_setbuf(struct IO *,struct Cell ***,struct Sim *);
void io_readbuf(struct IO *,struct Cell ***,struct Sim *);
//calls to hdf5 routines
void io_hdf5_out(struct IO *,struct Sim *,struct TimeStep *,char *);
void io_hdf5_in(struct IO *,struct Sim *,struct TimeStep *,char * );
#endif
