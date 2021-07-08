#ifndef __APPCTX_H
#define __APPCTX_H

#include <petscdmda.h>

#define stWidth 1

enum block {ss=0,sc=1,cs=2,cc=3};
enum var {s=0,c=1};
typedef struct{ PetscScalar s,c;} data_type;

//Dati simulazione Semplice, SISC(2010)
class sulfationProblem{
  public:
  const PetscScalar c0=10.;
  const PetscScalar s0=0.;
  const PetscScalar sExt=0.01;
  const PetscScalar a=1.0e3;
  const PetscScalar alpha=0.01;
  const PetscScalar beta=0.1;
  const PetscScalar mc=100.09;
  const PetscScalar ms=64.06;
  const PetscScalar d=1.0;

  PetscScalar phi(PetscScalar c) const
    {return alpha*c+beta;}
  PetscScalar phiDer(PetscScalar c) const
    {return alpha;}
};

typedef struct {
  PetscInt Nx,Ny,Nz; //no. of cells
  PetscScalar dx; //cell size
  PetscInt rank;  //rank of processor
  DM daAll;       //composite DA
  DMDALocalInfo daInfo;//, daInfoC, daInfoS;
  IS *is;
  DM *daField;

  PetscInt dim=3;

  Mat J; //Jacobian
  //Mat Jb[4]; //Jacobian blocks

  Vec U,U0,F,F0;
  Vec Uloc,PHIloc;
  sulfationProblem pb;
  PetscScalar theta; //theta method
  PetscScalar dt; //time step
} AppContext;

PetscErrorCode cleanUpContext(AppContext & ctx);

#endif
