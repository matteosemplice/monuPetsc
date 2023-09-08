#ifndef __APPCTX_H
#define __APPCTX_H

#include <petscdmda.h>

#include <vector>

#define stWidth 4

enum block {ss=0,sc=1,cs=2,cc=3};
enum var {s=0,c=1};
typedef struct{ PetscScalar s,c;} data_type;

enum stageNames {BOUNDARY,STENCILS,ASSEMBLY,SOLVING};

typedef struct ghost {
    int index;//indice del punto ghost
    //double xc; //non sembra usato...
    //double yc; //non sembra usato...
    //double zc; //non sembra usato...
    //boundary point
    double xb;
    double yb;
    double zb;
    // indici celle dello stencil
    int stencil[27];
    //coeff per Dirichlet
    double coeffsD[27];
    //coeff per Neumann
    double coeffs_dx[27];
    double coeffs_dy[27];
    double coeffs_dz[27];
    //normale nel punto bordo
    double nx;
    double ny;
    double nz;
    double dtau;
} ghost;

typedef struct ghost_Bdy {
    int index;
     double xc;
     double yc;
     double zc;
     double xb;
     double yb;
     double zb;
     int type;
     int stencil[3];
     double coeffsD[3];
     double coeffsN[3];
     double dtau;
} ghost_Bdy;

template< class T1, class T2>
struct Phi12Bdy_struct
{
    std::vector<T1> Phi1;
    std::vector<T2> Bdy;
};

//Dati simulazione Semplice, SISC(2010)
class sulfationProblem{
  public:
  const PetscScalar c0=10.;
  const PetscScalar s0=0.;
        PetscScalar sExt=0.01;
        PetscScalar a=1.0e3;
  const PetscScalar alpha=0.01;
  const PetscScalar beta=0.1;
  const PetscScalar mc=100.09;
  const PetscScalar ms=64.06;
  const PetscScalar d=0.1;

  PetscScalar phi(PetscScalar c) const
    {return alpha*c+beta;}
  PetscScalar phiDer(PetscScalar c) const
    {return alpha;}
};

typedef struct {
  PetscInt nx,ny,nz; //no. of cells
  PetscInt nnx,nny,nnz,nn123; //no. of points in DMDA
  PetscScalar xmin, xmax, ymin, ymax, zmin, zmax; //domain bounding box
  PetscScalar dx,dy,dz; //cell size
  int rank, size;  //rank of processor
  DM daAll;       //global DA
  DMDALocalInfo daInfo;//, daInfoC, daInfoS;
  IS *is;
  DM *daField;

  DM daCoord;     //DA for coordinates
  Vec coordsLocal;

  Vec Phi, local_Phi; //levelset function
  Vec NORMALS,BOUNDARY;
  Vec NODETYPE, local_NodeType;
  Vec Sigma; //dovrebbero essere superflui...
  Vec RHS;

  //MG
  //- multigrid-levels:
    //multigrid-levels > 0 ==> the solver is multigrid
    //multigrid-levels = 0 ==> the solver is ksp with matrix-free set in setMatrixShell()
    //multigrid-levels = -1 ==> the solver is ksp with matrix set on 3D DMDA in setMatrix()
    //multigrid-levels = -2 ==> the solver is ksp with matrix set in a LEX order grid in setMatrixSEQ() (WARNING: it does not work with more than 1 proc)
  int mgLevels=-1;
  int solver;
  Phi12Bdy_struct<ghost,ghost_Bdy> Ghost;

  PetscInt dim=3;

  Mat J; //Jacobian
  //Mat Jb[4]; //Jacobian blocks

  Vec U,U0,F;
  Vec Uloc,POROSloc;
  sulfationProblem pb;
  PetscScalar aExpl, aImpl, RKtoll; //coeffs for the RK stages (used by SNES functions)
  PetscScalar dt; //time step
  PetscScalar tLoad=0.; //time step
  PetscInt nLoad=0; //time step

  char domainName[255];

  PetscLogStage logStages[4];
} AppContext;

PetscErrorCode cleanUpContext(AppContext & ctx);

#endif
