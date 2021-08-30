#ifndef LEVELSET_H
#define LEVELSET_H

#include "appctx.h"

#define N_INACTIVE  -4
#define N_GHOSTBDY  -3
#define N_GHOSTPHI1 -2
#define N_INSIDE    -1

PetscErrorCode setPhi(AppContext &ctx);
PetscErrorCode setNormals(AppContext &ctx);
PetscErrorCode setBoundaryPoints(AppContext &ctx);

PetscErrorCode setGhost(AppContext &ctx);
PetscErrorCode setMatrix(AppContext &ctx, Mat A);

PetscScalar checkInterp(AppContext &ctx,DMDACoor3d ***P,PetscScalar xC,PetscScalar yC,PetscScalar zC,int stencil[], double coeffsD[]);

#endif
