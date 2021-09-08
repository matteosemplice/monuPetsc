#ifndef LEVELSET_H
#define LEVELSET_H

#include "appctx.h"
#include "domains.h"

#define N_INACTIVE  -4
#define N_GHOSTBDY  -3
#define N_GHOSTPHI1 -2
#define N_INSIDE    -1

PetscErrorCode setPhi(AppContext &ctx, levelSetFPointer domain);
PetscErrorCode setNormals(AppContext &ctx);
PetscErrorCode setBoundaryPoints(AppContext &ctx);

PetscErrorCode setGhost(AppContext &ctx);
PetscErrorCode setMatValuesHelmoltz(AppContext &ctx, DM da, Vec Gamma, Vec Sigma, PetscScalar alpha, Mat A);
//PetscErrorCode setMatrix(AppContext &ctx, Mat A);

PetscScalar checkInterp(AppContext &ctx,DMDACoor3d ***P,PetscScalar xC,PetscScalar yC,PetscScalar zC,int stencil[], double coeffsD[]);

inline void nGlob2IJK(AppContext ctx, int nGlob, int &i , int &j, int &k){
  k     = nGlob / (ctx.nnx*ctx.nny);
  nGlob = nGlob % (ctx.nnx*ctx.nny);
  j     = nGlob / ctx.nnx;
  i     = nGlob % ctx.nnx;
}

#endif
