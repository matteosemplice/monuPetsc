#ifndef LEVEL_SET_TEST_H
#define LEVEL_SET_TEST_H

#include "appctx.h"

PetscErrorCode setRHS(AppContext &ctx);
PetscErrorCode setSigma(AppContext &ctx);
PetscErrorCode setGamma(AppContext &ctx);

PetscErrorCode setExact(AppContext &ctx, Vec EXA);

#endif
