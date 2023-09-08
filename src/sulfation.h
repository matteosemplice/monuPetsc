#ifndef __SULFATION__H
#define __SULFATION__H

#include "appctx.h"

#include <petscsnes.h>

PetscErrorCode setInitialData(AppContext &ctx, Vec &U0);

PetscErrorCode computePorosity(AppContext &ctx, Vec U,Vec POROSloc);
PetscErrorCode FormSulfationF(SNES snes,Vec U,Vec F,void *_ctx);
PetscErrorCode FormSulfationRHS(AppContext &ctx, Vec U0,Vec F0, int stage);
PetscErrorCode FormSulfationJ(SNES snes,Vec U,Mat J, Mat P,void *_ctx);

#endif
