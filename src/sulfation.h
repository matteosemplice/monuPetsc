#ifndef __SULFATION__H
#define __SULFATION__H

#include "appctx.h"

#include <petscsnes.h>

PetscErrorCode setRHS(AppContext &ctx);
PetscErrorCode FormSulfationF(SNES snes,Vec U,Vec F,void *_ctx);
PetscErrorCode FormSulfationJ(SNES snes,Vec U,Mat J, Mat P,void *_ctx);

#endif
