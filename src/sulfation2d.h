#include <petscsnes.h>

PetscErrorCode setU02d(Vec U,void *_ctx);

PetscErrorCode FormRHS2d(Vec F0,void *_ctx);
PetscErrorCode FormFunction2d(SNES snes,Vec U,Vec F,void *ctx);
PetscErrorCode FormJacobian2d(SNES snes,Vec U,Mat J, Mat P,void *ctx);

