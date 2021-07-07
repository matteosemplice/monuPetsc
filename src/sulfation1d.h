#include <petscsnes.h>

PetscErrorCode setU0(Vec U,void *_ctx);

PetscErrorCode FormRHS(Vec F0,void *_ctx);
PetscErrorCode FormFunction(SNES snes,Vec U,Vec F,void *ctx);
PetscErrorCode FormJacobian(SNES snes,Vec U,Mat J, Mat P,void *ctx);

