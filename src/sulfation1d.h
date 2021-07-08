#include <petscsnes.h>

PetscErrorCode setU01d(Vec U,void *_ctx);

PetscErrorCode FormRHS1d(Vec F0,void *_ctx);
PetscErrorCode FormFunction1d(SNES snes,Vec U,Vec F,void *ctx);
PetscErrorCode FormJacobian1d(SNES snes,Vec U,Mat J, Mat P,void *ctx);

