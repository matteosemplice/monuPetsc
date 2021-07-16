#include <petscsnes.h>

PetscErrorCode setU03d(Vec U,void *_ctx);

PetscErrorCode FormRHS3d(Vec F0,void *_ctx);
PetscErrorCode FormFunction3d(SNES snes,Vec U,Vec F,void *ctx);
PetscErrorCode FormJacobian3d(SNES snes,Vec U,Mat J, Mat P,void *ctx);

