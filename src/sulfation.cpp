#include "sulfation.h"
#include "levelSet.h"
//#include "test.h"

#include <cassert>

PetscErrorCode FormSulfationF(SNES snes,Vec U,Vec F,void *_ctx){
  PetscErrorCode ierr;
  AppContext * ctx_p = (AppContext *) _ctx;
  AppContext &ctx = *ctx_p;

  //TODO: update local porosity

  ierr = setMatrix(ctx, ctx.J); CHKERRQ(ierr);
  ierr = MatMult(ctx.J,U,F); CHKERRQ(ierr);// F = J*U
  return ierr;
}

PetscErrorCode FormSulfationJ(SNES snes,Vec U,Mat J, Mat P,void *_ctx){
  PetscErrorCode ierr;
  AppContext * ctx_p = (AppContext *) _ctx;
  AppContext &ctx = *ctx_p;

  //TODO: update local porosity

  ierr = setMatrix(ctx, P); CHKERRQ(ierr);

  return ierr;
}
