#include "appctx.h"

#include <petscsys.h>

PetscErrorCode cleanUpContext(AppContext & ctx){
  PetscErrorCode ierr;
  ierr = DMDestroy(&ctx.daAll); CHKERRQ(ierr);
  //DMDA for fields
  ierr = DMDestroy(&ctx.daField[0]); CHKERRQ(ierr);
  ierr = DMDestroy(&ctx.daField[1]); CHKERRQ(ierr);
  PetscFree(ctx.daField);
  //index sets
  ierr = ISDestroy(&ctx.is[0]); CHKERRQ(ierr);
  ierr = ISDestroy(&ctx.is[1]); CHKERRQ(ierr);
  PetscFree(ctx.is);

  ierr = MatDestroy(&ctx.J); CHKERRQ(ierr);

  ierr= VecDestroy(&ctx.U ); CHKERRQ(ierr);
  ierr= VecDestroy(&ctx.U0); CHKERRQ(ierr);
  ierr= VecDestroy(&ctx.F ); CHKERRQ(ierr);
  ierr= VecDestroy(&ctx.F0); CHKERRQ(ierr);
  ierr= VecDestroy(&ctx.Uloc); CHKERRQ(ierr);
  ierr= VecDestroy(&ctx.PHIloc); CHKERRQ(ierr);

  return ierr;
}
