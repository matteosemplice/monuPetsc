static char help[] = "Cartesian monument sulfation\n";

#include <petscversion.h>
#include <petscdmda.h>
#include <petscviewer.h>
#include <petscsnes.h>
#include <petscsys.h>

#include "appctx.h"
#include "sulfation1d.h"
#include "sulfation2d.h"
#include "sulfation3d.h"
#include "hdf5Output.h"

int main(int argc, char **argv) {

  PetscErrorCode ierr;
  /* **************************************************************** */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help); CHKERRQ(ierr);
  /* **************************************************************** */
  {
    char petscVersion[255];
    ierr = PetscGetVersion(petscVersion,255);
    PetscPrintf(PETSC_COMM_WORLD,"Compiled with %s\n",petscVersion);
  }

  AppContext ctx;
  SNES snes;
  KSP kspJ;
  PC pcJ;

  //Cpu rank
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&ctx.rank);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_SELF,"CPU Rank=%d\n",ctx.rank); //numero della CPU=0,1,2,...

  //Space dimensions
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&ctx.dim,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_SELF,"Solving problem in %d space dimensions\n",ctx.dim);

  //Parametri della griglia
  ctx.Nx=5;
  ierr = PetscOptionsGetInt(NULL,NULL,"-Nx",&ctx.Nx,NULL);CHKERRQ(ierr);
  ctx.Ny = (ctx.dim>1 ? ctx.Nx : 1);
  ctx.Nz = (ctx.dim>2 ? ctx.Nx : 1);

  ctx.dx = 1.0 / (ctx.Nx-1);
  PetscPrintf(PETSC_COMM_WORLD,"Grid of %d cells per direction.\n",ctx.Nx);

  //Create DMDA
  switch (ctx.dim){
  case 1: ierr = DMDACreate1d(PETSC_COMM_WORLD,
                              DM_BOUNDARY_NONE,
                              ctx.Nx,
                              2,stWidth,NULL,
                              &(ctx.daAll));
          CHKERRQ(ierr);
          break;
  case 2: ierr = DMDACreate2d(PETSC_COMM_WORLD,
                              DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                              DMDA_STENCIL_STAR, //Note vtk needs BOX with many cpus
                              ctx.Nx,ctx.Ny, //global dim
                              PETSC_DECIDE,PETSC_DECIDE, //n proc on each dim
                              2,stWidth, //dof, stencil width
                              NULL, NULL, //n nodes per direction on each cpu
                              &(ctx.daAll));
          CHKERRQ(ierr);
          break;
  case 3: ierr = DMDACreate3d(PETSC_COMM_WORLD,
                              DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                              DMDA_STENCIL_STAR, //Note vtk needs BOX with many cpus
                              ctx.Nx,ctx.Ny,ctx.Nz, //global dim
                              PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE, //n proc on each dim
                              2,stWidth, //dof, stencil width
                              NULL,NULL,NULL, //n nodes per direction on each cpu
                              &(ctx.daAll));
          CHKERRQ(ierr);
          break;
  default: SETERRQ(PETSC_COMM_WORLD,1,"Numero di dimensioni non gestito");
  }

  ierr = DMSetFromOptions(ctx.daAll); CHKERRQ(ierr);
  ierr = DMSetUp(ctx.daAll); CHKERRQ(ierr); CHKERRQ(ierr);

  ierr = DMDASetUniformCoordinates(ctx.daAll, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0); CHKERRQ(ierr);
  ierr = DMDASetFieldName(ctx.daAll,0,"s"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(ctx.daAll,1,"c"); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(ctx.daAll,&ctx.daInfo); CHKERRQ(ierr);

  ierr = DMCreateFieldDecomposition(ctx.daAll,NULL, NULL, &ctx.is, &ctx.daField); CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&kspJ); CHKERRQ(ierr);
  ierr = KSPGetPC(kspJ,&pcJ); CHKERRQ(ierr);
  ierr = PCSetType(pcJ,PCFIELDSPLIT); CHKERRQ(ierr);
  ierr = PCFieldSplitSetIS(pcJ,"s",ctx.is[var::s]); CHKERRQ(ierr);
  ierr = PCFieldSplitSetIS(pcJ,"c",ctx.is[var::c]); CHKERRQ(ierr);
  ierr = PCFieldSplitSetType(pcJ,PC_COMPOSITE_MULTIPLICATIVE); CHKERRQ(ierr);

  //ierr = KSPSetType(kspJ,KSPPREONLY); CHKERRQ(ierr);
  //ierr = PCSetType(pcJ,PCLU); CHKERRQ(ierr);

  ierr = KSPSetFromOptions(kspJ);
  ierr = PCSetFromOptions(pcJ);

  ierr = DMCreateGlobalVector(ctx.daAll,&ctx.U); CHKERRQ(ierr);
  ierr = VecDuplicate(ctx.U,&ctx.U0); CHKERRQ(ierr);
  ierr = VecDuplicate(ctx.U,&ctx.F ); CHKERRQ(ierr);
  ierr = VecDuplicate(ctx.U,&ctx.F0); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(ctx.daAll,&ctx.Uloc); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(ctx.daField[var::c],&ctx.PHIloc); CHKERRQ(ierr);

  ierr = DMCreateMatrix(ctx.daAll,&ctx.J); CHKERRQ(ierr);

  //Use Crank-Nicolson
  ctx.dt   =ctx.dx;
  ctx.theta=0.5; //Crank-Nicolson, set to 0 for Implicit Euler

  //Initial data
  switch (ctx.dim){
  case 1: ierr = setU01d(ctx.U0,(void *) &ctx); CHKERRQ(ierr);
          break;
  case 2: ierr = setU02d(ctx.U0,(void *) &ctx); CHKERRQ(ierr);
          break;
  case 3: ierr = setU03d(ctx.U0,(void *) &ctx); CHKERRQ(ierr);
          break;
  }
  //PetscPrintf(PETSC_COMM_WORLD,"====== U0 ======\n");
  //ierr = VecView(ctx.U0,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  //PetscPrintf(PETSC_COMM_WORLD,"================\n");


  PetscViewer viewerS,viewerC;
  PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,"SO2",PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&viewerS);
  PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,"CaCO3",PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&viewerC);

  switch (ctx.dim){
  case 1: ierr = SNESSetFunction(snes,ctx.F      ,FormFunction1d,(void *) &ctx); CHKERRQ(ierr);
          ierr = SNESSetJacobian(snes,ctx.J,ctx.J,FormJacobian1d,(void *) &ctx); CHKERRQ(ierr);
          break;
  case 2: ierr = SNESSetFunction(snes,ctx.F      ,FormFunction2d,(void *) &ctx); CHKERRQ(ierr);
          ierr = SNESSetJacobian(snes,ctx.J,ctx.J,FormJacobian2d,(void *) &ctx); CHKERRQ(ierr);
          break;
  case 3: ierr = SNESSetFunction(snes,ctx.F      ,FormFunction3d,(void *) &ctx); CHKERRQ(ierr);
          ierr = SNESSetJacobian(snes,ctx.J,ctx.J,FormJacobian3d,(void *) &ctx); CHKERRQ(ierr);
          break;
  }

  PetscScalar t = 0.;
  const PetscScalar tFinal = 1.0;
  while (t<tFinal){
    if (t+ctx.dt>=tFinal)
      ctx.dt = (tFinal - t) + 1.e-15;

    switch (ctx.dim){
    case 1: ierr = FormRHS1d(ctx.F0,(void *) &ctx); CHKERRQ(ierr);
            break;
    case 2: ierr = FormRHS2d(ctx.F0,(void *) &ctx); CHKERRQ(ierr);
            break;
    case 3: ierr = FormRHS3d(ctx.F0,(void *) &ctx); CHKERRQ(ierr);
            break;
    default: SETERRQ(PETSC_COMM_WORLD,1,"Numero di dimensioni non gestito");
    }

    ierr = VecCopy(ctx.U0,ctx.U); CHKERRQ(ierr);
    ierr = SNESSolve(snes,ctx.F0,ctx.U); CHKERRQ(ierr);

    if (ctx.dim==1) {
      Vec uS,uC;
      ierr = VecGetSubVector(ctx.U,ctx.is[var::s],&uS); CHKERRQ(ierr);
      ierr = VecView(uS,viewerS); CHKERRQ(ierr);
      ierr = VecRestoreSubVector(ctx.U,ctx.is[var::s],&uS); CHKERRQ(ierr);
      ierr = VecGetSubVector(ctx.U,ctx.is[var::c],&uC); CHKERRQ(ierr);
      ierr = VecView(uC,viewerC); CHKERRQ(ierr);
      ierr = VecRestoreSubVector(ctx.U,ctx.is[var::c],&uC); CHKERRQ(ierr);
    }

    t += ctx.dt;
    ierr = VecSwap(ctx.U,ctx.U0); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"t=%f, still %g to go\n",t,std::max(tFinal-t,0.));
    ctx.dt = ctx.dx;
  }

  PetscViewerDestroy(&viewerS);
  PetscViewerDestroy(&viewerC);

  ierr = WriteHDF5(ctx, "finale", ctx.U);

  ierr = SNESDestroy(&snes); CHKERRQ(ierr);
  ierr = cleanUpContext(ctx); CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
