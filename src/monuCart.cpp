static char help[] = "Cartesian monument sulfation\n";

/*
 * mpirun -np NPROC ./monuCart -options_file ../data.txt [altre opzioni]
 *
 * dove data.txt contiene le opzioni principali. Ad esempio
 *
 * ===== data.txt =====
 * -Nx 32
 * -Ny 32
 * -Nz 32
 * -xmin -1
 * -xmax 1
 * -ymin -1
 * -ymax 1
 * -zmin -1
 * -zmax 1
 * -mglevels -1
 * =====================
 *
 * Le opzioni date in fondo sovrascrivono quelle nel file data.txt
 */

#include <petscversion.h>
#include <petscdmda.h>
#include <petscviewer.h>
#include <petscsnes.h>
#include <petscsys.h>

#include "appctx.h"
//#include "sulfation1d.h"
//#include "sulfation2d.h"
#include "sulfation3d.h"
#include "hdf5Output.h"
#include "levelSet.h"
#include "levelSetTest.h"

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

  //Cpu rank
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&ctx.rank);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_SELF,"CPU Rank=%d\n",ctx.rank); //numero della CPU=0,1,2,...

  //Space dimensions
  //ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&ctx.dim,NULL);CHKERRQ(ierr);
  //PetscPrintf(PETSC_COMM_SELF,"Solving problem in %d space dimensions\n",ctx.dim);

  //Parametri della griglia
  ctx.nx=5;
  ierr = PetscOptionsGetInt(NULL,NULL,"-Nx",&ctx.nx,NULL);CHKERRQ(ierr);
  ctx.ny = (ctx.dim>1 ? ctx.nx : 1);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Ny",&ctx.ny,NULL);CHKERRQ(ierr);
  ctx.nz = (ctx.dim>2 ? ctx.nx : 1);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Nz",&ctx.nz,NULL);CHKERRQ(ierr);

  ctx.nnx = ctx.nx+1;
  ctx.nny = ctx.ny+1;
  ctx.nnz = ctx.nz+1;
  ctx.nn123  = ctx.nnx * ctx.nny * ctx.nnz;

  ierr = PetscOptionsGetScalar(NULL,NULL,"-xmin",&ctx.xmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-xmax",&ctx.xmax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-ymin",&ctx.ymin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-ymax",&ctx.ymax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-zmin",&ctx.zmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-zmax",&ctx.zmax,NULL);CHKERRQ(ierr);

  ctx.dx = (ctx.xmax-ctx.xmin) / (ctx.nx);
  ctx.dy = (ctx.ymax-ctx.ymin) / (ctx.ny);
  ctx.dz = (ctx.zmax-ctx.zmin) / (ctx.nz);

  PetscPrintf(PETSC_COMM_WORLD,"Grid of %dX%dX%d cells.\n",ctx.nx,ctx.ny,ctx.nz);

  ierr = PetscOptionsGetInt(NULL,NULL,"-mglevels",&ctx.mgLevels,NULL);CHKERRQ(ierr);
  if (ctx.mgLevels>0)
    ctx.solver=1;
  else {
    ctx.solver=ctx.mgLevels;
    ctx.mgLevels=0;
  }

  //Create DMDA
  ierr = DMDACreate3d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                      DMDA_STENCIL_STAR, //Note vtk needs BOX with many cpus
                      ctx.nnx,ctx.nny,ctx.nnz, //global dim
                      PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE, //n proc on each dim
                      2,stWidth, //dof, stencil width
                      NULL,NULL,NULL, //n nodes per direction on each cpu
                      &(ctx.daAll));
  CHKERRQ(ierr);

  ierr = DMSetFromOptions(ctx.daAll); CHKERRQ(ierr);
  ierr = DMSetUp(ctx.daAll); CHKERRQ(ierr); CHKERRQ(ierr);

  ierr = DMDASetUniformCoordinates(ctx.daAll, ctx.xmin, ctx.xmax, ctx.ymin, ctx.ymax, ctx.zmin, ctx.zmax); CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(ctx.daAll, &(ctx.daCoord)); CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(ctx.daAll,&(ctx.coordsLocal)); CHKERRQ(ierr);

  ierr = DMDASetFieldName(ctx.daAll,0,"s"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(ctx.daAll,1,"c"); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(ctx.daAll,&ctx.daInfo); CHKERRQ(ierr);

  ierr = DMCreateFieldDecomposition(ctx.daAll,NULL, NULL, &ctx.is, &ctx.daField); CHKERRQ(ierr);

  //load level-set function
  ierr = setPhi(ctx); CHKERRQ(ierr);
  ierr = setNormals(ctx); CHKERRQ(ierr);
  ierr = setBoundaryPoints(ctx); CHKERRQ(ierr);
  ierr = setGhost(ctx); CHKERRQ(ierr);

  {
    DMDACoor3d ***P;
    ierr = DMDAVecGetArrayRead(ctx.daCoord, ctx.coordsLocal, &P); CHKERRQ(ierr);
    for (auto it = ctx.Ghost.Phi1.begin(); it != ctx.Ghost.Phi1.end(); it++) {
      PetscScalar interpError = checkInterp(ctx,P,it->xb,it->yb,it->zb,it->stencil,it->coeffsD);
      if (fabs(interpError)>1e-15)
        PetscPrintf(PETSC_COMM_SELF,"MMMM Interpolation error for quadratic function at %d: %e\n",it->index,interpError);
    }
    ierr = DMDAVecRestoreArrayRead(ctx.daCoord, ctx.coordsLocal, &P); CHKERRQ(ierr);
  }

  //// Create solvers
  //ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  //ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);
  //ierr = SNESGetKSP(snes,&kspJ); CHKERRQ(ierr);
  //ierr = KSPGetPC(kspJ,&pcJ); CHKERRQ(ierr);
  //ierr = PCSetType(pcJ,PCFIELDSPLIT); CHKERRQ(ierr);
  //ierr = PCFieldSplitSetIS(pcJ,"s",ctx.is[var::s]); CHKERRQ(ierr);
  //ierr = PCFieldSplitSetIS(pcJ,"c",ctx.is[var::c]); CHKERRQ(ierr);
  //ierr = PCFieldSplitSetType(pcJ,PC_COMPOSITE_MULTIPLICATIVE); CHKERRQ(ierr);

  ////ierr = KSPSetType(kspJ,KSPPREONLY); CHKERRQ(ierr);
  ////ierr = PCSetType(pcJ,PCLU); CHKERRQ(ierr);

  //ierr = KSPSetFromOptions(kspJ);
  //ierr = PCSetFromOptions(pcJ);

  ierr = DMCreateGlobalVector(ctx.daAll,&ctx.U); CHKERRQ(ierr);
  ierr = VecDuplicate(ctx.U,&ctx.U0); CHKERRQ(ierr);
  ierr = VecDuplicate(ctx.U,&ctx.F ); CHKERRQ(ierr);
  ierr = VecDuplicate(ctx.U,&ctx.F0); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(ctx.daAll,&ctx.Uloc); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(ctx.daField[var::c],&ctx.POROSloc); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(ctx.daField[var::c],&ctx.Sigma); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(ctx.daField[var::s], &ctx.Sigma); CHKERRQ(ierr);
  ierr = setSigma(ctx); CHKERRQ(ierr);
  ierr = setGamma(ctx); CHKERRQ(ierr);

  //ierr = DMCreateMatrix(ctx.daAll,&ctx.J); CHKERRQ(ierr);
  ierr = setMatrix(ctx); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(ctx.daAll, &ctx.RHS); CHKERRQ(ierr);
  ierr = setRHS(ctx); CHKERRQ(ierr);

  //{
    //PetscViewer viewer;
    //PetscViewerASCIIOpen(PETSC_COMM_WORLD,"J.m",&viewer);
    //PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);
    //MatView(ctx.J,viewer);
    //PetscViewerDestroy(&viewer);
  //}
  //{
    //PetscViewer viewer;
    //PetscViewerASCIIOpen(PETSC_COMM_WORLD,"rhs.m",&viewer);
    //PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);
    //VecView(ctx.RHS,viewer);
    //PetscViewerDestroy(&viewer);
  //}

  SNES snes;
  KSP kspJ;
  PC pcJ;

  ierr = KSPCreate(PETSC_COMM_WORLD,&kspJ); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(kspJ); CHKERRQ(ierr);

  ierr = KSPSetOperators(kspJ,ctx.J,ctx.J); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Solving...\n");
  ierr = KSPSolve(kspJ, ctx.RHS, ctx.U0); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Done solving.\n");

  ierr = WriteHDF5(ctx, "soluzione", ctx.U0); CHKERRQ(ierr);

  ierr = setExact(ctx, ctx.U); CHKERRQ(ierr);
  ierr = WriteHDF5(ctx, "esatta", ctx.U); CHKERRQ(ierr);

  ierr = VecScale(ctx.U,-1);  
  ierr = MatMultAdd(ctx.J,ctx.U,ctx.RHS,ctx.RHS);//v3 = v2 + A * v1
  ierr = WriteHDF5(ctx, "residuo", ctx.RHS); CHKERRQ(ierr);

  {
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD,"res.m",&viewer);
    PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);
    VecView(ctx.RHS,viewer);
    PetscViewerDestroy(&viewer);
  }


  ////Use Crank-Nicolson
  //ctx.dt   =ctx.dx;
  //ctx.theta=0.5; //Crank-Nicolson, set to 0 for Implicit Euler

  //Initial data
  //ierr = setU03d(ctx.U0,ctx); CHKERRQ(ierr);

  //PetscViewer viewerS,viewerC;
  //PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,"SO2",PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&viewerS);
  //PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,"CaCO3",PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&viewerC);

  //switch (ctx.dim){
  //case 1: ierr = SNESSetFunction(snes,ctx.F      ,FormFunction1d,(void *) &ctx); CHKERRQ(ierr);
          //ierr = SNESSetJacobian(snes,ctx.J,ctx.J,FormJacobian1d,(void *) &ctx); CHKERRQ(ierr);
          //break;
  //case 2: ierr = SNESSetFunction(snes,ctx.F      ,FormFunction2d,(void *) &ctx); CHKERRQ(ierr);
          //ierr = SNESSetJacobian(snes,ctx.J,ctx.J,FormJacobian2d,(void *) &ctx); CHKERRQ(ierr);
          //break;
  //case 3: ierr = SNESSetFunction(snes,ctx.F      ,FormFunction3d,(void *) &ctx); CHKERRQ(ierr);
          //ierr = SNESSetJacobian(snes,ctx.J,ctx.J,FormJacobian3d,(void *) &ctx); CHKERRQ(ierr);
          //break;
  //}

  //PetscScalar t = 0.;
  //const PetscScalar tFinal = 1.0;
  //while (t<tFinal){
    //if (t+ctx.dt>=tFinal)
      //ctx.dt = (tFinal - t) + 1.e-15;

    //switch (ctx.dim){
    //case 1: ierr = FormRHS1d(ctx.F0,(void *) &ctx); CHKERRQ(ierr);
            //break;
    //case 2: ierr = FormRHS2d(ctx.F0,(void *) &ctx); CHKERRQ(ierr);
            //break;
    //case 3: ierr = FormRHS3d(ctx.F0,(void *) &ctx); CHKERRQ(ierr);
            //break;
    //default: SETERRQ(PETSC_COMM_WORLD,1,"Numero di dimensioni non gestito");
    //}

    //ierr = VecCopy(ctx.U0,ctx.U); CHKERRQ(ierr);
    //ierr = SNESSolve(snes,ctx.F0,ctx.U); CHKERRQ(ierr);

    //if (ctx.dim==1) {
      //Vec uS,uC;
      //ierr = VecGetSubVector(ctx.U,ctx.is[var::s],&uS); CHKERRQ(ierr);
      //ierr = VecView(uS,viewerS); CHKERRQ(ierr);
      //ierr = VecRestoreSubVector(ctx.U,ctx.is[var::s],&uS); CHKERRQ(ierr);
      //ierr = VecGetSubVector(ctx.U,ctx.is[var::c],&uC); CHKERRQ(ierr);
      //ierr = VecView(uC,viewerC); CHKERRQ(ierr);
      //ierr = VecRestoreSubVector(ctx.U,ctx.is[var::c],&uC); CHKERRQ(ierr);
    //}

    //t += ctx.dt;
    //ierr = VecSwap(ctx.U,ctx.U0); CHKERRQ(ierr);
    //PetscPrintf(PETSC_COMM_WORLD,"t=%f, still %g to go\n",t,std::max(tFinal-t,0.));
    //ctx.dt = ctx.dx;
  //}

  //PetscViewerDestroy(&viewerS);
  //PetscViewerDestroy(&viewerC);

  //ierr = WriteHDF5(ctx, "finale", ctx.U);

  //ierr = SNESDestroy(&snes); CHKERRQ(ierr);
  ierr = cleanUpContext(ctx); CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
