#include "levelSetTest.h"
#include "test.h"

PetscErrorCode setRHS(AppContext &ctx)
{
  PetscErrorCode ierr;
  PetscPrintf(PETSC_COMM_WORLD, "set right-hand side...\n");

  PetscScalar ****rhs, ***nodetype;
  ierr = DMDAVecGetArrayDOF(ctx.daAll, ctx.RHS, &rhs);
  ierr = DMDAVecGetArrayRead(ctx.daField[var::s], ctx.NODETYPE, &nodetype);

  DMDACoor3d ***P, ***B;
  ierr = DMDAVecGetArrayRead(ctx.daCoord,ctx.coordsLocal,&P); CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(ctx.daCoord,ctx.BOUNDARY   ,&B); CHKERRQ(ierr);
  
  for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++){
    for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++){
      for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
        //PetscInt kg = i + j * nn1 + k * nn1 * nn2;
        if(nodetype[k][j][i] == -1){
            // ------------------------------
            //Inside: set r.h.s. function f
            // ------------------------------
            //PetscPrintf(PETSC_COMM_SELF,"#%d] Inner (%d,%d,%d) at (%f,%f,%f)\n",
              //ctx.rank,i,j,k,P[k][j][i].x, P[k][j][i].y, P[k][j][i].z);
            
            rhs[k][j][i][var::s]=f_(P[k][j][i].x, P[k][j][i].y, P[k][j][i].z);
            rhs[k][j][i][var::c]=c_(P[k][j][i].x, P[k][j][i].y, P[k][j][i].z);
        }
        else if (nodetype[k][j][i] >= 0) {
          if (nodetype[k][j][i] < ctx.nn123){
            // ------------------------------
            // Ghost.Phi1: set boundary condition on Phi1
            // ------------------------------
            //PetscPrintf(PETSC_COMM_SELF,"#%d] PhiBdry (%d,%d,%d) at (%f,%f,%f); P=(%f,%f,%f) \n",
              //ctx.rank,i,j,k,
              //B[k][j][i].x, B[k][j][i].y, B[k][j][i].z,
              //P[k][j][i].x, P[k][j][i].y, P[k][j][i].z);
            rhs[k][j][i][var::s]=s_(B[k][j][i].x, B[k][j][i].y, B[k][j][i].z);
            rhs[k][j][i][var::c]=c_(B[k][j][i].x, B[k][j][i].y, B[k][j][i].z);
          }
          else{
            // ------------------------------
            // Ghost.Bdy: set boundary condition on the box (ignore if the domain is contained in the inner part of the box)
            // ------------------------------
            //PetscPrintf(PETSC_COMM_SELF,"#%d] GhostBdry (%d,%d,%d) at (%f,%f,%f); P=(%f,%f,%f) \n",
              //ctx.rank,i,j,k,
              //B[k][j][i].x, B[k][j][i].y, B[k][j][i].z,
              //P[k][j][i].x, P[k][j][i].y, P[k][j][i].z);
            rhs[k][j][i][var::s]=s_(B[k][j][i].x, B[k][j][i].y, B[k][j][i].z);
            rhs[k][j][i][var::c]=c_(B[k][j][i].x, B[k][j][i].y, B[k][j][i].z);
          }
        } else {
            rhs[k][j][i][var::s]=0.;//s_(P[k][j][i].x, P[k][j][i].y, P[k][j][i].z);
            rhs[k][j][i][var::c]=0.;//c_(P[k][j][i].x, P[k][j][i].y, P[k][j][i].z);
        }
      }
    }
  }
             
  ierr = DMDAVecRestoreArrayDOF(ctx.daAll, ctx.RHS, &rhs);
  ierr = DMDAVecRestoreArrayRead(ctx.daField[var::s], ctx.NODETYPE, &nodetype);
  ierr = DMDAVecRestoreArrayRead(ctx.daCoord,ctx.coordsLocal,&P); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(ctx.daCoord,ctx.BOUNDARY,&B); CHKERRQ(ierr);

    //for(PetscInt level=0;level<levels;++level){
        //ierr = DMCreateGlobalVector(CartesianGrid3D_p[level], &RHS_p[level]);
    //}
    //VecCopy(RHS,RHS_p[0]);

  PetscPrintf(PETSC_COMM_WORLD, "set right-hand side...DONE\n");
  return ierr;
}

PetscErrorCode setSigma(AppContext &ctx)
/**************************************************************************/
{
  PetscErrorCode ierr;
  PetscPrintf(PETSC_COMM_WORLD,"set Sigma...\n");

  PetscScalar ***sigma;
  //ierr = DMCreateLocalVector(CartesianGrid3D, &local_Sigma);
  ierr = DMDAVecGetArrayWrite(ctx.daField[var::s], ctx.Sigma, &sigma);

  DMDACoor3d ***P;
  ierr = DMDAVecGetArrayRead(ctx.daCoord,ctx.coordsLocal,&P); CHKERRQ(ierr);
  
  for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++){
    for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++){
      for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
        sigma[k][j][i]=Sigma_(P[k][j][i].x, P[k][j][i].y, P[k][j][i].z);
      }
    }
  }

  ierr = DMDAVecRestoreArrayWrite(ctx.daField[var::s], ctx.Sigma, &sigma);

    //for(PetscInt level=0;level<levels;++level){
        //ierr = DMCreateGlobalVector(CartesianGrid3D_p[level], &Sigma_p[level]);
        //ierr = DMCreateLocalVector(CartesianGrid3D_p[level], &local_Sigma_p[level]);
        //ierr = DMDAVecGetArrayWrite(CartesianGrid3D_p[level], Sigma_p[level], &sigma);
        //ierr = DMDAGetCorners(CartesianGrid3D_p[level], &ys, &xs, &zs, &ym, &xm, &zm);

    //for (PetscInt k = zs; k < zs + zm; ++k)
        //for (PetscInt j = xs; j < xs + xm; ++j)
            //for (PetscInt i = ys; i < ys + ym; ++i)
                //sigma[k][j][i]=Sigma_(x_p[level][j],y_p[level][i],z_p[level][k]);

    //ierr = DMDAVecRestoreArrayWrite(CartesianGrid3D_p[level],Sigma_p[level],&sigma);
    //}

  PetscPrintf(PETSC_COMM_WORLD,"set Sigma... DONE\n");
  return ierr;
}

PetscErrorCode setGamma(AppContext &ctx)
/**************************************************************************/
{
  PetscErrorCode ierr;
  PetscPrintf(PETSC_COMM_WORLD,"set Gamma...\n");

  PetscScalar ***gamma;
  Vec Gamma;
  ierr = DMGetGlobalVector(ctx.daField[var::s], &Gamma); CHKERRQ(ierr);
  ierr = DMDAVecGetArrayWrite(ctx.daField[var::s], Gamma, &gamma); CHKERRQ(ierr);

  DMDACoor3d ***P;
  ierr = DMDAVecGetArrayRead(ctx.daCoord,ctx.coordsLocal,&P); CHKERRQ(ierr);
  for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++){
    for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++){
      for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
        gamma[k][j][i]=Gamma_(P[k][j][i].x, P[k][j][i].y, P[k][j][i].z);
      }
    }
  }
  ierr = DMDAVecRestoreArrayWrite(ctx.daField[var::s], Gamma, &gamma); CHKERRQ(ierr);

  ierr = DMGlobalToLocal(ctx.daField[var::s],Gamma, INSERT_VALUES,ctx.POROSloc);  CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(ctx.daField[var::s], &Gamma); CHKERRQ(ierr);


    //for(PetscInt level=0;level<levels;++level){
        //ierr = DMCreateGlobalVector(CartesianGrid3D_p[level], &Gamma_p[level]);
        //ierr = DMCreateLocalVector(CartesianGrid3D_p[level], &local_Gamma_p[level]);
        //ierr = DMDAVecGetArrayWrite(CartesianGrid3D_p[level], Gamma_p[level], &gamma);
        //ierr = DMDAGetCorners(CartesianGrid3D_p[level], &ys, &xs, &zs, &ym, &xm, &zm);

    //for (PetscInt k = zs; k < zs + zm; ++k)
        //for (PetscInt j = xs; j < xs + xm; ++j)
            //for (PetscInt i = ys; i < ys + ym; ++i)
                //gamma[k][j][i]=Gamma_(x_p[level][j],y_p[level][i],z_p[level][k]);

    //ierr = DMDAVecRestoreArrayWrite(CartesianGrid3D_p[level],Gamma_p[level],&gamma);
    //}

  PetscPrintf(PETSC_COMM_WORLD,"set Gamma... DONE\n");
  return ierr;
}

PetscErrorCode setExact(AppContext &ctx, Vec EXA)
{
  PetscErrorCode ierr;
  PetscPrintf(PETSC_COMM_WORLD, "Set exact solution...\n");

  PetscScalar ****exa, ***nodetype;
  ierr = DMDAVecGetArrayDOF(ctx.daAll, EXA, &exa);
  ierr = DMDAVecGetArrayRead(ctx.daField[var::s], ctx.NODETYPE, &nodetype);

  DMDACoor3d ***P;
  ierr = DMDAVecGetArrayRead(ctx.daCoord,ctx.coordsLocal,&P); CHKERRQ(ierr);
  
  for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++){
    for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++){
      for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
        //PetscInt kg = i + j * nn1 + k * nn1 * nn2;
        if(nodetype[k][j][i] > -4){
            // ------------------------------
            //Inside: set r.h.s. function f
            // ------------------------------
            //PetscPrintf(PETSC_COMM_SELF,"#%d] Inner (%d,%d,%d) at (%f,%f,%f)\n",
              //ctx.rank,i,j,k,P[k][j][i].x, P[k][j][i].y, P[k][j][i].z);
            
            exa[k][j][i][var::s]=s_(P[k][j][i].x, P[k][j][i].y, P[k][j][i].z);
            exa[k][j][i][var::c]=c_(P[k][j][i].x, P[k][j][i].y, P[k][j][i].z);
        }
        else{
            exa[k][j][i][var::s]=0.;
            exa[k][j][i][var::c]=0.;
        }
      }
    }
  }
             
  ierr = DMDAVecRestoreArrayDOF(ctx.daAll, EXA, &exa);
  ierr = DMDAVecRestoreArrayRead(ctx.daField[var::s], ctx.NODETYPE, &nodetype);
  ierr = DMDAVecRestoreArrayRead(ctx.daCoord,ctx.coordsLocal,&P); CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD, "Set exact solution: DONE.\n");
  return ierr;
}
