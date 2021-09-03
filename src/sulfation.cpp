#include "sulfation.h"
#include "levelSet.h"
//#include "test.h"

#include <cassert>

int contaF=0;

PetscErrorCode odeFun(AppContext &ctx, Vec POROSloc, Vec Uin, Vec Uout);
PetscErrorCode setMatValuesSC(AppContext &ctx, Vec U, DM da, Vec POROSloc, Mat A);
PetscErrorCode setMatValuesCSCC(AppContext &ctx, Vec U, DM da, Vec POROSloc, Mat A);

PetscErrorCode FormSulfationF(SNES snes,Vec U,Vec F,void *_ctx){
  PetscErrorCode ierr;
  AppContext * ctx_p = (AppContext *) _ctx;
  AppContext &ctx = *ctx_p;

  contaF++;
  if (contaF%100==0)
    PetscPrintf(PETSC_COMM_WORLD,"Computing F: %d\n",contaF);

  ierr = computePorosity(ctx, U, ctx.POROSloc);CHKERRQ(ierr);

  ierr = odeFun(ctx, ctx.POROSloc, U, F);
  PetscScalar ****u, ****f;
  PetscScalar ***nodetype, ***poros;

  ierr = DMDAVecGetArrayRead(ctx.daField[var::s], ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(ctx.daField[var::c], ctx.POROSloc, &poros);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOFRead(ctx.daAll, U, &u);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOFRead(ctx.daAll, F, &f);CHKERRQ(ierr);

  for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++){
    for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++){
      for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
        //const PetscScalar cRhoS = uIn[k][j][i][var::c] * poros[k][j][i] * uIn[k][j][i][var::s];
        if(nodetype[k][j][i]==N_INSIDE)
          f[k][j][i][var::s] = poros[k][j][i] * u[k][j][i][var::s]
                               + ctx.dt * (ctx.theta-1.0) * f[k][j][i][var::s];
        if(nodetype[k][j][i]>=0) //ghost points
          f[k][j][i][var::c] = u[k][j][i][var::c]
                               + ctx.dt * (ctx.theta-1.0) * f[k][j][i][var::c];
      }
    }
  }

  ierr = DMDAVecRestoreArrayRead(ctx.daField[var::s], ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(ctx.daField[var::c], ctx.POROSloc, &poros);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(ctx.daAll, U, &u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(ctx.daAll, F, &f);CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode FormSulfationRHS(AppContext &ctx,Vec U0,Vec F0){
  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD,"Computing RHS\n");

  ierr = computePorosity(ctx, U0, ctx.POROSloc);CHKERRQ(ierr);

  ierr = odeFun(ctx, ctx.POROSloc, U0, F0);CHKERRQ(ierr);
  PetscScalar ****u0, ****f0;
  PetscScalar ***nodetype, ***poros;

  ierr = DMDAVecGetArrayRead(ctx.daField[var::s], ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(ctx.daField[var::c], ctx.POROSloc, &poros);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOFRead(ctx.daAll, U0, &u0);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOFRead(ctx.daAll, F0, &f0);CHKERRQ(ierr);

  for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++){
    for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++){
      for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
        //const PetscScalar cRhoS = uIn[k][j][i][var::c] * poros[k][j][i] * uIn[k][j][i][var::s];
        if(nodetype[k][j][i]==N_INSIDE)
          f0[k][j][i][var::s] = poros[k][j][i] * u0[k][j][i][var::s]
                               + ctx.dt * ctx.theta * f0[k][j][i][var::s];
        if(nodetype[k][j][i]>=0){ //ghost points
          f0[k][j][i][var::s] = ctx.pb.sExt;
          f0[k][j][i][var::c] = u0[k][j][i][var::c]
                               + ctx.dt * ctx.theta * f0[k][j][i][var::c];
        }
      }
    }
  }

  ierr = DMDAVecRestoreArrayRead(ctx.daField[var::s], ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(ctx.daField[var::c], ctx.POROSloc, &poros);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(ctx.daAll, U0, &u0);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(ctx.daAll, F0, &f0);CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode FormSulfationJ(SNES snes,Vec U,Mat J, Mat P,void *_ctx){
  PetscErrorCode ierr;
  AppContext * ctx_p = (AppContext *) _ctx;
  AppContext &ctx = *ctx_p;

  PetscPrintf(PETSC_COMM_WORLD,"Computing J\n");
  ierr = PetscLogStagePush(ctx.logStages[ASSEMBLY]);CHKERRQ(ierr);

  ierr = computePorosity(ctx, U, ctx.POROSloc);CHKERRQ(ierr);

  {
    const PetscScalar As = ctx.pb.a / ctx.pb.mc;
    PetscScalar ***sigma, ***poros;
    PetscScalar ****u;
    ierr = DMDAVecGetArrayRead(ctx.daField[var::c], ctx.POROSloc, &poros);CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(ctx.daField[var::c], ctx.Sigma   , &sigma);CHKERRQ(ierr);
    ierr = DMDAVecGetArrayDOFRead(ctx.daAll, U , &u);CHKERRQ(ierr);
    for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++)
      for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++)
        for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++)
          sigma[k][j][i] = poros[k][j][i] +
                          ctx.dt * (ctx.theta-1.0) * As * u[k][j][i][var::c] * poros[k][j][i];
    ierr = DMDAVecRestoreArrayRead(ctx.daField[var::c], ctx.POROSloc, &poros);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(ctx.daField[var::c], ctx.Sigma   , &sigma);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayDOFRead(ctx.daAll, U , &u);CHKERRQ(ierr);
  }

  ierr = setMatValuesHelmoltz(ctx, ctx.daField[var::s], ctx.POROSloc, ctx.Sigma, ctx.pb.d*ctx.dt*(ctx.theta-1.0), P);CHKERRQ(ierr);
  ierr = setMatValuesSC(ctx, U, ctx.daField[var::s], ctx.POROSloc, P);CHKERRQ(ierr);
  ierr = setMatValuesCSCC(ctx, U, ctx.daField[var::c], ctx.POROSloc, P);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode setMatValuesCSCC(AppContext &ctx, Vec U, DM da, Vec POROSloc, Mat A)
{
  //Calls MatSetValues on A to insert values for the linear operator
  // c -> 1+alpha*s*(poros+c*dporos/dc) on inner and ghost nodes
  //
  // The matrix A should already exist and
  // assembly routines should be called afterwards by the caller.
  //
  // POROSloc should be a local vector with ghost values set correctly:
  // we do not call communication routines on gamma before using it.

  PetscErrorCode ierr;

  //PetscInt xs, ys, zs, xm, ym, zm;
  PetscScalar ***poros, ***nodetype;
  PetscScalar ****u;

  ierr = DMDAVecGetArrayRead(da, POROSloc, &poros);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da, ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOFRead(ctx.daAll, U, &u);CHKERRQ(ierr);

  for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++){
    for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++){
      for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
        const PetscScalar Ac = ctx.pb.a / ctx.pb.ms;
        MatStencil row, cols[2];
        row.i = i; row.j = j; row.k = k; row.c=var::c;
        cols[0].i = i; cols[0].j = j; cols[0].k = k; cols[0].c=var::c;
        cols[1].i = i; cols[1].j = j; cols[1].k = k; cols[1].c=var::s;
        PetscScalar vals[2]={1.,0.};
        if(nodetype[k][j][i]==N_INACTIVE)
        { //identity matrix
          MatSetValuesStencil(A,1,&row,2,cols,vals,INSERT_VALUES);
        }
        else if(nodetype[k][j][i]>=N_INSIDE ) //inner and ghosts
        {
          //diagonal in Jcc
          vals[0] = 1.0 + ctx.dt*(ctx.theta-1.0)*Ac * u[k][j][i][var::s] * (poros[k][j][i] + u[k][j][i][var::c] * ctx.pb.phiDer(u[k][j][i][var::c]));
          //diagonal in Jsc
          vals[1] = ctx.dt*(ctx.theta-1.0)*Ac * u[k][j][i][var::c] * poros[k][j][i];
          MatSetValuesStencil(A,1,&row,2,cols,vals,INSERT_VALUES);
        }
        else
          SETERRQ(PETSC_COMM_SELF,1,"Error: nodetype has values that are not supported.");
      }
    }
  }

  ierr = DMDAVecRestoreArrayRead(da, POROSloc, &poros);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da, ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(ctx.daAll, U, &u);CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode setMatValuesSC(AppContext &ctx, Vec U, DM da, Vec POROSloc, Mat A)
{
  PetscErrorCode ierr;

  PetscScalar ***poros, ***nodetype;
  PetscScalar ****u;

  Vec UinLoc;
  ierr = DMGetLocalVector(ctx.daAll,&UinLoc); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(ctx.daAll,U,INSERT_VALUES,UinLoc);CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(da, POROSloc, &poros);CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(da, ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (ctx.daAll,U,INSERT_VALUES,UinLoc);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOFRead(ctx.daAll, UinLoc, &u);CHKERRQ(ierr);

  const PetscScalar dx2=ctx.dx*ctx.dx;
  const PetscScalar dy2=ctx.dy*ctx.dy;
  const PetscScalar dz2=ctx.dz*ctx.dz;
  
  for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++){
    for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++){
      for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
        if (nodetype[k][j][i]==N_INSIDE ){  //inner 
          const PetscScalar As = ctx.pb.a / ctx.pb.mc;
          MatStencil row, cols[7];
          row.i = i; row.j = j; row.k = k; row.c=var::s;
          cols[0].i = i;   cols[0].j = j;   cols[0].k = k;   cols[0].c=var::c;
          cols[1].i = i-1; cols[1].j = j;   cols[1].k = k;   cols[1].c=var::c;
          cols[2].i = i+1; cols[2].j = j;   cols[2].k = k;   cols[2].c=var::c;
          cols[3].i = i;   cols[3].j = j-1; cols[3].k = k;   cols[3].c=var::c;
          cols[4].i = i;   cols[4].j = j+1; cols[4].k = k;   cols[4].c=var::c;
          cols[5].i = i;   cols[5].j = j;   cols[5].k = k-1; cols[5].c=var::c;
          cols[6].i = i;   cols[6].j = j;   cols[6].k = k+1; cols[6].c=var::c;
          
          const PetscScalar dsX0 = 0.5* ( u[k][j][i][var::s] - u[k][j][i-1][var::s]);
          const PetscScalar dsX1 = 0.5* ( u[k][j][i][var::s] - u[k][j][i+1][var::s]); //N.B. -diff
          const PetscScalar dsY0 = 0.5* ( u[k][j][i][var::s] - u[k][j-1][i][var::s]);
          const PetscScalar dsY1 = 0.5* ( u[k][j][i][var::s] - u[k][j+1][i][var::s]); //N.B. -diff
          const PetscScalar dsZ0 = 0.5* ( u[k][j][i][var::s] - u[k-1][j][i][var::s]);
          const PetscScalar dsZ1 = 0.5* ( u[k][j][i][var::s] - u[k+1][j][i][var::s]); //N.B. -diff

          const PetscScalar extraDiag = ctx.pb.phiDer(u[k][j][i][var::c]) * u[k][j][i][var::s]
                                        +ctx.dt*(ctx.theta-1.0)*As*u[k][j][i][var::s]
                                               *(poros[k][j][i] + ctx.pb.phiDer(u[k][j][i][var::c])*u[k][j][i][var::c]);
          const PetscScalar factor = ctx.pb.d * ctx.dt*(ctx.theta-1.0);
          
          PetscScalar vals[7]={
            extraDiag + 
            factor * ctx.pb.phiDer(u[k][j][i][var::c]) 
                   * ( (dsX1-dsX0)/dx2 + (dsY1-dsY0)/dy2 + (dsZ1-dsZ0)/dz2 ),
            factor * ctx.pb.phiDer(u[k][j][i-1][var::c]) * dsX0/dx2,
            factor * ctx.pb.phiDer(u[k][j][i+1][var::c]) * dsX1/dx2,
            factor * ctx.pb.phiDer(u[k][j-1][i][var::c]) * dsY0/dy2,
            factor * ctx.pb.phiDer(u[k][j+1][i][var::c]) * dsY1/dy2,
            factor * ctx.pb.phiDer(u[k-1][j][i][var::c]) * dsY0/dz2,
            factor * ctx.pb.phiDer(u[k+1][j][i][var::c]) * dsY1/dz2
            };
          MatSetValuesStencil(A,1,&row,7,cols,vals,INSERT_VALUES);
        }
      }
    }
  }

  ierr = DMDAVecRestoreArrayRead(da, POROSloc, &poros);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da, ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(ctx.daAll, UinLoc, &u);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(ctx.daAll,&UinLoc); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode computePorosity(AppContext &ctx, Vec U,Vec POROSloc){
  PetscErrorCode ierr;
  PetscScalar ***poros;
  PetscScalar ****u;

  ierr = DMDAVecGetArray(ctx.daField[var::c], POROSloc, &poros);CHKERRQ(ierr);
  //ierr = DMDAVecGetArrayRead(da, ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(ctx.daAll, U, &u);CHKERRQ(ierr);

  for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++)
    for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++)
      for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
        //PetscScalar c=u[k][j][i][var::c];
        //PetscPrintf(PETSC_COMM_WORLD,"(%d,%d,%d): c=%f\n",i,j,k,c);
        //PetscScalar p=ctx.pb.phi(c);
        //PetscPrintf(PETSC_COMM_WORLD,"            p=%f\n",i,j,k,p);
        poros[k][j][i] = ctx.pb.phi(u[k][j][i][var::c]);
      }

  ierr = DMDAVecRestoreArray(ctx.daField[var::c], POROSloc, &poros);CHKERRQ(ierr);
  //ierr = DMDAVecRestoreArrayRead(da, ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(ctx.daAll, U, &u);CHKERRQ(ierr);

  ierr =  DMLocalToLocalBegin(ctx.daField[var::c],POROSloc,INSERT_VALUES,POROSloc);
  ierr =  DMLocalToLocalEnd  (ctx.daField[var::c],POROSloc,INSERT_VALUES,POROSloc);

  return ierr;
}

PetscErrorCode setInitialData(AppContext &ctx, Vec U0){
  PetscErrorCode ierr;
  PetscScalar ****u;
  PetscScalar *** nodetype;

  ierr = DMDAVecGetArrayDOF(ctx.daAll, U0, &u);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(ctx.daField[var::s], ctx.NODETYPE, &nodetype);CHKERRQ(ierr);

  for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++)
    for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++)
      for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
        u[k][j][i][var::s] = (nodetype[k][j][i] <= N_INSIDE ? ctx.pb.s0 : ctx.pb.sExt);
        u[k][j][i][var::c] = ctx.pb.c0;
      }

  ierr = DMDAVecRestoreArrayDOF(ctx.daAll, U0, &u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(ctx.daField[var::s], ctx.NODETYPE, &nodetype);CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode setF0(AppContext &ctx, Vec F0){
  PetscErrorCode ierr;
  //PetscScalar ****u;
  //PetscScalar *** nodetype;

  //ierr = DMDAVecGetArrayDOF(ctx.daAll, U0, &u);CHKERRQ(ierr);
  //ierr = DMDAVecGetArrayRead(ctx.daAll, ctx.NODETYPE, &nodetype);CHKERRQ(ierr);

  //for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++)
    //for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++)
      //for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
        //u[k][j][i][var::s] = (nodetype[k][j][i] <= N_INSIDE ? ctx.pb.s0 : ctx.pb.sExt);
        //u[k][j][i][var::c] = ctx.pb.c0;
      //}

  //ierr = DMDAVecRestoreArrayDOF(ctx.daAll, U0, &u);CHKERRQ(ierr);
  //ierr = DMDAVecRestoreArrayRead(ctx.daAll, ctx.NODETYPE, &nodetype);CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode odeFun(AppContext &ctx, Vec POROSloc, Vec Uin, Vec Uout)
{
  // Uout = (rho*s;c) +alpha*f(s,c)
  // dove f(s,c) è il RHS della ODE solfatazione

  PetscErrorCode ierr;
  Vec UinLoc;
  ierr = DMGetLocalVector(ctx.daAll,&UinLoc); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(ctx.daAll,Uin,INSERT_VALUES,UinLoc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (ctx.daAll,Uin,INSERT_VALUES,UinLoc);CHKERRQ(ierr);

  PetscScalar ****uIn, ****uOut;
  PetscScalar ***nodetype, ***poros;

  const PetscScalar dx2=ctx.dx*ctx.dx;
  const PetscScalar dy2=ctx.dy*ctx.dy;
  const PetscScalar dz2=ctx.dz*ctx.dz;

  ierr = DMDAVecGetArrayRead(ctx.daField[var::s], ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(ctx.daField[var::c], POROSloc, &poros);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOFRead(ctx.daAll, Uin, &uIn);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOFRead(ctx.daAll, Uout, &uOut);CHKERRQ(ierr);

  const PetscScalar Ac = ctx.pb.a / ctx.pb.ms;
  const PetscScalar As = ctx.pb.a / ctx.pb.mc;

  for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++){
    for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++){
      for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
        const PetscScalar cRhoS = uIn[k][j][i][var::c] * poros[k][j][i] * uIn[k][j][i][var::s];
        if (nodetype[k][j][i]==N_INACTIVE) {
          uOut[k][j][i][var::s] = 0.;
          uOut[k][j][i][var::s] = 0.;
        } else if(nodetype[k][j][i]==N_INSIDE){
          const PetscScalar porosX1 = (poros[k][j][i] + poros[k][j][i-1]) / 2.;
          const PetscScalar porosX2 = (poros[k][j][i] + poros[k][j][i+1]) / 2.;
          const PetscScalar porosY1 = (poros[k][j][i] + poros[k][j-1][i]) / 2.;
          const PetscScalar porosY2 = (poros[k][j][i] + poros[k][j+1][i]) / 2.;
          const PetscScalar porosZ1 = (poros[k][j][i] + poros[k-1][j][i]) / 2.;
          const PetscScalar porosZ2 = (poros[k][j][i] + poros[k+1][j][i]) / 2.;

          uOut[k][j][i][var::s] = - As * cRhoS
                                  + ((porosX1+porosX2)/dx2+(porosY1+porosY2)/dy2+(porosZ1+porosZ2)/dz2) * uIn[k][j][i][var::s]
                                  -porosX1/dx2 * uIn[k][j][i-1][var::s]
                                  -porosX2/dx2 * uIn[k][j][i+1][var::s]
                                  -porosY1/dy2 * uIn[k][j-1][i][var::s]
                                  -porosY2/dy2 * uIn[k][j-1][i][var::s]
                                  -porosZ1/dz2 * uIn[k-1][j][i][var::s]
                                  -porosZ2/dz2 * uIn[k+1][j][i][var::s];
          uOut[k][j][i][var::c] = - Ac * cRhoS;
        } else { //ghost points
          assert(nodetype[k][j][i]>=0);
          uOut[k][j][i][var::s] = 0.;
          if (nodetype[k][j][i] < ctx.nn123){ // Ghost.Phi1
            ghost & current = ctx.Ghost.Phi1[nodetype[k][j][i]];
            for(int cont=0;cont<27;++cont){
                int kg_ghost=current.stencil[cont];
                int i_ghost, j_ghost, k_ghost;
                nGlob2IJK(ctx, kg_ghost, i_ghost, j_ghost, k_ghost);
                uOut[k][j][i][var::s] += current.coeffsD[cont] * uIn[k_ghost][j_ghost][i_ghost][var::s];
            }
            uOut[k][j][i][var::s] = - Ac * cRhoS;
          } else { // Ghost.Bdy
            SETERRQ(PETSC_COMM_SELF,1,"Not (yet) implemented");
          }
        }
      }
    }
  }

  ierr = DMDAVecRestoreArrayRead(ctx.daField[var::s], ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(ctx.daAll, Uin, &uIn);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(ctx.daAll, Uout, &uOut);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(ctx.daAll,&UinLoc); CHKERRQ(ierr);

  return ierr;
}

