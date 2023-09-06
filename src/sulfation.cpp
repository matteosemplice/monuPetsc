#include "sulfation.h"
#include "levelSet.h"

#include <petscviewerhdf5.h>
#include <cassert>

/*
 * Using the DIRK embedded pair
 * l | l     0
 * 1 | (1-l) l
 * -----------
 *   | (1-l) l   <-- second order solution
 *   | (1-b) b   <-- first order solution
 * where l=sqrt(2)/2 and b is any number different from l
 *
 * First stage value solves
 *     U1 - (1-l)*dt*f(U1) = Un
 * Second stage value solves
 *     U2 -    l *dt*f(U2) = Un + (1-l)*dt*K1
 * and U2 coincides with U_{n+1}
 * and stages are K1=f(U1), K2=f(U2)
 * The estimate of the local truncation error is thus:
 *  (U2 - (Un+dt*(1-b)K1+dt*b*K2)) / dt  = (l-b)*K1 + (b-l)*K2
 *
 * A time step requires:
 *  1. solve for U1
 *  2. compute K1=f(U1)
 *  3. solve for U_{n+1}=U2
 *  4. compute K2=f(U2)
 *  5. compute error estimate using K1, K2
 *
 * The two nonlinear solves are generalized as:
 *
 *   U - lambda*dt*f(U) = Un + aExpl*dt*K1\
 * where aExpl=0        for the first stage
 *       aExpl=1-lambda for the second stage
 *
 * */

const PetscScalar RKlambda = 1.-sqrt(2.)/2.;
const PetscScalar RKb      = 0.5;
const PetscScalar aExpl[2] = {0. , 1.-RKlambda};

//int contaF=0;

PetscErrorCode odeFun(AppContext &ctx, Vec POROSloc, Vec Uin, Vec Uout);
PetscErrorCode setMatValuesSC(AppContext &ctx, Vec U, DM da, Vec POROSloc, Mat A);
PetscErrorCode setMatValuesCSCC(AppContext &ctx, Vec U, DM da, Vec POROSloc, Mat A);
PetscErrorCode FormStage(AppContext &ctx,Vec U,Vec F);

// Un is taken from ctx.U0, solution at time t_{n+1} is left in ctx.U
// Also ctx.dt is updated with the initial guess for the next step
PetscErrorCode computeSulfationStep(AppContext &ctx, SNES snes){
  PetscErrorCode ierr;
  PetscLogDouble timeStart, timeEnd;
  ierr=PetscTime(&timeStart);CHKERRQ(ierr);

  bool again=true;
  SNESConvergedReason reason;

  while (again){
    PetscPrintf(PETSC_COMM_WORLD," ==== Trying step with dt %3.2e \n",ctx.dt);
    ierr = VecZeroEntries(ctx.K1); CHKERRQ(ierr);
    ierr = VecZeroEntries(ctx.K2); CHKERRQ(ierr);
    ierr = FormSulfationRHS(ctx, ctx.U0, ctx.RHS, 0);CHKERRQ(ierr);
    ierr = VecCopy(ctx.U0,ctx.U); CHKERRQ(ierr);
    //ierr = PetscLogStagePush(ctx.logStages[SOLVING]);CHKERRQ(ierr);
    ierr = SNESSolve(snes,ctx.RHS,ctx.U); CHKERRQ(ierr); //solve for U1, first stage value
    ierr = SNESGetConvergedReason(snes, &reason); CHKERRQ(ierr);
    if (reason<0){
      ctx.dt *= 0.5;
      PetscPrintf(PETSC_COMM_WORLD," ---  Newton solver for stage 1 diverged: halving timestep (now %3.2e )\n",ctx.dt);
      continue;
    }
    ierr = FormStage(ctx,ctx.U,ctx.K1); CHKERRQ(ierr); // compute k1

    ierr = FormSulfationRHS(ctx, ctx.U0, ctx.RHS, 1);CHKERRQ(ierr);
    ierr = SNESSolve(snes,ctx.RHS,ctx.U); CHKERRQ(ierr); //solve for U2, second stage value and solution at t_{n+1}
    ierr = SNESGetConvergedReason(snes, &reason); CHKERRQ(ierr);
    if (reason<0){
      PetscPrintf(PETSC_COMM_WORLD," --- Newton solver for stage 2 diverged: halving timestep\n");
      ctx.dt *= 0.5;
      continue;
    }
    ierr = FormStage(ctx,ctx.U,ctx.K2); CHKERRQ(ierr); // compute k2

    //error estimate: eta = |U2-U1|/dt
    // K2 = a*K2+b*K1
    ierr = VecAXPBY(ctx.K2, (RKb-RKlambda), (RKlambda-RKb), ctx.K1); CHKERRQ(ierr);
    PetscScalar eta;
    ierr = VecNorm(ctx.K2, NORM_2, &eta);
    eta *= std::sqrt( ctx.dx*ctx.dx*ctx.dx );
    PetscScalar factor = ctx.RKtoll/eta;
    PetscPrintf(PETSC_COMM_WORLD," ---  RK dt factor= %3.2e \n",factor);

    //factor = std::min(factor, 2.0); //do not increase more than 2-fold
    //factor = std::max(factor, 0.25); //do not decrease more than 4-fold

    //step check/rejection and next dt estimate
    if (factor>=1.){
      ierr=PetscTime(&timeEnd);CHKERRQ(ierr);
      timeEnd-=timeStart;
      MPI_Reduce( (void *) &timeEnd, (void *) &timeStart, 1, MPI_DOUBLE, MPI_MIN, 0, PETSC_COMM_WORLD);
      PetscPrintf(PETSC_COMM_WORLD," ===  step of dt=%3.2e completed in (%f - ",ctx.dt,timeStart);
      MPI_Reduce( (void *) &timeEnd, (void *) &timeStart, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);
      PetscPrintf(PETSC_COMM_WORLD,"%f s).\n",timeStart);
      again = false;
    } else {
      PetscPrintf(PETSC_COMM_WORLD," /-/  rejected step of dt=%3.2e \n ",ctx.dt);
    }
    ctx.dt *= 0.95 * factor;

  } //while (again)

  //ierr = PetscLogStagePop();CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode FormSulfationF(SNES snes,Vec U,Vec F,void *_ctx){
  PetscErrorCode ierr;
  AppContext * ctx_p = (AppContext *) _ctx;
  AppContext &ctx = *ctx_p;

  //PetscPrintf(PETSC_COMM_WORLD,"Computing F ...");
  PetscLogDouble timeStart, timeEnd;
  ierr=PetscTime(&timeStart);CHKERRQ(ierr);
  //contaF++;
  //if (contaF%100==0)
    //PetscPrintf(PETSC_COMM_WORLD,"Computing F: %d\n",contaF);

  //Vec UinLoc;
  //ierr = DMGetLocalVector(ctx.daAll,&UinLoc); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(ctx.daAll,U,INSERT_VALUES,ctx.Uloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (ctx.daAll,U,INSERT_VALUES,ctx.Uloc);CHKERRQ(ierr);

  ierr = computePorosity(ctx, ctx.Uloc, ctx.POROSloc);CHKERRQ(ierr);
  ierr = odeFun(ctx, ctx.POROSloc, ctx.Uloc, F);
  //ierr = DMRestoreLocalVector(ctx.daAll,&UinLoc); CHKERRQ(ierr);

  PetscScalar ****u, ****f;
  PetscScalar ***nodetype, ***poros;

  ierr = DMDAVecGetArrayRead(ctx.daField[var::s], ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(ctx.daField[var::c], ctx.POROSloc, &poros);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOFRead(ctx.daAll, ctx.Uloc, &u);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOFRead(ctx.daAll, F, &f);CHKERRQ(ierr);

  for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++){
    for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++){
      for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
        //const PetscScalar cRhoS = uIn[k][j][i][var::c] * poros[k][j][i] * uIn[k][j][i][var::s];
        if(nodetype[k][j][i]==N_INSIDE)
          f[k][j][i][var::s] = poros[k][j][i] * u[k][j][i][var::s]
                               - ctx.dt * RKlambda * f[k][j][i][var::s];
        if(nodetype[k][j][i]>=0){ //ghost points
          f[k][j][i][var::s] = 0.;
          if (nodetype[k][j][i] < ctx.nn123){ // Ghost.Phi1
            ghost & current = ctx.Ghost.Phi1[nodetype[k][j][i]];
            for(int cont=0;cont<27;++cont){
              int kg_ghost=current.stencil[cont];
              int i_ghost, j_ghost, k_ghost;
              nGlob2IJK(ctx, kg_ghost, i_ghost, j_ghost, k_ghost);
              f[k][j][i][var::s] += current.coeffsD[cont] * u[k_ghost][j_ghost][i_ghost][var::s];
            }
          } else {
            // Ghost.Bdy
            SETERRQ(PETSC_COMM_SELF,1,"Not (yet) implemented");
          }
        }
        if(nodetype[k][j][i]>=N_INSIDE) //inner and ghost points
          f[k][j][i][var::c] = u[k][j][i][var::c]
                               - ctx.dt * RKlambda * f[k][j][i][var::c];
      }
    }
  }

  ierr = DMDAVecRestoreArrayRead(ctx.daField[var::s], ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(ctx.daField[var::c], ctx.POROSloc, &poros);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(ctx.daAll, ctx.Uloc, &u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(ctx.daAll, F, &f);CHKERRQ(ierr);

  ierr=PetscTime(&timeEnd);CHKERRQ(ierr);
  //PetscPrintf(PETSC_COMM_SELF,"%d] Computing F done (%f s).\n",ctx.rank,timeEnd-timeStart);
  //timeEnd-=timeStart;
  //MPI_Reduce(
    //(void *) &timeEnd,
    //(void *) &timeStart,
    //1,
    //MPI_DOUBLE,
    //MPI_MIN,
    //0,
    //PETSC_COMM_WORLD);
  //PetscPrintf(PETSC_COMM_WORLD," done in (%f - ",timeStart);
  //MPI_Reduce(
    //(void *) &timeEnd,
    //(void *) &timeStart,
    //1,
    //MPI_DOUBLE,
    //MPI_MAX,
    //0,
    //PETSC_COMM_WORLD);
  //PetscPrintf(PETSC_COMM_WORLD,"%f s).\n",timeStart);

  return ierr;
}

PetscErrorCode FormStage(AppContext &ctx,Vec U,Vec F){
  PetscErrorCode ierr;

  ierr = DMGlobalToLocalBegin(ctx.daAll,U,INSERT_VALUES,ctx.Uloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (ctx.daAll,U,INSERT_VALUES,ctx.Uloc);CHKERRQ(ierr);
  ierr = computePorosity(ctx, ctx.Uloc, ctx.POROSloc);CHKERRQ(ierr);

  ierr = odeFun(ctx, ctx.POROSloc, ctx.Uloc, F);CHKERRQ(ierr);

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
        if(nodetype[k][j][i] != N_INSIDE){ //ghost points
          f[k][j][i][var::s] = 0.;
          f[k][j][i][var::c] = 0.;
        }
      }
    }
  }

  ierr = DMDAVecRestoreArrayRead(ctx.daField[var::s], ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(ctx.daField[var::c], ctx.POROSloc, &poros);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(ctx.daAll, U, &u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(ctx.daAll, F, &f);CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode FormSulfationRHS(AppContext &ctx,Vec U0,Vec F0, int stage){
  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD,"Computing RHS\n");

  //Vec U0loc;
  //ierr = DMGetLocalVector(ctx.daAll,&U0loc); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(ctx.daAll,U0,INSERT_VALUES,ctx.Uloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (ctx.daAll,U0,INSERT_VALUES,ctx.Uloc);CHKERRQ(ierr);
  ierr = computePorosity(ctx, ctx.Uloc, ctx.POROSloc);CHKERRQ(ierr);

  ierr = odeFun(ctx, ctx.POROSloc, ctx.Uloc, F0);CHKERRQ(ierr);
  //ierr = DMRestoreLocalVector(ctx.daAll,&U0loc); CHKERRQ(ierr);

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
        if(nodetype[k][j][i]==N_INSIDE){
          f0[k][j][i][var::s] = poros[k][j][i] * u0[k][j][i][var::s]
                               + ctx.dt * aExpl[stage] * f0[k][j][i][var::s];
          f0[k][j][i][var::c] = u0[k][j][i][var::c]
                               + ctx.dt * aExpl[stage] * f0[k][j][i][var::c];
        }
        if(nodetype[k][j][i]>=0){ //ghost points
          f0[k][j][i][var::s] = ctx.pb.sExt;
          f0[k][j][i][var::c] = u0[k][j][i][var::c]
                               + ctx.dt * aExpl[stage] * f0[k][j][i][var::c];
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

  //PetscPrintf(PETSC_COMM_WORLD,"Computing J ...");
  PetscLogDouble timeStart, timeEnd;
  ierr=PetscTime(&timeStart);CHKERRQ(ierr);
  //ierr = PetscLogStagePush(ctx.logStages[ASSEMBLY]);CHKERRQ(ierr);

  //Vec UinLoc;
  //ierr = DMGetLocalVector(ctx.daAll,&UinLoc); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(ctx.daAll,U,INSERT_VALUES,ctx.Uloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (ctx.daAll,U,INSERT_VALUES,ctx.Uloc);CHKERRQ(ierr);

  ierr = computePorosity(ctx, ctx.Uloc, ctx.POROSloc);CHKERRQ(ierr);

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
                          ctx.dt * RKlambda * As * u[k][j][i][var::c] * poros[k][j][i];
    ierr = DMDAVecRestoreArrayRead(ctx.daField[var::c], ctx.POROSloc, &poros);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(ctx.daField[var::c], ctx.Sigma   , &sigma);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayDOFRead(ctx.daAll, U , &u);CHKERRQ(ierr);
  }

  ierr = setMatValuesHelmoltz(ctx, ctx.daField[var::s], ctx.POROSloc, ctx.Sigma, -ctx.pb.d*ctx.dt*RKlambda, P);CHKERRQ(ierr);
  ierr = setMatValuesSC(ctx, ctx.Uloc, ctx.daField[var::s], ctx.POROSloc, P);CHKERRQ(ierr);
  ierr = setMatValuesCSCC(ctx, U, ctx.daField[var::c], ctx.POROSloc, P);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    //ierr = DMRestoreLocalVector(ctx.daAll,&UinLoc); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr=PetscTime(&timeEnd);CHKERRQ(ierr);
  //PetscPrintf(PETSC_COMM_SELF,"%d] Computing J done (%f s).\n",ctx.rank,timeEnd-timeStart);

  //timeEnd-=timeStart;
  //MPI_Reduce(
    //(void *) &timeEnd,
    //(void *) &timeStart,
    //1,
    //MPI_DOUBLE,
    //MPI_MIN,
    //0,
    //PETSC_COMM_WORLD);
  //PetscPrintf(PETSC_COMM_WORLD," done in (%f - ",timeStart);
  //MPI_Reduce(
    //(void *) &timeEnd,
    //(void *) &timeStart,
    //1,
    //MPI_DOUBLE,
    //MPI_MAX,
    //0,
    //PETSC_COMM_WORLD);
  //PetscPrintf(PETSC_COMM_WORLD,"%f s).\n",timeStart);

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
          MatSetValuesStencil(A,1,&row,1,cols,vals,INSERT_VALUES);
        }
        else if(nodetype[k][j][i]>=N_INSIDE ) //inner and ghosts
        {
          //diagonal in Jcc
          vals[0] = 1.0 + ctx.dt*RKlambda*Ac * u[k][j][i][var::s] * (poros[k][j][i] + u[k][j][i][var::c] * ctx.pb.phiDer(u[k][j][i][var::c]));
          //diagonal in Jsc
          vals[1] =     + ctx.dt*RKlambda*Ac * u[k][j][i][var::c] * poros[k][j][i];
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

PetscErrorCode setMatValuesSC(AppContext &ctx, Vec UinLoc, DM da, Vec POROSloc, Mat A)
{
  PetscErrorCode ierr;

  PetscScalar ***poros, ***nodetype;
  PetscScalar ****u;

  ierr = DMDAVecGetArrayRead(da, POROSloc, &poros);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da, ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
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
                                        +ctx.dt*RKlambda*As*u[k][j][i][var::s]
                                               *(poros[k][j][i] + ctx.pb.phiDer(u[k][j][i][var::c])*u[k][j][i][var::c]);
          const PetscScalar factor = -ctx.pb.d * ctx.dt*RKlambda;
          
          PetscScalar vals[7]={
            extraDiag + 
            factor * ctx.pb.phiDer(u[k][j][i][var::c]) 
                   * ( (dsX1+dsX0)/dx2 + (dsY1+dsY0)/dy2 + (dsZ1+dsZ0)/dz2 ),
            factor * ctx.pb.phiDer(u[k][j][i-1][var::c]) * dsX0/dx2,
            factor * ctx.pb.phiDer(u[k][j][i+1][var::c]) * dsX1/dx2,
            factor * ctx.pb.phiDer(u[k][j-1][i][var::c]) * dsY0/dy2,
            factor * ctx.pb.phiDer(u[k][j+1][i][var::c]) * dsY1/dy2,
            factor * ctx.pb.phiDer(u[k-1][j][i][var::c]) * dsZ0/dz2,
            factor * ctx.pb.phiDer(u[k+1][j][i][var::c]) * dsZ1/dz2
            };
          MatSetValuesStencil(A,1,&row,7,cols,vals,INSERT_VALUES);
        }
      }
    }
  }

  ierr = DMDAVecRestoreArrayRead(da, POROSloc, &poros);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da, ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(ctx.daAll, UinLoc, &u);CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode computePorosity(AppContext &ctx, Vec Uloc,Vec POROSloc){
  PetscErrorCode ierr;
  PetscScalar ***poros;
  PetscScalar ****u;

  ierr = DMDAVecGetArray(ctx.daField[var::c], POROSloc, &poros);CHKERRQ(ierr);
  //ierr = DMDAVecGetArrayRead(da, ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(ctx.daAll, Uloc, &u);CHKERRQ(ierr);

  for (PetscInt k=ctx.daInfo.gzs; k<ctx.daInfo.gzs+ctx.daInfo.gzm; k++)
    for (PetscInt j=ctx.daInfo.gys; j<ctx.daInfo.gys+ctx.daInfo.gym; j++)
      for (PetscInt i=ctx.daInfo.gxs; i<ctx.daInfo.gxs+ctx.daInfo.gxm; i++){
        //PetscScalar c=u[k][j][i][var::c];
        //PetscPrintf(PETSC_COMM_WORLD,"(%d,%d,%d): c=%f\n",i,j,k,c);
        //PetscScalar p=ctx.pb.phi(c);
        //PetscPrintf(PETSC_COMM_WORLD,"            p=%f\n",i,j,k,p);
        poros[k][j][i] = ctx.pb.phi(u[k][j][i][var::c]);
      }

  ierr = DMDAVecRestoreArray(ctx.daField[var::c], POROSloc, &poros);CHKERRQ(ierr);
  //ierr = DMDAVecRestoreArrayRead(da, ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(ctx.daAll, Uloc, &u);CHKERRQ(ierr);

  ierr =  DMLocalToLocalBegin(ctx.daField[var::c],POROSloc,INSERT_VALUES,POROSloc);
  ierr =  DMLocalToLocalEnd  (ctx.daField[var::c],POROSloc,INSERT_VALUES,POROSloc);

  return ierr;
}

PetscErrorCode loadInitialData(AppContext &ctx, Vec &U0){
  PetscErrorCode ierr;

  char  hdf5name[256];
  PetscSNPrintf(hdf5name,256,"%s_%d.h5","monumento",ctx.nLoad);

  PetscViewer viewer;
  PetscPrintf(PETSC_COMM_WORLD,"Loading initial data from %s\n",hdf5name);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,hdf5name,FILE_MODE_READ,&viewer); CHKERRQ(ierr);
  Vec uField;

  //s
  ierr = DMGetGlobalVector(ctx.daField[var::s], &uField); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) uField, "S");
  ierr = VecLoad(uField,viewer); CHKERRQ(ierr);
  ierr = VecStrideScatter(uField,var::s,U0,INSERT_VALUES); CHKERRQ(ierr);
  //ierr = DMRestoreGlobalVector(ctx.daField[var::s], &uField); CHKERRQ(ierr);

  //c
  //ierr = DMGetGlobalVector(ctx.daField[var::c], &uField); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) uField, "C");
  ierr = VecLoad(uField,viewer); CHKERRQ(ierr);
  ierr = VecStrideScatter(uField,var::c,U0,INSERT_VALUES); CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(ctx.daField[var::c], &uField); CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode setInitialData(AppContext &ctx, Vec &U0){
  PetscErrorCode ierr;
  PetscScalar ****u;
  PetscScalar *** nodetype;

  ierr = PetscOptionsGetScalar(NULL,NULL,"-sExt",&(ctx.pb.sExt),NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"setting sExt to %f\n",ctx.pb.sExt);

  PetscBool tLoadGiven, nLoadGiven;
  ierr = PetscOptionsGetScalar(NULL,NULL,"-tload",&ctx.tLoad,&tLoadGiven);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt   (NULL,NULL,"-nload",&ctx.nLoad,&nLoadGiven);CHKERRQ(ierr);
  if (nLoadGiven){
    if (tLoadGiven){
      ierr = loadInitialData(ctx,U0);CHKERRQ(ierr);
      return ierr;
    } else {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER_INPUT,"Options -tload and -nload MUST be used together");
    }
  }

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

PetscErrorCode odeFun(AppContext &ctx, Vec POROSloc, Vec UinLoc, Vec Uout)
{
  // Uout = f(s,c)
  // dove f(s,c) è il RHS della ODE solfatazione
  // Nei ghost per s usa operatore lineare di interpolazione

  PetscErrorCode ierr;

  PetscScalar ****uIn, ****uOut;
  PetscScalar ***nodetype, ***poros;

  const PetscScalar dx2=ctx.dx*ctx.dx;
  const PetscScalar dy2=ctx.dy*ctx.dy;
  const PetscScalar dz2=ctx.dz*ctx.dz;

  ierr = DMDAVecGetArrayRead(ctx.daField[var::s], ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(ctx.daField[var::c], POROSloc, &poros);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOFRead(ctx.daAll, UinLoc, &uIn);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOFRead(ctx.daAll, Uout, &uOut);CHKERRQ(ierr);

  const PetscScalar Ac = ctx.pb.a / ctx.pb.ms;
  const PetscScalar As = ctx.pb.a / ctx.pb.mc;

  for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++){
    for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++){
      for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
        const PetscScalar cRhoS = uIn[k][j][i][var::c] * poros[k][j][i] * uIn[k][j][i][var::s];
        if (nodetype[k][j][i]==N_INACTIVE) {
          uOut[k][j][i][var::s] = 0.;
          uOut[k][j][i][var::c] = 0.;
        } else if(nodetype[k][j][i]==N_INSIDE){
          const PetscScalar porosX1 = (poros[k][j][i] + poros[k][j][i-1]) / 2.;
          const PetscScalar porosX2 = (poros[k][j][i] + poros[k][j][i+1]) / 2.;
          const PetscScalar porosY1 = (poros[k][j][i] + poros[k][j-1][i]) / 2.;
          const PetscScalar porosY2 = (poros[k][j][i] + poros[k][j+1][i]) / 2.;
          const PetscScalar porosZ1 = (poros[k][j][i] + poros[k-1][j][i]) / 2.;
          const PetscScalar porosZ2 = (poros[k][j][i] + poros[k+1][j][i]) / 2.;

          uOut[k][j][i][var::s] = - As * cRhoS
                                  + ctx.pb.d*(
                                  -((porosX1+porosX2)/dx2+(porosY1+porosY2)/dy2+(porosZ1+porosZ2)/dz2) * uIn[k][j][i][var::s]
                                  +porosX1/dx2 * uIn[k][j][i-1][var::s]
                                  +porosX2/dx2 * uIn[k][j][i+1][var::s]
                                  +porosY1/dy2 * uIn[k][j-1][i][var::s]
                                  +porosY2/dy2 * uIn[k][j+1][i][var::s]
                                  +porosZ1/dz2 * uIn[k-1][j][i][var::s]
                                  +porosZ2/dz2 * uIn[k+1][j][i][var::s]
                                  );
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
                //PetscPrintf(PETSC_COMM_WORLD," ghost %d -> %d with coeff %f\n", 2*(i+ctx.nnx*(j+ctx.nny*k)), 2*(kg_ghost), current.coeffsD[cont]);
            }
            //abort();
          } else { // Ghost.Bdy
            SETERRQ(PETSC_COMM_SELF,1,"Not (yet) implemented");
          }
          uOut[k][j][i][var::c] = - Ac * cRhoS;
        }
      }
    }
  }

  ierr = DMDAVecRestoreArrayRead(ctx.daField[var::s], ctx.NODETYPE, &nodetype);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(ctx.daField[var::c], POROSloc, &poros);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(ctx.daAll, UinLoc, &uIn);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(ctx.daAll, Uout, &uOut);CHKERRQ(ierr);

  return ierr;
}

