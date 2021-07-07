
#include "appctx.h"
#include "sulfation1d.h"

#include <petscdmcomposite.h>

PetscErrorCode FormJacobian(SNES snes,Vec U,Mat J, Mat P,void *_ctx){
  PetscErrorCode ierr;
  AppContext * ctx = (AppContext *) _ctx;
  PetscScalar *phi;
  data_type *u;

  const PetscScalar As = ctx->pb.a/ctx->pb.mc;
  const PetscScalar Ac = ctx->pb.a/ctx->pb.ms;
  const PetscScalar dOverDX2 = ctx->pb.d / (ctx->dx*ctx->dx);
  const PetscScalar dtFactor = -ctx->dt*(1.-ctx->theta);

  MatZeroEntries(P);
  //Vettori locali con la U=(s,c)
  ierr = DMGlobalToLocalBegin(ctx->daAll,U,INSERT_VALUES,ctx->Uloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (ctx->daAll,U,INSERT_VALUES,ctx->Uloc); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(ctx->daAll,ctx->Uloc,&u); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(ctx->daField[var::c],ctx->PHIloc,&phi); CHKERRQ(ierr);
  //Calcolo di phi
  for (PetscInt i=ctx->daInfo.gxs; i<ctx->daInfo.gxs+ctx->daInfo.gxm; i++)
    phi[i] = ctx->pb.phi(u[i].c);

  MatStencil row,cols[3];
  PetscScalar vals[3]={0.,0.,0.};
  const PetscScalar factordDX2 = dtFactor * dOverDX2;

  //JacSS block
  row.c = var::s;
  for (int i=0; i<3; ++i){
    cols[i].c = var::s;
  }
  for (PetscInt i=ctx->daInfo.xs; i<ctx->daInfo.xs+ctx->daInfo.xm; i++){
    row.i = i;
    const PetscScalar diag = phi[i] + dtFactor*(-As*u[i].c*phi[i]);
    if (i == 0) {
      cols[0].i=i;
      cols[1].i=i+1;
      vals[0] = diag -factordDX2*0.5*(phi[1]+3*phi[0]);
      vals[1] =       factordDX2*0.5*(phi[1]+  phi[0]);
      ierr = MatSetValuesStencil(P,1,&row,2,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
    } else if (i == ctx->daInfo.mx-1) {
      cols[0].i=i-1;
      cols[1].i=i;
      vals[0] =        factordDX2*0.5*(phi[i]+phi[i-1]);
      vals[1] = diag - factordDX2*0.5*(phi[i]+phi[i-1]);
      MatSetValuesStencil(J,1,&row,2,cols,vals,INSERT_VALUES);
    } else {
      cols[0].i=i-1;
      cols[1].i=i;
      cols[2].i=i+1;
      vals[0] =        factordDX2*0.5*(phi[i-1]+   phi[i]);
      vals[1] = diag - factordDX2*0.5*(phi[i-1]+2.*phi[i]+phi[i+1]);
      vals[2] =        factordDX2*0.5*(            phi[i]+phi[i+1]);
      MatSetValuesStencil(J,1,&row,3,cols,vals,INSERT_VALUES);
    }
  }

  //JacSC block
  for (int i=0; i<3; ++i)
    cols[i].c=var::c;
  for (PetscInt i=ctx->daInfo.xs; i<ctx->daInfo.xs+ctx->daInfo.xm; i++){
    row.i = i;//-ctx->daInfoS.gxs;
    cols[0].i=i-1;
    cols[1].i=i;
    cols[2].i=i+1;
    const PetscScalar diag = u[i].s*ctx->pb.phiDer(u[i].c) + dtFactor*(-As)*u[i].s*(phi[i]+u[i].c*ctx->pb.phiDer(u[i].c));
    if (i == 0) {
      cols[0].i=i;
      cols[1].i=i+1;
      vals[0] = diag + factordDX2*0.5*(u[1].s-3.*u[0].s+2.*ctx->pb.sExt) * ctx->pb.phiDer(u[0].c);
      vals[1] =        factordDX2*0.5*(u[1].s-   u[0].s                ) * ctx->pb.phiDer(u[1].c);
      ierr = MatSetValuesStencil(P,1,&row,2,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
    } else if (i == ctx->daInfo.mx-1) {
      vals[0] =        factordDX2*0.5*(u[i-1].s - u[i].s) * ctx->pb.phiDer(u[i-1].c);
      vals[1] = diag + factordDX2*0.5*(u[i-1].s - u[i].s) * ctx->pb.phiDer(u[i  ].c);
      MatSetValuesStencil(J,1,&row,2,cols,vals,INSERT_VALUES);
    } else {
      vals[0] =        factordDX2*0.5*(u[i-1].s - u[i].s             ) * ctx->pb.phiDer(u[i-1].c);
      vals[1] = diag + factordDX2*0.5*(u[i-1].s - 2.*u[i].s +u[i+1].s) * ctx->pb.phiDer(u[i].c);
      vals[2] =        factordDX2*0.5*(u[i+1].s - u[i].s             ) * ctx->pb.phiDer(u[i-1].c);
      MatSetValuesStencil(J,1,&row,3,cols,vals,INSERT_VALUES);
    }
  }

  row.c=var::c;
  for (int i=0; i<3; ++i)
    cols[i].c=var::s;

  //JacCS block
  for (PetscInt i=ctx->daInfo.xs; i<ctx->daInfo.xs+ctx->daInfo.xm; i++){
    row.i = i;//-ctx->daInfoS.gxs;
    cols[0].i=i;
    vals[0] = dtFactor*(-Ac)*u[i].c*phi[i];
    ierr = MatSetValuesStencil(P,1,&row,1,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
  }

  for (int i=0; i<3; ++i)
    cols[i].c=var::c;

  //JacCC block
  for (PetscInt i=ctx->daInfo.xs; i<ctx->daInfo.xs+ctx->daInfo.xm; i++){
    row.i = i;
    cols[0].i=i;
    vals[0] = 1. + dtFactor*(-Ac)*u[i].s*(phi[i]+u[i].c*ctx->pb.phiDer(u[i].c));
    ierr = MatSetValuesStencil(P,1,&row,1,cols,vals,INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  //Ripristino vettori della U e della PHI
  ierr = DMDAVecRestoreArray(ctx->daField[var::c],ctx->PHIloc,&phi); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(ctx->daAll,ctx->Uloc,&u); CHKERRQ(ierr);

  return ierr;

}

void formF(data_type *u, PetscScalar *phi, data_type *f, PetscScalar dtFactor, AppContext* ctx){
  const PetscScalar As = ctx->pb.a/ctx->pb.mc;
  const PetscScalar Ac = ctx->pb.a/ctx->pb.ms;
  const PetscScalar dOverDX2 = ctx->pb.d / (ctx->dx*ctx->dx);

  for (PetscInt i=ctx->daInfo.xs; i<ctx->daInfo.xs+ctx->daInfo.xm; i++){
    // s component
    const PetscScalar phicsc = u[i].s * u[i].c * phi[i];
    f[i].s=phi[i]*u[i].s - dtFactor * As * phicsc;

    if (i<ctx->daInfo.mx-1){//homog. Neumann on the right
      const PetscScalar phiMed = 0.5*(phi[i]+phi[i+1]);
      f[i].s += dtFactor * dOverDX2 * phiMed * (u[i+1].s-u[i].s);
    }

    if (i>0){
      const PetscScalar phiMed = 0.5*(phi[i]+phi[i-1]);
      f[i].s -= dtFactor * dOverDX2 * phiMed * (u[i].s-u[i-1].s);
    }
    else
      f[0].s -= dtFactor * dOverDX2 * phi[0] * (u[0].s-ctx->pb.sExt);

    // c component
    f[i].c = u[i].c - dtFactor * Ac * phicsc;
  }
}

PetscErrorCode FormRHS(Vec F0,void *_ctx){
  PetscErrorCode ierr;
  AppContext * ctx = (AppContext *) _ctx;
  data_type *u0, *f0;
  PetscScalar *phi0;

  ierr = DMDAVecGetArray(ctx->daAll,F0,&f0); CHKERRQ(ierr);

  //Vettori locali con la U=(s,c)
  ierr = DMGlobalToLocalBegin(ctx->daAll,ctx->U0,INSERT_VALUES,ctx->Uloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (ctx->daAll,ctx->U0,INSERT_VALUES,ctx->Uloc); CHKERRQ(ierr);

  ierr = DMDAVecGetArray(ctx->daAll,ctx->Uloc,&u0); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(ctx->daField[var::c],ctx->PHIloc,&phi0); CHKERRQ(ierr);

  //Calcolo di phi
  for (PetscInt i=ctx->daInfo.gxs; i<ctx->daInfo.gxs+ctx->daInfo.gxm; i++)
    phi0[i] = ctx->pb.phi(u0[i].c);

  //calcolo della F
  formF(u0,phi0,f0,ctx->dt*ctx->theta,ctx);

  //Ripristino vettori della U e della PHI
  ierr = DMDAVecRestoreArray(ctx->daField[var::c],ctx->PHIloc,&phi0); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(ctx->daAll,ctx->Uloc,&u0); CHKERRQ(ierr);

  //Ripristino vettori della F
  ierr = DMDAVecRestoreArray(ctx->daAll,F0,&f0); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode FormFunction(SNES snes,Vec U,Vec F,void *_ctx){
  PetscErrorCode ierr;
  AppContext * ctx = (AppContext *) _ctx;
  data_type *u, *f;
  PetscScalar *phi;

  ierr = DMDAVecGetArray(ctx->daAll,F,&f); CHKERRQ(ierr);

  //Vettori locali con la U=(s,c)
  ierr = DMGlobalToLocalBegin(ctx->daAll,U,INSERT_VALUES,ctx->Uloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (ctx->daAll,U,INSERT_VALUES,ctx->Uloc); CHKERRQ(ierr);

  //PetscPrintf(PETSC_COMM_WORLD,"====== (U) ======\n");
  //VecView(ctx->Uloc,PETSC_VIEWER_STDOUT_WORLD);
  //PetscPrintf(PETSC_COMM_WORLD,"=================\n");

  ierr = DMDAVecGetArray(ctx->daAll,ctx->Uloc,&u); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(ctx->daField[var::c],ctx->PHIloc,&phi); CHKERRQ(ierr);

  //Calcolo di phi
  for (PetscInt i=ctx->daInfo.gxs; i<ctx->daInfo.gxs+ctx->daInfo.gxm; i++)
    phi[i] = ctx->pb.phi(u[i].c);

  //calcolo della F
  formF(u,phi,f,-ctx->dt*(1.-ctx->theta),ctx);

  //Ripristino vettori della U e della PHI
  ierr = DMDAVecRestoreArray(ctx->daField[var::c],ctx->PHIloc,&phi); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(ctx->daAll,ctx->Uloc,&u); CHKERRQ(ierr);

  //Ripristino vettori della F
  ierr = DMDAVecRestoreArray(ctx->daAll,F,&f); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode setU0(Vec U,void *_ctx){
  PetscErrorCode ierr;
  AppContext * ctx = (AppContext *) _ctx;

  ierr = VecISSet(ctx->U0,ctx->is[var::s],ctx->pb.s0); CHKERRQ(ierr);
  ierr = VecISSet(ctx->U0,ctx->is[var::c],ctx->pb.c0); CHKERRQ(ierr);

  return ierr;
}
