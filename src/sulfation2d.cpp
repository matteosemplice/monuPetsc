
#include "appctx.h"
#include "sulfation1d.h"

#include <petscdmcomposite.h>

PetscErrorCode FormJacobian2d(SNES snes,Vec U,Mat J, Mat P,void *_ctx){
  PetscErrorCode ierr;
  AppContext * ctx = (AppContext *) _ctx;
  PetscScalar **phi;
  data_type **u;

  const PetscScalar As = ctx->pb.a/ctx->pb.mc;
  const PetscScalar Ac = ctx->pb.a/ctx->pb.ms;
  const PetscScalar dOverDX2 = ctx->pb.d / (ctx->dx*ctx->dx);
  const PetscScalar dtFactor = -ctx->dt*(1.-ctx->theta);

  MatZeroEntries(P);
  //Vettori locali con la U=(s,c)
  ierr = DMGlobalToLocalBegin(ctx->daAll,U,INSERT_VALUES,ctx->Uloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (ctx->daAll,U,INSERT_VALUES,ctx->Uloc); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(ctx->daAll,ctx->Uloc,&u); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(ctx->daField[var::c],ctx->POROSloc,&phi); CHKERRQ(ierr);
  //Calcolo di phi
  for (PetscInt j=ctx->daInfo.gys; j<ctx->daInfo.gys+ctx->daInfo.gym; j++)
    for (PetscInt i=ctx->daInfo.gxs; i<ctx->daInfo.gxs+ctx->daInfo.gxm; i++)
      phi[j][i] = ctx->pb.phi(u[j][i].c);

  MatStencil row,cols[3];
  PetscScalar vals[3]={0.,0.,0.};
  const PetscScalar factordDX2 = dtFactor * dOverDX2;

  //JacSS block
  row.c = var::s;
  for (int i=0; i<3; ++i){
    cols[i].c = var::s;
  }
  for (PetscInt j=ctx->daInfo.ys; j<ctx->daInfo.ys+ctx->daInfo.ym; j++)
    for (PetscInt i=ctx->daInfo.xs; i<ctx->daInfo.xs+ctx->daInfo.xm; i++){
      row.i = i; row.j=j;
      const PetscScalar diag = phi[j][i] + dtFactor*(-As*u[j][i].c*phi[j][i]);
      //x direction
      if (i == 0) {
        cols[0].i=i;   cols[0].j=j;
        cols[1].i=i+1; cols[1].j=j;
        vals[0] = diag -factordDX2*0.5*(phi[j][i+1]+3*phi[j][i]);
        vals[1] =       factordDX2*0.5*(phi[j][i+1]+  phi[j][i]);
        ierr = MatSetValuesStencil(P,1,&row,2,cols,vals,ADD_VALUES); CHKERRQ(ierr);
      } else if (i == ctx->daInfo.mx-1) {
        cols[0].i=i-1; cols[0].j=j;
        cols[1].i=i;   cols[1].j=j;
        vals[0] =        factordDX2*0.5*(phi[j][i]+phi[j][i-1]);
        vals[1] = diag - factordDX2*0.5*(phi[j][i]+phi[j][i-1]);
        MatSetValuesStencil(J,1,&row,2,cols,vals,ADD_VALUES);
      } else {
        cols[0].i=i-1; cols[0].j=j;
        cols[1].i=i;   cols[1].j=j;
        cols[2].i=i+1; cols[2].j=j;
        vals[0] =        factordDX2*0.5*(phi[j][i-1]+   phi[j][i]);
        vals[1] = diag - factordDX2*0.5*(phi[j][i-1]+2.*phi[j][i]+phi[j][i+1]);
        vals[2] =        factordDX2*0.5*(               phi[j][i]+phi[j][i+1]);
        MatSetValuesStencil(J,1,&row,3,cols,vals,ADD_VALUES);
      }
      //y direction
      if (j == 0) {
        cols[0].i=i; cols[0].j=j;
        cols[1].i=i; cols[1].j=j+1;
        vals[0] = - factordDX2*0.5*(phi[j+1][i]+3*phi[j][i]);
        vals[1] =   factordDX2*0.5*(phi[j+1][i]+  phi[j][i]);
        ierr = MatSetValuesStencil(P,1,&row,2,cols,vals,ADD_VALUES); CHKERRQ(ierr);
      } else if (j == ctx->daInfo.my-1) {
        cols[0].i=i; cols[0].j=j-1;
        cols[1].i=i; cols[1].j=j;
        vals[0] =   factordDX2*0.5*(phi[j][i]+phi[j-1][i]);
        vals[1] = - factordDX2*0.5*(phi[j][i]+phi[j-1][i]);
        MatSetValuesStencil(J,1,&row,2,cols,vals,ADD_VALUES);
      } else {
        cols[0].i=i; cols[0].j=j-1;
        cols[1].i=i; cols[1].j=j;
        cols[2].i=i; cols[2].j=j+1;
        vals[0] =   factordDX2*0.5*(phi[j-1][i]+   phi[j][i]);
        vals[1] = - factordDX2*0.5*(phi[j-1][i]+2.*phi[j][i]+phi[j+1][i]);
        vals[2] =   factordDX2*0.5*(               phi[j][i]+phi[j+1][i]);
        MatSetValuesStencil(J,1,&row,3,cols,vals,ADD_VALUES);
      }
    }

  //JacSC block
  for (int i=0; i<3; ++i)
    cols[i].c=var::c;
  for (PetscInt j=ctx->daInfo.ys; j<ctx->daInfo.ys+ctx->daInfo.ym; j++)
    for (PetscInt i=ctx->daInfo.xs; i<ctx->daInfo.xs+ctx->daInfo.xm; i++){
      row.i = i; row.j=j;
      const PetscScalar diag = u[j][i].s*ctx->pb.phiDer(u[j][i].c)
            + dtFactor*(-As)*u[j][i].s*(phi[j][i]+u[j][i].c*ctx->pb.phiDer(u[j][i].c));
      //x direction
      if (i == 0) {
        cols[0].i=i;   cols[0].j=j;
        cols[1].i=i+1; cols[1].j=j;
        vals[0] = diag + factordDX2*0.5*(u[j][i+1].s-3.*u[j][i].s+2.*ctx->pb.sExt) * ctx->pb.phiDer(u[j][i  ].c);
        vals[1] =        factordDX2*0.5*(u[j][i+1].s-   u[j][i].s                ) * ctx->pb.phiDer(u[j][i+1].c);
        ierr = MatSetValuesStencil(P,1,&row,2,cols,vals,ADD_VALUES); CHKERRQ(ierr);
      } else if (i == ctx->daInfo.mx-1) {
        cols[0].i=i-1; cols[0].j=j;
        cols[1].i=i;   cols[1].j=j;
        vals[0] =        factordDX2*0.5*(u[j][i-1].s - u[j][i].s) * ctx->pb.phiDer(u[j][i-1].c);
        vals[1] = diag + factordDX2*0.5*(u[j][i-1].s - u[j][i].s) * ctx->pb.phiDer(u[j][i  ].c);
        MatSetValuesStencil(J,1,&row,2,cols,vals,ADD_VALUES);
      } else {
        cols[0].i=i-1; cols[0].j=j;
        cols[1].i=i;   cols[1].j=j;
        cols[2].i=i+1; cols[2].j=j;
        vals[0] =        factordDX2*0.5*(u[j][i-1].s -    u[j][i].s             ) * ctx->pb.phiDer(u[j][i-1].c);
        vals[1] = diag + factordDX2*0.5*(u[j][i-1].s - 2.*u[j][i].s +u[j][i+1].s) * ctx->pb.phiDer(u[j][i].c);
        vals[2] =        factordDX2*0.5*(u[j][i+1].s -    u[j][i].s             ) * ctx->pb.phiDer(u[j][i-1].c);
        MatSetValuesStencil(J,1,&row,3,cols,vals,ADD_VALUES);
      }
      //y direction
      if (j == 0) {
        cols[0].i=i; cols[0].j=j;
        cols[1].i=i; cols[1].j=j+1;
        vals[0] = factordDX2*0.5*(u[j+1][i].s-3.*u[j][i].s+2.*ctx->pb.sExt) * ctx->pb.phiDer(u[j  ][i].c);
        vals[1] = factordDX2*0.5*(u[j+1][i].s-   u[j][i].s                ) * ctx->pb.phiDer(u[j+1][i].c);
        ierr = MatSetValuesStencil(P,1,&row,2,cols,vals,ADD_VALUES); CHKERRQ(ierr);
      } else if (j == ctx->daInfo.my-1) {
        cols[0].i=i; cols[0].j=j-1;
        cols[1].i=i; cols[1].j=j;
        vals[0] = factordDX2*0.5*(u[j-1][i].s - u[j][i].s) * ctx->pb.phiDer(u[j-1][i].c);
        vals[1] = factordDX2*0.5*(u[j-1][i].s - u[j][i].s) * ctx->pb.phiDer(u[j  ][i].c);
        MatSetValuesStencil(J,1,&row,2,cols,vals,ADD_VALUES);
      } else {
        cols[0].i=i; cols[0].j=j-1;
        cols[1].i=i; cols[1].j=j;
        cols[2].i=i; cols[2].j=j+1;
        vals[0] = factordDX2*0.5*(u[j-1][i].s -    u[j][i].s             ) * ctx->pb.phiDer(u[j-1][i].c);
        vals[1] = factordDX2*0.5*(u[j-1][i].s - 2.*u[j][i].s +u[j+1][i].s) * ctx->pb.phiDer(u[j  ][i].c);
        vals[2] = factordDX2*0.5*(u[j+1][i].s -    u[j][i].s             ) * ctx->pb.phiDer(u[j+1][i].c);
        MatSetValuesStencil(J,1,&row,3,cols,vals,ADD_VALUES);
      }
    }

  row.c=var::c;
  for (int i=0; i<3; ++i)
    cols[i].c=var::s;

  //JacCS block
  for (PetscInt j=ctx->daInfo.ys; j<ctx->daInfo.ys+ctx->daInfo.ym; j++)
    for (PetscInt i=ctx->daInfo.xs; i<ctx->daInfo.xs+ctx->daInfo.xm; i++){
      row.i = i; row.j=j;
      cols[0].i=i; cols[0].j=j;
      vals[0] = dtFactor*(-Ac)*u[j][i].c*phi[j][i];
      ierr = MatSetValuesStencil(P,1,&row,1,cols,vals,ADD_VALUES); CHKERRQ(ierr);
    }

  for (int i=0; i<3; ++i)
    cols[i].c=var::c;

  //JacCC block
  for (PetscInt j=ctx->daInfo.ys; j<ctx->daInfo.ys+ctx->daInfo.ym; j++)
    for (PetscInt i=ctx->daInfo.xs; i<ctx->daInfo.xs+ctx->daInfo.xm; i++){
      row.i = i; row.j=j;
      cols[0].i=i; cols[0].j=j;
      vals[0] = 1. + dtFactor*(-Ac)*u[j][i].s*(phi[j][i]+u[j][i].c*ctx->pb.phiDer(u[j][i].c));
      ierr = MatSetValuesStencil(P,1,&row,1,cols,vals,ADD_VALUES); CHKERRQ(ierr);
    }

  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  //Ripristino vettori della U e della PHI
  ierr = DMDAVecRestoreArray(ctx->daField[var::c],ctx->POROSloc,&phi); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(ctx->daAll,ctx->Uloc,&u); CHKERRQ(ierr);

  return ierr;

}

void formF2d(data_type **u, PetscScalar **phi, data_type **f, PetscScalar dtFactor, AppContext* ctx){
  const PetscScalar As = ctx->pb.a/ctx->pb.mc;
  const PetscScalar Ac = ctx->pb.a/ctx->pb.ms;
  const PetscScalar dOverDX2 = ctx->pb.d / (ctx->dx*ctx->dx);

  for (PetscInt j=ctx->daInfo.ys; j<ctx->daInfo.ys+ctx->daInfo.ym; j++)
    for (PetscInt i=ctx->daInfo.xs; i<ctx->daInfo.xs+ctx->daInfo.xm; i++){
      // s component
      const PetscScalar phicsc = u[j][i].s * u[j][i].c * phi[j][i];
      f[j][i].s=phi[j][i]*u[j][i].s - dtFactor * As * phicsc;

      //x direction
      if (i<ctx->daInfo.mx-1){//homog. Neumann east
        const PetscScalar phiMed = 0.5*(phi[j][i]+phi[j][i+1]);
        f[j][i].s += dtFactor * dOverDX2 * phiMed * (u[j][i+1].s-u[j][i].s);
      }

      if (i>0){
        const PetscScalar phiMed = 0.5*(phi[j][i]+phi[j][i-1]);
        f[j][i].s -= dtFactor * dOverDX2 * phiMed * (u[j][i].s-u[j][i-1].s);
      }
      else //Dirichlet condition west
        f[j][i].s -= dtFactor * dOverDX2 * phi[j][i] * (u[j][i].s-ctx->pb.sExt);

      //y direction
      if (j<ctx->daInfo.my-1){//homog. Neumann north
        const PetscScalar phiMed = 0.5*(phi[j][i]+phi[j+1][i]);
        f[j][i].s += dtFactor * dOverDX2 * phiMed * (u[j+1][i].s-u[j][i].s);
      }

      if (j>0){
        const PetscScalar phiMed = 0.5*(phi[j][i]+phi[j-1][i]);
        f[j][i].s -= dtFactor * dOverDX2 * phiMed * (u[j][i].s-u[j-1][i].s);
      }
      else //Dirichlet condition south
        f[j][i].s -= dtFactor * dOverDX2 * phi[j][i] * (u[j][i].s-ctx->pb.sExt);


      // c component
      f[j][i].c = u[j][i].c - dtFactor * Ac * phicsc;
    }
}

PetscErrorCode FormRHS2d(Vec F0,void *_ctx){
  PetscErrorCode ierr;
  AppContext * ctx = (AppContext *) _ctx;
  data_type **u0, **f0;
  PetscScalar **phi0;

  ierr = DMDAVecGetArray(ctx->daAll,F0,&f0); CHKERRQ(ierr);

  //Vettori locali con la U=(s,c)
  ierr = DMGlobalToLocalBegin(ctx->daAll,ctx->U0,INSERT_VALUES,ctx->Uloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (ctx->daAll,ctx->U0,INSERT_VALUES,ctx->Uloc); CHKERRQ(ierr);

  ierr = DMDAVecGetArray(ctx->daAll,ctx->Uloc,&u0); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(ctx->daField[var::c],ctx->POROSloc,&phi0); CHKERRQ(ierr);

  //Calcolo di phi
  //Calcolo di phi
  for (PetscInt j=ctx->daInfo.gys; j<ctx->daInfo.gys+ctx->daInfo.gym; j++)
    for (PetscInt i=ctx->daInfo.gxs; i<ctx->daInfo.gxs+ctx->daInfo.gxm; i++)
      phi0[j][i] = ctx->pb.phi(u0[j][i].c);

  //calcolo della F
  formF2d(u0,phi0,f0,ctx->dt*ctx->theta,ctx);

  //Ripristino vettori della U e della PHI
  ierr = DMDAVecRestoreArray(ctx->daField[var::c],ctx->POROSloc,&phi0); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(ctx->daAll,ctx->Uloc,&u0); CHKERRQ(ierr);

  //Ripristino vettori della F
  ierr = DMDAVecRestoreArray(ctx->daAll,F0,&f0); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode FormFunction2d(SNES snes,Vec U,Vec F,void *_ctx){
  PetscErrorCode ierr;
  AppContext * ctx = (AppContext *) _ctx;
  data_type **u, **f;
  PetscScalar **phi;

  ierr = DMDAVecGetArray(ctx->daAll,F,&f); CHKERRQ(ierr);

  //Vettori locali con la U=(s,c)
  ierr = DMGlobalToLocalBegin(ctx->daAll,U,INSERT_VALUES,ctx->Uloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (ctx->daAll,U,INSERT_VALUES,ctx->Uloc); CHKERRQ(ierr);

  //PetscPrintf(PETSC_COMM_WORLD,"====== (U) ======\n");
  //VecView(ctx->Uloc,PETSC_VIEWER_STDOUT_WORLD);
  //PetscPrintf(PETSC_COMM_WORLD,"=================\n");

  ierr = DMDAVecGetArray(ctx->daAll,ctx->Uloc,&u); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(ctx->daField[var::c],ctx->POROSloc,&phi); CHKERRQ(ierr);

  //Calcolo di phi
  for (PetscInt j=ctx->daInfo.gys; j<ctx->daInfo.gys+ctx->daInfo.gym; j++)
    for (PetscInt i=ctx->daInfo.gxs; i<ctx->daInfo.gxs+ctx->daInfo.gxm; i++)
      phi[j][i] = ctx->pb.phi(u[j][i].c);

  //calcolo della F
  formF2d(u,phi,f,-ctx->dt*(1.-ctx->theta),ctx);

  //Ripristino vettori della U e della PHI
  ierr = DMDAVecRestoreArray(ctx->daField[var::c],ctx->POROSloc,&phi); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(ctx->daAll,ctx->Uloc,&u); CHKERRQ(ierr);

  //Ripristino vettori della F
  ierr = DMDAVecRestoreArray(ctx->daAll,F,&f); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode setU02d(Vec U,void *_ctx){
  PetscErrorCode ierr;
  AppContext * ctx = (AppContext *) _ctx;

  ierr = VecISSet(ctx->U0,ctx->is[var::s],ctx->pb.s0); CHKERRQ(ierr);
  ierr = VecISSet(ctx->U0,ctx->is[var::c],ctx->pb.c0); CHKERRQ(ierr);

  return ierr;
}
