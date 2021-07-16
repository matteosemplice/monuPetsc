
#include "appctx.h"
#include "sulfation1d.h"

#include <petscdmcomposite.h>

PetscErrorCode FormJacobian3d(SNES snes,Vec U,Mat J, Mat P,void *_ctx){
  PetscErrorCode ierr;
  AppContext * ctx = (AppContext *) _ctx;
  PetscScalar ***phi;
  data_type ***u;

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
  for (PetscInt k=ctx->daInfo.gzs; k<ctx->daInfo.gzs+ctx->daInfo.gzm; k++)
    for (PetscInt j=ctx->daInfo.gys; j<ctx->daInfo.gys+ctx->daInfo.gym; j++)
      for (PetscInt i=ctx->daInfo.gxs; i<ctx->daInfo.gxs+ctx->daInfo.gxm; i++)
        phi[k][j][i] = ctx->pb.phi(u[k][j][i].c);

  MatStencil row,cols[3];
  PetscScalar vals[3]={0.,0.,0.};
  const PetscScalar factordDX2 = dtFactor * dOverDX2;

  //JacSS block
  row.c = var::s;
  for (int i=0; i<3; ++i){
    cols[i].c = var::s;
  }
  for (PetscInt k=ctx->daInfo.zs; k<ctx->daInfo.zs+ctx->daInfo.zm; k++)
    for (PetscInt j=ctx->daInfo.ys; j<ctx->daInfo.ys+ctx->daInfo.ym; j++)
      for (PetscInt i=ctx->daInfo.xs; i<ctx->daInfo.xs+ctx->daInfo.xm; i++){
        row.i = i; row.j=j; row.k=k;
        const PetscScalar diag = phi[k][j][i] + dtFactor*(-As*u[k][j][i].c*phi[k][j][i]);
        //x direction
        if (i == 0) {
          cols[0].i=i;   cols[0].j=j; cols[0].k=k;
          cols[1].i=i+1; cols[1].j=j; cols[1].k=k;
          vals[0] = diag -factordDX2*0.5*(phi[k][j][i+1]+3*phi[k][j][i]);
          vals[1] =       factordDX2*0.5*(phi[k][j][i+1]+  phi[k][j][i]);
          ierr = MatSetValuesStencil(P,1,&row,2,cols,vals,ADD_VALUES); CHKERRQ(ierr);
        } else if (i == ctx->daInfo.mx-1) {
          cols[0].i=i-1; cols[0].j=j; cols[0].k=k;
          cols[1].i=i;   cols[1].j=j; cols[1].k=k;
          vals[0] =        factordDX2*0.5*(phi[k][j][i]+phi[k][j][i-1]);
          vals[1] = diag - factordDX2*0.5*(phi[k][j][i]+phi[k][j][i-1]);
          MatSetValuesStencil(J,1,&row,2,cols,vals,ADD_VALUES);
        } else {
          cols[0].i=i-1; cols[0].j=j; cols[0].k=k;
          cols[1].i=i;   cols[1].j=j; cols[1].k=k;
          cols[2].i=i+1; cols[2].j=j; cols[2].k=k;
          vals[0] =        factordDX2*0.5*(phi[k][j][i-1]+   phi[k][j][i]);
          vals[1] = diag - factordDX2*0.5*(phi[k][j][i-1]+2.*phi[k][j][i]+phi[k][j][i+1]);
          vals[2] =        factordDX2*0.5*(               phi[k][j][i]+phi[k][j][i+1]);
          MatSetValuesStencil(J,1,&row,3,cols,vals,ADD_VALUES);
        }
        //y direction
        if (j == 0) {
          cols[0].i=i; cols[0].j=j;   cols[0].k=k;
          cols[1].i=i; cols[1].j=j+1; cols[1].k=k;
          vals[0] = - factordDX2*0.5*(phi[k][j+1][i]+3*phi[k][j][i]);
          vals[1] =   factordDX2*0.5*(phi[k][j+1][i]+  phi[k][j][i]);
          ierr = MatSetValuesStencil(P,1,&row,2,cols,vals,ADD_VALUES); CHKERRQ(ierr);
        } else if (j == ctx->daInfo.my-1) {
          cols[0].i=i; cols[0].j=j-1; cols[0].k=k;
          cols[1].i=i; cols[1].j=j;   cols[1].k=k;
          vals[0] =   factordDX2*0.5*(phi[k][j][i]+phi[k][j-1][i]);
          vals[1] = - factordDX2*0.5*(phi[k][j][i]+phi[k][j-1][i]);
          MatSetValuesStencil(J,1,&row,2,cols,vals,ADD_VALUES);
        } else {
          cols[0].i=i; cols[0].j=j-1; cols[0].k=k;
          cols[1].i=i; cols[1].j=j;   cols[1].k=k;
          cols[2].i=i; cols[2].j=j+1; cols[2].k=k;
          vals[0] =   factordDX2*0.5*(phi[k][j-1][i]+   phi[k][j][i]);
          vals[1] = - factordDX2*0.5*(phi[k][j-1][i]+2.*phi[k][j][i]+phi[k][j+1][i]);
          vals[2] =   factordDX2*0.5*(               phi[k][j][i]+phi[k][j+1][i]);
          MatSetValuesStencil(J,1,&row,3,cols,vals,ADD_VALUES);
        }
        //z direction
        if (k == 0) {
          cols[0].i=i; cols[0].j=j; cols[0].k=k;
          cols[1].i=i; cols[1].j=j; cols[1].k=k+1;
          vals[0] = - factordDX2*0.5*(phi[k+1][j][i]+3*phi[k][j][i]);
          vals[1] =   factordDX2*0.5*(phi[k+1][j][i]+  phi[k][j][i]);
          ierr = MatSetValuesStencil(P,1,&row,2,cols,vals,ADD_VALUES); CHKERRQ(ierr);
        } else if (k == ctx->daInfo.mz-1) {
          cols[0].i=i; cols[0].j=j; cols[0].k=k-1;
          cols[1].i=i; cols[1].j=j; cols[1].k=k;
          vals[0] =   factordDX2*0.5*(phi[k][j][i]+phi[k-1][j][i]);
          vals[1] = - factordDX2*0.5*(phi[k][j][i]+phi[k-1][j][i]);
          MatSetValuesStencil(J,1,&row,2,cols,vals,ADD_VALUES);
        } else {
          cols[0].i=i; cols[0].j=j; cols[0].k=k-1;
          cols[1].i=i; cols[1].j=j; cols[1].k=k;
          cols[2].i=i; cols[2].j=j; cols[2].k=k+1;
          vals[0] =   factordDX2*0.5*(phi[k-1][j][i]+   phi[k][j][i]);
          vals[1] = - factordDX2*0.5*(phi[k-1][j][i]+2.*phi[k][j][i]+phi[k+1][j][i]);
          vals[2] =   factordDX2*0.5*(                  phi[k][j][i]+phi[k+1][j][i]);
          MatSetValuesStencil(J,1,&row,3,cols,vals,ADD_VALUES);
        }
      }

  //JacSC block
  for (int i=0; i<3; ++i)
    cols[i].c=var::c;
  for (PetscInt k=ctx->daInfo.zs; k<ctx->daInfo.zs+ctx->daInfo.zm; k++)
    for (PetscInt j=ctx->daInfo.ys; j<ctx->daInfo.ys+ctx->daInfo.ym; j++)
      for (PetscInt i=ctx->daInfo.xs; i<ctx->daInfo.xs+ctx->daInfo.xm; i++){
        row.i = i; row.j=j; row.k=k;
        const PetscScalar diag = u[k][j][i].s*ctx->pb.phiDer(u[k][j][i].c)
              + dtFactor*(-As)*u[k][j][i].s*(phi[k][j][i]+u[k][j][i].c*ctx->pb.phiDer(u[k][j][i].c));
        //x direction
        if (i == 0) {
          cols[0].i=i;   cols[0].j=j; cols[0].k=k;
          cols[1].i=i+1; cols[1].j=j; cols[1].k=k;
          vals[0] = diag + factordDX2*0.5*(u[k][j][i+1].s-3.*u[k][j][i].s+2.*ctx->pb.sExt) * ctx->pb.phiDer(u[k][j][i  ].c);
          vals[1] =        factordDX2*0.5*(u[k][j][i+1].s-   u[k][j][i].s                ) * ctx->pb.phiDer(u[k][j][i+1].c);
          ierr = MatSetValuesStencil(P,1,&row,2,cols,vals,ADD_VALUES); CHKERRQ(ierr);
        } else if (i == ctx->daInfo.mx-1) {
          cols[0].i=i-1; cols[0].j=j; cols[0].k=k;
          cols[1].i=i;   cols[1].j=j; cols[1].k=k;
          vals[0] =        factordDX2*0.5*(u[k][j][i-1].s - u[k][j][i].s) * ctx->pb.phiDer(u[k][j][i-1].c);
          vals[1] = diag + factordDX2*0.5*(u[k][j][i-1].s - u[k][j][i].s) * ctx->pb.phiDer(u[k][j][i  ].c);
          MatSetValuesStencil(J,1,&row,2,cols,vals,ADD_VALUES);
        } else {
          cols[0].i=i-1; cols[0].j=j; cols[0].k=k;
          cols[1].i=i;   cols[1].j=j; cols[1].k=k;
          cols[2].i=i+1; cols[2].j=j; cols[2].k=k;
          vals[0] =        factordDX2*0.5*(u[k][j][i-1].s -    u[k][j][i].s             )    * ctx->pb.phiDer(u[k][j][i-1].c);
          vals[1] = diag + factordDX2*0.5*(u[k][j][i-1].s - 2.*u[k][j][i].s +u[k][j][i+1].s) * ctx->pb.phiDer(u[k][j][i].c);
          vals[2] =        factordDX2*0.5*(u[k][j][i+1].s -    u[k][j][i].s             )    * ctx->pb.phiDer(u[k][j][i-1].c);
          MatSetValuesStencil(J,1,&row,3,cols,vals,ADD_VALUES);
        }
        //y direction
        if (j == 0) {
          cols[0].i=i; cols[0].j=j;   cols[0].k=k;
          cols[1].i=i; cols[1].j=j+1; cols[1].k=k;
          vals[0] = factordDX2*0.5*(u[k][j+1][i].s-3.*u[k][j][i].s+2.*ctx->pb.sExt) * ctx->pb.phiDer(u[k][j  ][i].c);
          vals[1] = factordDX2*0.5*(u[k][j+1][i].s-   u[k][j][i].s                ) * ctx->pb.phiDer(u[k][j+1][i].c);
          ierr = MatSetValuesStencil(P,1,&row,2,cols,vals,ADD_VALUES); CHKERRQ(ierr);
        } else if (j == ctx->daInfo.my-1) {
          cols[0].i=i; cols[0].j=j-1; cols[0].k=k;
          cols[1].i=i; cols[1].j=j;   cols[1].k=k;
          vals[0] = factordDX2*0.5*(u[k][j-1][i].s - u[k][j][i].s) * ctx->pb.phiDer(u[k][j-1][i].c);
          vals[1] = factordDX2*0.5*(u[k][j-1][i].s - u[k][j][i].s) * ctx->pb.phiDer(u[k][j  ][i].c);
          MatSetValuesStencil(J,1,&row,2,cols,vals,ADD_VALUES);
        } else {
          cols[0].i=i; cols[0].j=j-1; cols[0].k=k;
          cols[1].i=i; cols[1].j=j;   cols[1].k=k;
          cols[2].i=i; cols[2].j=j+1; cols[2].k=k;
          vals[0] = factordDX2*0.5*(u[k][j-1][i].s -    u[k][j][i].s             )    * ctx->pb.phiDer(u[k][j-1][i].c);
          vals[1] = factordDX2*0.5*(u[k][j-1][i].s - 2.*u[k][j][i].s +u[k][j+1][i].s) * ctx->pb.phiDer(u[k][j  ][i].c);
          vals[2] = factordDX2*0.5*(u[k][j+1][i].s -    u[k][j][i].s             )    * ctx->pb.phiDer(u[k][j+1][i].c);
          MatSetValuesStencil(J,1,&row,3,cols,vals,ADD_VALUES);
        }
        //z direction
        if (k == 0) {
          cols[0].i=i; cols[0].j=j; cols[0].k=k;
          cols[1].i=i; cols[1].j=j; cols[1].k=k+1;
          vals[0] = factordDX2*0.5*(u[k+1][j][i].s-3.*u[k][j][i].s+2.*ctx->pb.sExt) * ctx->pb.phiDer(u[k  ][j][i].c);
          vals[1] = factordDX2*0.5*(u[k+1][j][i].s-   u[k][j][i].s                ) * ctx->pb.phiDer(u[k+1][j][i].c);
          ierr = MatSetValuesStencil(P,1,&row,2,cols,vals,ADD_VALUES); CHKERRQ(ierr);
        } else if (k == ctx->daInfo.mz-1) {
          cols[0].i=i; cols[0].j=j; cols[0].k=k-1;
          cols[1].i=i; cols[1].j=j; cols[1].k=k;
          vals[0] = factordDX2*0.5*(u[k-1][j][i].s - u[k][j][i].s) * ctx->pb.phiDer(u[k-1][j][i].c);
          vals[1] = factordDX2*0.5*(u[k-1][j][i].s - u[k][j][i].s) * ctx->pb.phiDer(u[k  ][j][i].c);
          MatSetValuesStencil(J,1,&row,2,cols,vals,ADD_VALUES);
        } else {
          cols[0].i=i; cols[0].j=j; cols[0].k=k-1;
          cols[1].i=i; cols[1].j=j; cols[1].k=k;
          cols[2].i=i; cols[2].j=j; cols[2].k=k+1;
          vals[0] = factordDX2*0.5*(u[k-1][j][i].s -    u[k][j][i].s                ) * ctx->pb.phiDer(u[k-1][j][i].c);
          vals[1] = factordDX2*0.5*(u[k-1][j][i].s - 2.*u[k][j][i].s +u[k+1][j][i].s) * ctx->pb.phiDer(u[k  ][j][i].c);
          vals[2] = factordDX2*0.5*(u[k+1][j][i].s -    u[k][j][i].s                ) * ctx->pb.phiDer(u[k+1][j][i].c);
          MatSetValuesStencil(J,1,&row,3,cols,vals,ADD_VALUES);
        }
      }

  row.c=var::c;
  for (int i=0; i<3; ++i)
    cols[i].c=var::s;

  //JacCS block
  for (PetscInt k=ctx->daInfo.zs; k<ctx->daInfo.zs+ctx->daInfo.zm; k++)
    for (PetscInt j=ctx->daInfo.ys; j<ctx->daInfo.ys+ctx->daInfo.ym; j++)
      for (PetscInt i=ctx->daInfo.xs; i<ctx->daInfo.xs+ctx->daInfo.xm; i++){
        row.i = i; row.j=j; row.k=k;
        cols[0].i=i; cols[0].j=j; cols[0].k=k;
        vals[0] = dtFactor*(-Ac)*u[k][j][i].c*phi[k][j][i];
        ierr = MatSetValuesStencil(P,1,&row,1,cols,vals,ADD_VALUES); CHKERRQ(ierr);
      }

  for (int i=0; i<3; ++i)
    cols[i].c=var::c;

  //JacCC block
  for (PetscInt k=ctx->daInfo.zs; k<ctx->daInfo.zs+ctx->daInfo.zm; k++)
    for (PetscInt j=ctx->daInfo.ys; j<ctx->daInfo.ys+ctx->daInfo.ym; j++)
      for (PetscInt i=ctx->daInfo.xs; i<ctx->daInfo.xs+ctx->daInfo.xm; i++){
        row.i = i; row.j=j; row.k=k;
        cols[0].i=i; cols[0].j=j; cols[0].k=k;
        vals[0] = 1. + dtFactor*(-Ac)*u[k][j][i].s*(phi[k][j][i]+u[k][j][i].c*ctx->pb.phiDer(u[k][j][i].c));
        ierr = MatSetValuesStencil(P,1,&row,1,cols,vals,ADD_VALUES); CHKERRQ(ierr);
      }

  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  //Ripristino vettori della U e della PHI
  ierr = DMDAVecRestoreArray(ctx->daField[var::c],ctx->PHIloc,&phi); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(ctx->daAll,ctx->Uloc,&u); CHKERRQ(ierr);

  return ierr;

}

void formF3d(data_type ***u, PetscScalar ***phi, data_type ***f, PetscScalar dtFactor, AppContext* ctx){
  const PetscScalar As = ctx->pb.a/ctx->pb.mc;
  const PetscScalar Ac = ctx->pb.a/ctx->pb.ms;
  const PetscScalar dOverDX2 = ctx->pb.d / (ctx->dx*ctx->dx);

  for (PetscInt k=ctx->daInfo.zs; k<ctx->daInfo.zs+ctx->daInfo.zm; k++)
    for (PetscInt j=ctx->daInfo.ys; j<ctx->daInfo.ys+ctx->daInfo.ym; j++)
      for (PetscInt i=ctx->daInfo.xs; i<ctx->daInfo.xs+ctx->daInfo.xm; i++){
        // s component
        const PetscScalar phicsc = u[k][j][i].s * u[k][j][i].c * phi[k][j][i];
        f[k][j][i].s=phi[k][j][i]*u[k][j][i].s - dtFactor * As * phicsc;

        //x direction
        if (i<ctx->daInfo.mx-1){//homog. Neumann east
          const PetscScalar phiMed = 0.5*(phi[k][j][i]+phi[k][j][i+1]);
          f[k][j][i].s += dtFactor * dOverDX2 * phiMed * (u[k][j][i+1].s-u[k][j][i].s);
        }

        if (i>0){
          const PetscScalar phiMed = 0.5*(phi[k][j][i]+phi[k][j][i-1]);
          f[k][j][i].s -= dtFactor * dOverDX2 * phiMed * (u[k][j][i].s-u[k][j][i-1].s);
        }
        else //Dirichlet condition west
          f[k][j][i].s -= dtFactor * dOverDX2 * phi[k][j][i] * (u[k][j][i].s-ctx->pb.sExt);

        //y direction
        if (j<ctx->daInfo.my-1){//homog. Neumann north
          const PetscScalar phiMed = 0.5*(phi[k][j][i]+phi[k][j+1][i]);
          f[k][j][i].s += dtFactor * dOverDX2 * phiMed * (u[k][j+1][i].s-u[k][j][i].s);
        }

        if (j>0){
          const PetscScalar phiMed = 0.5*(phi[k][j][i]+phi[k][j-1][i]);
          f[k][j][i].s -= dtFactor * dOverDX2 * phiMed * (u[k][j][i].s-u[k][j-1][i].s);
        }
        else //Dirichlet condition south
          f[k][j][i].s -= dtFactor * dOverDX2 * phi[k][j][i] * (u[k][j][i].s-ctx->pb.sExt);

        //z direction
        if (k<ctx->daInfo.mz-1){//homog. Neumann north
          const PetscScalar phiMed = 0.5*(phi[k][j][i]+phi[k+1][j][i]);
          f[k][j][i].s += dtFactor * dOverDX2 * phiMed * (u[k+1][j][i].s-u[k][j][i].s);
        }

        if (k>0){
          const PetscScalar phiMed = 0.5*(phi[k][j][i]+phi[k-1][j][i]);
          f[k][j][i].s -= dtFactor * dOverDX2 * phiMed * (u[k][j][i].s-u[k-1][j][i].s);
        }
        else //Dirichlet condition south
          f[k][j][i].s -= dtFactor * dOverDX2 * phi[k][j][i] * (u[k][j][i].s-ctx->pb.sExt);

        // c component
        f[k][j][i].c = u[k][j][i].c - dtFactor * Ac * phicsc;
      }
}

PetscErrorCode FormRHS3d(Vec F0,void *_ctx){
  PetscErrorCode ierr;
  AppContext * ctx = (AppContext *) _ctx;
  data_type ***u0, ***f0;
  PetscScalar ***phi0;

  ierr = DMDAVecGetArray(ctx->daAll,F0,&f0); CHKERRQ(ierr);

  //Vettori locali con la U=(s,c)
  ierr = DMGlobalToLocalBegin(ctx->daAll,ctx->U0,INSERT_VALUES,ctx->Uloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (ctx->daAll,ctx->U0,INSERT_VALUES,ctx->Uloc); CHKERRQ(ierr);

  ierr = DMDAVecGetArray(ctx->daAll,ctx->Uloc,&u0); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(ctx->daField[var::c],ctx->PHIloc,&phi0); CHKERRQ(ierr);

  //Calcolo di phi
  for (PetscInt k=ctx->daInfo.gzs; k<ctx->daInfo.gzs+ctx->daInfo.gzm; k++)
    for (PetscInt j=ctx->daInfo.gys; j<ctx->daInfo.gys+ctx->daInfo.gym; j++)
      for (PetscInt i=ctx->daInfo.gxs; i<ctx->daInfo.gxs+ctx->daInfo.gxm; i++)
        phi0[k][j][i] = ctx->pb.phi(u0[k][j][i].c);

  //calcolo della F
  formF3d(u0,phi0,f0,ctx->dt*ctx->theta,ctx);

  //Ripristino vettori della U e della PHI
  ierr = DMDAVecRestoreArray(ctx->daField[var::c],ctx->PHIloc,&phi0); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(ctx->daAll,ctx->Uloc,&u0); CHKERRQ(ierr);

  //Ripristino vettori della F
  ierr = DMDAVecRestoreArray(ctx->daAll,F0,&f0); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode FormFunction3d(SNES snes,Vec U,Vec F,void *_ctx){
  PetscErrorCode ierr;
  AppContext * ctx = (AppContext *) _ctx;
  data_type ***u, ***f;
  PetscScalar ***phi;

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
  for (PetscInt k=ctx->daInfo.gzs; k<ctx->daInfo.gzs+ctx->daInfo.gzm; k++)
    for (PetscInt j=ctx->daInfo.gys; j<ctx->daInfo.gys+ctx->daInfo.gym; j++)
      for (PetscInt i=ctx->daInfo.gxs; i<ctx->daInfo.gxs+ctx->daInfo.gxm; i++)
        phi[k][j][i] = ctx->pb.phi(u[k][j][i].c);

  //calcolo della F
  formF3d(u,phi,f,-ctx->dt*(1.-ctx->theta),ctx);

  //Ripristino vettori della U e della PHI
  ierr = DMDAVecRestoreArray(ctx->daField[var::c],ctx->PHIloc,&phi); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(ctx->daAll,ctx->Uloc,&u); CHKERRQ(ierr);

  //Ripristino vettori della F
  ierr = DMDAVecRestoreArray(ctx->daAll,F,&f); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode setU03d(Vec U,void *_ctx){
  PetscErrorCode ierr;
  AppContext * ctx = (AppContext *) _ctx;

  ierr = VecISSet(ctx->U0,ctx->is[var::s],ctx->pb.s0); CHKERRQ(ierr);
  ierr = VecISSet(ctx->U0,ctx->is[var::c],ctx->pb.c0); CHKERRQ(ierr);

  return ierr;
}
