#include "levelSet.h"

#include <petscdmda.h>
#include <bitset>

PetscScalar Phi1_(DMDACoor3d p);

template <typename T>
inline int SGN(T x)
{
    return x>=0 ? 1 : -1;
}
inline int SIGN( double x)
{
    return (x-1e-14 > 0) - (x+1e-14 < 0);
}

//TODO: da migliorare
inline int powI(int base, int exp)
{
    int result=1;
    for(int i=0;i<exp;++i)
      result*=base;
    return result;
}

typedef struct{
  PetscScalar x,y,z;
  unsigned int n;
} shiftedStencil;

PetscErrorCode findBoundaryPoints(
  AppContext &ctx,
  PetscInt level,
  //PetscInt xs, PetscInt ys, PetscInt zs,
  //PetscInt xm, PetscInt ym, PetscInt zm,
  DMDACoor3d ***coords,
  PetscScalar ***phi,
  DMDACoor3d ***N,
  DMDACoor3d ***B);

PetscErrorCode setGhostStencil(AppContext & ctx, PetscInt kg,
  PetscScalar ***phi, PetscScalar ***nodetype,
  DMDACoor3d ***P,
  double& xC, double& yC, double& zC,
  int stencil[], double coeffsD[], double coeffs_dx[], double coeffs_dy[], double coeffs_dz[],
  double& nxb, double& nyb, double& nzb,
  int upwind,
  std::vector<shiftedStencil> &critici);


PetscErrorCode setPhi(AppContext &ctx)
/**************************************************************************/
{
  PetscErrorCode ierr;
  PetscPrintf(PETSC_COMM_WORLD,"%d] Setting phi values... \n",ctx.rank);

  DMDACoor3d ***P;
  ierr = DMDAVecGetArrayRead(ctx.daCoord,ctx.coordsLocal,&P); CHKERRQ(ierr);

  PetscScalar ***phi;
  ierr = DMCreateGlobalVector(ctx.daField[var::s], &ctx.Phi); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) ctx.Phi, "Phi");
  ierr = DMDAVecGetArrayWrite(ctx.daField[var::s], ctx.Phi, &phi); CHKERRQ(ierr);

  for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++)
    for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++)
      for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++)
        phi[k][j][i]=Phi1_(P[k][j][i]);

  ierr = DMDAVecRestoreArrayWrite(ctx.daField[var::s],ctx.Phi,&phi); CHKERRQ(ierr);

  //for(PetscInt level=0;level<levels;++level){
      //ierr = DMCreateGlobalVector(CartesianGrid3D_p[level], &Phi_p[level]); CHKERRQ(ierr);
      //ierr = DMCreateLocalVector(CartesianGrid3D_p[level], &local_Phi_p[level]); CHKERRQ(ierr);
      //ierr = DMDAVecGetArrayWrite(CartesianGrid3D_p[level], Phi_p[level], &phi); CHKERRQ(ierr);
      //ierr = DMDAGetCorners(CartesianGrid3D_p[level], &ys, &xs, &zs, &ym, &xm, &zm); CHKERRQ(ierr);

  //for (PetscInt k = zs; k < zs + zm; ++k)
      //for (PetscInt j = xs; j < xs + xm; ++j)
          //for (PetscInt i = ys; i < ys + ym; ++i)
              //phi[k][j][i]=Phi1_(x_p[level][j],y_p[level][i],z_p[level][k]);

  //ierr = DMDAVecRestoreArrayWrite(CartesianGrid3D_p[level],Phi_p[level],&phi);                CHKERRQ(ierr);
  //}

  ierr = DMDAVecRestoreArrayRead(ctx.daCoord,ctx.coordsLocal,&P); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(ctx.daField[var::s], &ctx.local_Phi); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(ctx.daField[var::s], ctx.Phi, INSERT_VALUES, ctx.local_Phi); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (ctx.daField[var::s], ctx.Phi, INSERT_VALUES, ctx.local_Phi); CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"Setting phi values... DONE!\n");
  return ierr;
}

PetscErrorCode setNormals(AppContext &ctx)
/**************************************************************************/
{
  PetscErrorCode ierr;
  PetscPrintf(PETSC_COMM_WORLD,"Setting normals...\n");
  PetscInt xd, yd, zd;//, xs, ys, zs, xm, ym, zm;
  PetscScalar PhiX1, PhiX2, PhiY1, PhiY2, PhiZ1, PhiZ2, nx_temp, ny_temp, nz_temp, module;
  PetscScalar ***phi;
  DMDACoor3d ***N;

  ierr = DMCreateGlobalVector(ctx.daCoord, &ctx.NORMALS); CHKERRQ(ierr);

  ierr = DMDAVecGetArrayRead(ctx.daField[var::s], ctx.local_Phi, &phi); CHKERRQ(ierr);
  ierr = DMDAVecGetArrayWrite(ctx.daCoord, ctx.NORMALS, &N); CHKERRQ(ierr);

  for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++)
    for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++)
      for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
        xd=yd=zd=2;

        PhiX1 = i>0 ? phi[k][j][i-1] : phi[k][j][i];
        xd = i>0 ? xd : 1;
        PhiX2 = i<ctx.nx ? phi[k][j][i+1] : phi[k][j][i];
        xd = i<ctx.nx ? xd : 1;

        PhiY1 = j>0 ? phi[k][j-1][i] : phi[k][j][i];
        yd = j>0 ? yd : 1;
        PhiY2 = j<ctx.ny ? phi[k][j+1][i] : phi[k][j][i];
        yd = i<ctx.ny ? yd : 1;

        PhiZ1 = k>0 ? phi[k-1][j][i] : phi[k][j][i];
        zd = k>0 ? zd : 1;
        PhiZ2 = k<ctx.nz ? phi[k+1][j][i] : phi[k][j][i];
        zd = k<ctx.nz ? zd : 1;

        nx_temp = (PhiX2-PhiX1)/(xd*ctx.dx);
        ny_temp = (PhiY2-PhiY1)/(yd*ctx.dy);
        nz_temp = (PhiZ2-PhiZ1)/(zd*ctx.dz);
        module=std::sqrt(nx_temp*nx_temp+ny_temp*ny_temp+nz_temp*nz_temp);

        N[k][j][i].x=nx_temp/module;
        N[k][j][i].y=ny_temp/module;
        N[k][j][i].z=nz_temp/module;

        //PetscPrintf(PETSC_COMM_WORLD,"(%d,%d,%d) N=(%f,%f,%f)\n",i,j,k,N[k][j][i].x,N[k][j][i].y,N[k][j][i].z);

      }

  ierr = DMDAVecRestoreArrayRead(ctx.daField[var::s], ctx.local_Phi, &phi); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayWrite(ctx.daCoord, ctx.NORMALS, &N); CHKERRQ(ierr);

     /////////////////////////////////////////////////////////////////////////////////////////////////
    
    //VecDuplicate(Nx,&Nx_p[0]);
    //VecDuplicate(Ny,&Ny_p[0]);
    //VecDuplicate(Nz,&Nz_p[0]);
    //VecCopy(Nx,Nx_p[0]);
    //VecCopy(Ny,Ny_p[0]);
    //VecCopy(Nz,Nz_p[0]);

    //for(PetscInt level=1;level<levels;++level){
        //PetscScalar ***nx_fine, ***ny_fine, ***nz_fine;
        //ierr = DMCreateGlobalVector(CartesianGrid3D_p[level], &Nx_p[level]);
        //ierr = DMCreateGlobalVector(CartesianGrid3D_p[level], &Ny_p[level]);
        //ierr = DMCreateGlobalVector(CartesianGrid3D_p[level], &Nz_p[level]);

        //ierr = DMDAVecGetArrayWrite(CartesianGrid3D_p[level], Nx_p[level], &nx);
        //ierr = DMDAVecGetArrayWrite(CartesianGrid3D_p[level], Ny_p[level], &ny);
        //ierr = DMDAVecGetArrayWrite(CartesianGrid3D_p[level], Nz_p[level], &nz);

        //ierr = DMDAVecGetArrayWrite(CartesianGrid3D_p[level-1], Nx_p[level-1], &nx_fine);
        //ierr = DMDAVecGetArrayWrite(CartesianGrid3D_p[level-1], Ny_p[level-1], &ny_fine);
        //ierr = DMDAVecGetArrayWrite(CartesianGrid3D_p[level-1], Nz_p[level-1], &nz_fine);

        //ierr = RestrictionInjection(level, nx, nx_fine);
        //ierr = RestrictionInjection(level, ny, ny_fine);
        //ierr = RestrictionInjection(level, nz, nz_fine);

        //ierr = DMDAVecRestoreArrayWrite(CartesianGrid3D_p[level],Nx_p[level],&nx);
        //ierr = DMDAVecRestoreArrayWrite(CartesianGrid3D_p[level],Ny_p[level],&ny);
        //ierr = DMDAVecRestoreArrayWrite(CartesianGrid3D_p[level],Nz_p[level],&nz);

        //ierr = DMDAVecRestoreArrayWrite(CartesianGrid3D_p[level-1], Nx_p[level-1], &nx_fine);
        //ierr = DMDAVecRestoreArrayWrite(CartesianGrid3D_p[level-1], Ny_p[level-1], &ny_fine);
        //ierr = DMDAVecRestoreArrayWrite(CartesianGrid3D_p[level-1], Nz_p[level-1], &nz_fine);

        ///*
        //ierr = DMGlobalToLocalBegin(CartesianGrid3D_p[level], Phi_p[level], INSERT_VALUES, local_Phi_p[level]);
        //ierr = DMGlobalToLocalEnd(CartesianGrid3D_p[level], Phi_p[level], INSERT_VALUES, local_Phi_p[level]);

        //ierr = DMDAVecGetArrayRead(CartesianGrid3D_p[level], local_Phi_p[level], &phi);
        //ierr = DMDAVecGetArrayWrite(CartesianGrid3D_p[level], Nx_p[level], &nx);
        //ierr = DMDAVecGetArrayWrite(CartesianGrid3D_p[level], Ny_p[level], &ny);
        //ierr = DMDAVecGetArrayWrite(CartesianGrid3D_p[level], Nz_p[level], &nz);
        //ierr = DMDAGetCorners(CartesianGrid3D_p[level], &ys, &xs, &zs, &ym, &xm, &zm);
    
        //PetscInt nn1_l = (nn1-1)/pow(2,level)+1;
        //PetscInt nn2_l = (nn2-1)/pow(2,level)+1;
        //PetscInt nn3_l = (nn3-1)/pow(2,level)+1;
        //PetscScalar dx_l=dx*pow(2,level);
        //PetscScalar dy_l=dy*pow(2,level);
        //PetscScalar dz_l=dz*pow(2,level);
        //for (PetscInt k = zs; k < zs + zm; ++k)
            //for (PetscInt j = xs; j < xs + xm; ++j)
                //for (PetscInt i = ys; i < ys + ym; ++i){
                    //xd=yd=zd=2;
                    //PhiX1 = j>0 ? phi[k][j-1][i] : phi[k][j][i];
                    //xd = j>0 ? xd : 1;
                    //PhiX2 = j<nn2_l-1 ? phi[k][j+1][i] : phi[k][j][i];
                    //xd = j<nn2_l-1 ? xd : 1;
                    //PhiY1 = i>0 ? phi[k][j][i-1] : phi[k][j][i];
                    //yd = i>0 ? yd : 1;
                    //PhiY2 = i<nn1_l-1 ? phi[k][j][i+1] : phi[k][j][i];
                    //yd = i<nn1_l-1 ? yd : 1;
                    //PhiZ1 = k>0 ? phi[k-1][j][i] : phi[k][j][i];
                    //zd = k>0 ? zd : 1;
                    //PhiZ2 = k<nn3_l-1 ? phi[k+1][j][i] : phi[k][j][i];
                    //zd = k<nn3_l-1 ? zd : 1;
                    //nx_temp = (PhiX2-PhiX1)/(xd*dx_l);
                    //ny_temp = (PhiY2-PhiY1)/(yd*dy_l);
                    //nz_temp = (PhiZ2-PhiZ1)/(zd*dz_l);
                    //module=sqrt(nx_temp*nx_temp+ny_temp*ny_temp+nz_temp*nz_temp);
                    //nx[k][j][i]=nx_temp/module;
                    //ny[k][j][i]=ny_temp/module;
                    //nz[k][j][i]=nz_temp/module;
                //}

        //ierr = DMDAVecRestoreArrayRead(CartesianGrid3D_p[level], local_Phi_p[level], &phi);
        //ierr = DMDAVecRestoreArrayWrite(CartesianGrid3D_p[level],Nx_p[level],&nx);
        //ierr = DMDAVecRestoreArrayWrite(CartesianGrid3D_p[level],Ny_p[level],&ny);
        //ierr = DMDAVecRestoreArrayWrite(CartesianGrid3D_p[level],Nz_p[level],&nz);
        //*/
    //}

    PetscPrintf(PETSC_COMM_WORLD, "Setting normals...DONE!\n");
    return ierr;
}

PetscErrorCode setBoundaryPoints(AppContext &ctx)
/**************************************************************************/
{
  PetscErrorCode ierr;
  PetscPrintf(PETSC_COMM_WORLD,"set boundary points ...\n");

  PetscScalar ***phi;
  DMDACoor3d ***N, ***B, ***P;

  ierr = DMCreateGlobalVector(ctx.daCoord, &ctx.BOUNDARY); CHKERRQ(ierr);

  ierr = DMDAVecGetArrayRead(ctx.daField[var::s], ctx.local_Phi, &phi); CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(ctx.daCoord, ctx.NORMALS, &N); CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(ctx.daCoord,ctx.coordsLocal,&P); CHKERRQ(ierr);
  ierr = DMDAVecGetArrayWrite(ctx.daCoord, ctx.BOUNDARY, &B); CHKERRQ(ierr);

  ierr = findBoundaryPoints(ctx,
                            0,//ctx.daInfo.xs,ctx.daInfo.ys,ctx.daInfo.zs,ctx.daInfo.xm,ctx.daInfo.ym,ctx.daInfo.zm,
                            P,phi,N,B); CHKERRQ(ierr);

  ierr = DMDAVecRestoreArrayRead(ctx.daField[var::s], ctx.local_Phi, &phi); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(ctx.daCoord, ctx.NORMALS, &N); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(ctx.daCoord,ctx.coordsLocal,&P); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayWrite(ctx.daCoord, ctx.BOUNDARY, &B); CHKERRQ(ierr);

    ///////////////////////////////////////////////////////////////////////////
    //VecDuplicate(Xb,&Xb_p[0]);
    //VecDuplicate(Yb,&Yb_p[0]);
    //VecDuplicate(Zb,&Zb_p[0]);
    //VecCopy(Xb,Xb_p[0]);
    //VecCopy(Yb,Yb_p[0]);
    //VecCopy(Zb,Zb_p[0]);

    //for(PetscInt level=1; level<levels;++level){
        //ierr = DMCreateGlobalVector(CartesianGrid3D_p[level], &Xb_p[level]);
        //ierr = DMCreateGlobalVector(CartesianGrid3D_p[level], &Yb_p[level]);
        //ierr = DMCreateGlobalVector(CartesianGrid3D_p[level], &Zb_p[level]);

        //ierr = DMGlobalToLocalBegin(CartesianGrid3D_p[level], Phi_p[level], INSERT_VALUES, local_Phi_p[level]);
        //ierr = DMGlobalToLocalEnd(CartesianGrid3D_p[level], Phi_p[level], INSERT_VALUES, local_Phi_p[level]);

        //ierr = DMDAVecGetArrayRead(CartesianGrid3D_p[level], local_Phi_p[level], &phi);
        //ierr = DMDAVecGetArrayRead(CartesianGrid3D_p[level], Nx_p[level], &nx);
        //ierr = DMDAVecGetArrayRead(CartesianGrid3D_p[level], Ny_p[level], &ny);
        //ierr = DMDAVecGetArrayRead(CartesianGrid3D_p[level], Nz_p[level], &nz);
        //ierr = DMDAVecGetArrayWrite(CartesianGrid3D_p[level], Xb_p[level], &xb);
        //ierr = DMDAVecGetArrayWrite(CartesianGrid3D_p[level], Yb_p[level], &yb);
        //ierr = DMDAVecGetArrayWrite(CartesianGrid3D_p[level], Zb_p[level], &zb);
        //ierr = DMDAGetCorners(CartesianGrid3D_p[level], &ys, &xs, &zs, &ym, &xm, &zm);

        //findBoundaryPoints(level,xs,ys,zs,xm,ym,zm,x_p[level],y_p[level],z_p[level],phi,nx,ny,nz,xb,yb,zb);

        //ierr = DMDAVecRestoreArrayWrite(CartesianGrid3D_p[level],Xb_p[level],&xb);
        //ierr = DMDAVecRestoreArrayWrite(CartesianGrid3D_p[level],Yb_p[level],&yb);
        //ierr = DMDAVecRestoreArrayWrite(CartesianGrid3D_p[level],Zb_p[level],&zb);
        //ierr = DMDAVecRestoreArrayRead(CartesianGrid3D_p[level], local_Phi_p[level], &phi);
        //ierr = DMDAVecRestoreArrayRead(CartesianGrid3D_p[level], Nx_p[level], &nx);
        //ierr = DMDAVecRestoreArrayRead(CartesianGrid3D_p[level], Ny_p[level], &ny);
        //ierr = DMDAVecRestoreArrayRead(CartesianGrid3D_p[level], Nz_p[level], &nz);

    //}
    PetscPrintf(PETSC_COMM_WORLD,"set boundary points ...DONE\n");
    return ierr;
}

PetscErrorCode findBoundaryPoints(
  AppContext &ctx,
  PetscInt level,
  //PetscInt xs, PetscInt ys, PetscInt zs,
  //PetscInt xm, PetscInt ym, PetscInt zm,
  DMDACoor3d ***coords,
  PetscScalar ***phi,
  DMDACoor3d ***N,
  DMDACoor3d ***B)
/**************************************************************************/
{
  PetscErrorCode ierr=0;

  const PetscInt s=3;
  const PetscInt nn1_l = (ctx.nx)/pow(2,level)+1;
  const PetscInt nn2_l = (ctx.ny)/pow(2,level)+1;
  const PetscInt nn3_l = (ctx.nz)/pow(2,level)+1;
  const PetscScalar dx_l = ctx.dx*pow(2,level);
  const PetscScalar dy_l = ctx.dy*pow(2,level);
  const PetscScalar dz_l = ctx.dz*pow(2,level);
  PetscScalar weights_x[s], weights_y[s], weights_z[s],
              weights_dx[s], weights_dy[s], weights_dz[s],
              Phi_dxB, Phi_dyB, Phi_dzB, PhiI(NAN),
              thtx, thty, thtz,
              xI(NAN), yI(NAN), zI(NAN);
  const PetscScalar diag_cell=sqrt(dx_l*dx_l+dy_l*dy_l+dz_l*dz_l);
  PetscScalar nx_temp, ny_temp, nz_temp;

  for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++){
    for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++){
      for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){

        PetscScalar R=phi[k][j][i];
        nx_temp=N[k][j][i].x, ny_temp=N[k][j][i].y, nz_temp=N[k][j][i].z;        
        PetscInt sx=SGN(-nx_temp*R),  sy=SGN(-ny_temp*R),    sz=SGN(-nz_temp*R);

        //PetscPrintf(PETSC_COMM_WORLD,"(%d,%d,%d) N=(%f,%f,%f) R=%f s=(%d,%d,%d)\n",i,j,k,N[k][j][i].x,N[k][j][i].y,N[k][j][i].z,R,sx,sy,sz);

        if((i<s-1 && sx<0) || (i>nn1_l-s && sx>0) || (j<s-1 && sy<0) || (j>nn2_l-s && sy>0) || (k<s-1 && sz<0) || (k>nn3_l-s && sz>0)){
          SETERRQ(PETSC_COMM_SELF,1,"findBoundaryPoints cannot be run as the stencil goes outside the domain.");
          //cerr << "ERROR: findBoundaryPoints cannot be run as the stencil goes outside the domain." << endl;
          continue;
        }

        PetscScalar xP = coords[k][j][i].x;
        PetscScalar yP = coords[k][j][i].y;
        PetscScalar zP = coords[k][j][i].z;

        PetscScalar xA=xP;
        PetscScalar yA=yP;
        PetscScalar zA=zP;
        PetscScalar xB=xP;
        PetscScalar yB=yP;
        PetscScalar zB=zP;
        PetscScalar PhiA=phi[k][j][i]+1.e-16;
        PetscScalar PhiB=phi[k][j][i]+1.e-16;

        const PetscScalar cc=0.25;

        while(PhiA*PhiB>0 && !isnan(PhiB)){
          xA=xB;
          yA=yB;
          zA=zB;
          PhiA=PhiB;

          xB = xA - SGN(R)*cc*diag_cell*nx_temp;
          yB = yA - SGN(R)*cc*diag_cell*ny_temp;
          zB = zA - SGN(R)*cc*diag_cell*nz_temp;

          thtx=sx*(xB-xP)/(s-1)/dx_l;
          thty=sy*(yB-yP)/(s-1)/dy_l;
          thtz=sz*(zB-zP)/(s-1)/dz_l;

          if(s==2){
            weights_x[0] =1-thtx  , weights_x[1] =thtx;
            weights_y[0] =1-thty  , weights_y[1] =thty;
            weights_z[0] =1-thtz  , weights_z[1] =thtz;
            weights_dx[0]=-1./dx_l, weights_dx[1]=1./dx_l;
            weights_dy[0]=-1./dy_l, weights_dy[1]=1./dy_l;
            weights_dz[0]=-1./dz_l, weights_dz[1]=1./dz_l;
          }
          else if(s==3){
            weights_x[0] =(1-2*thtx)*(1-thtx)    , weights_x[1] =4*thtx*(1-thtx)          , weights_x[2] =thtx*(2*thtx-1);
            weights_y[0] =(1-2*thty)*(1-thty)    , weights_y[1] =4*thty*(1-thty)          , weights_y[2] =thty*(2*thty-1);
            weights_z[0] =(1-2*thtz)*(1-thtz)    , weights_z[1] =4*thtz*(1-thtz)          , weights_z[2] =thtz*(2*thtz-1);
            weights_dx[0]=(-1.+(2*thtx-0.5))/dx_l, weights_dx[1]=(1.-2.*(2*thtx-0.5))/dx_l, weights_dx[2]=(2*thtx-0.5)/dx_l;
            weights_dy[0]=(-1.+(2*thty-0.5))/dy_l, weights_dy[1]=(1.-2.*(2*thty-0.5))/dy_l, weights_dy[2]=(2*thty-0.5)/dy_l;
            weights_dz[0]=(-1.+(2*thtz-0.5))/dz_l, weights_dz[1]=(1.-2.*(2*thtz-0.5))/dz_l, weights_dz[2]=(2*thtz-0.5)/dz_l;
          }
          else
            SETERRQ(PETSC_COMM_SELF,1,"Error: value of s is not supported.");
            //cerr << "Error: value of s is not supported." << endl;

          PhiB=Phi_dxB=Phi_dyB=Phi_dzB=0;
          for(int kk=0;kk<s;++kk){
            for(int jj=0;jj<s;++jj){
              for(int ii=0;ii<s;++ii){
                PetscScalar Phi_stencil=phi[k+sz*kk][j+sy*jj][i+sx*ii];
                PhiB+=Phi_stencil*weights_x[ii]*weights_y[jj]*weights_z[kk];
                Phi_dxB+=Phi_stencil*sx*weights_dx[ii]*weights_y[jj] *weights_z[kk];
                Phi_dyB+=Phi_stencil*sy*weights_x[ii] *weights_dy[jj]*weights_z[kk];
                Phi_dzB+=Phi_stencil*sz*weights_x[ii] *weights_y[jj] *weights_dz[kk];
              }
            }
          }

          PetscScalar modlB=std::sqrt(Phi_dxB*Phi_dxB+Phi_dyB*Phi_dyB+Phi_dzB*Phi_dzB);
          nx_temp=Phi_dxB/modlB;
          ny_temp=Phi_dyB/modlB;
          nz_temp=Phi_dzB/modlB;
        }

        PetscScalar distAB=std::sqrt((xB-xA)*(xB-xA)+(yB-yA)*(yB-yA)+(zB-zA)*(zB-zA));
        PetscScalar accuracy_bisection=0.0001*dx_l;
        PetscInt niter=(int)(ceil(log(distAB/accuracy_bisection)/log(2))+1);

        for(PetscInt iter=0;iter<niter;++iter){
          xI=(xA+xB)/2;
          yI=(yA+yB)/2;
          zI=(zA+zB)/2;

          thtx=sx*(xI-xP)/(s-1)/dx_l;
          thty=sy*(yI-yP)/(s-1)/dy_l;
          thtz=sz*(zI-zP)/(s-1)/dz_l;

          if(s==2){
              weights_x[0]=1-thtx, weights_x[1]=thtx;
              weights_y[0]=1-thty, weights_y[1]=thty;
              weights_z[0]=1-thtz, weights_z[1]=thtz;
          }
          else if(s==3){
              weights_x[0]=(1-2*thtx)*(1-thtx), weights_x[1]=4*thtx*(1-thtx), weights_x[2]=thtx*(2*thtx-1);
              weights_y[0]=(1-2*thty)*(1-thty), weights_y[1]=4*thty*(1-thty), weights_y[2]=thty*(2*thty-1);
              weights_z[0]=(1-2*thtz)*(1-thtz), weights_z[1]=4*thtz*(1-thtz), weights_z[2]=thtz*(2*thtz-1);
          }
          else
            SETERRQ(PETSC_COMM_SELF,1,"Error: value of s is not supported.");
            //cerr << "Error: value of s is not supported." << endl;

          PhiI=0;
          for(int kk=0;kk<s;++kk)
            for(int jj=0;jj<s;++jj)
              for(int ii=0;ii<s;++ii)
                PhiI+=phi[k+sz*kk][j+sy*jj][i+sx*ii]*weights_x[ii]*weights_y[jj]*weights_z[kk];

          if(PhiA*PhiI>0){
              xA=xI;
              yA=yI;
              zA=zI;
              PhiA=PhiI;
          } else {
              xB=xI;
              yB=yI;
              zB=zI;
              PhiB=PhiI;
          }
        }

        B[k][j][i].x=xI;
        B[k][j][i].y=yI;
        B[k][j][i].z=zI;
      }
    }
  }
  return ierr;
}

void scriviListaPunti(std::vector<shiftedStencil> lista, const char * basename){
  FILE *file;
  char  filename[256];
  sprintf(filename, "%s.vtu",basename);
  file = fopen(filename, "w");
  fprintf(file, "<VTKFile type=\"UnstructuredGrid\" byte_order=\"LittleEndian\">\n");

  fprintf(file, "<UnstructuredGrid>\n");
  fprintf(file, "  <Piece  NumberOfPoints=\"%d\" NumberOfCells=\"0\">\n",
                lista.size());
  fprintf(file, "    <PointData Scalars=\"nShifts\"> \n");
  fprintf(file, "      <DataArray type=\"UInt8\" Name=\"nShifts\">\n        ");
  for (auto it = lista.begin(); it != lista.end(); it++) {
    fprintf(file, "        %d\n", it->n);
  }
  fprintf(file, "      </DataArray> \n");
  fprintf(file, "    </PointData> \n");
  fprintf(file, "    <CellData> \n");
  fprintf(file, "    </CellData> \n");

  fprintf(file, "    <Points>\n");
  fprintf(file, "      <DataArray name=\"Position\" type=\"Float32\" NumberOfComponents=\"3\"  format=\"ascii\"> \n");
  for (auto it = lista.begin(); it != lista.end(); it++) {
    fprintf(file, "        %f %f %f\n", it->x, it->y, it->z);
  }
  fprintf(file, "      </DataArray> \n");
  fprintf(file, "    </Points>\n");

  fprintf(file, "    <Cells>\n");
  fprintf(file, "      <DataArray type=\"Int32\" Name=\"connectivity\">");
  fprintf(file, "      </DataArray> \n");

  fprintf(file, "      <DataArray type=\"Int32\" Name=\"offsets\">");
  fprintf(file, "      </DataArray> \n");

  fprintf(file, "      <DataArray type=\"UInt8\" Name=\"types\">\n        ");
  fprintf(file, "      </DataArray> \n");
  fprintf(file, "    </Cells>\n");

  fprintf(file, "   </Piece> \n");
  fprintf(file, "</UnstructuredGrid>\n");
  fprintf(file, "</VTKFile>\n");
  fclose(file);
}

PetscErrorCode setGhost(AppContext &ctx)
{
  std::vector<shiftedStencil> critici;
  critici.resize(0);

  PetscErrorCode ierr=0;
  PetscPrintf(PETSC_COMM_WORLD, "Setting ghost stencils ...\n");
  int stencil[27];
  double coeffsD[27], coeffs_dx[27], coeffs_dy[27], coeffs_dz[27],
         nxb, nyb, nzb;
  ghost current;
  ghost_Bdy current_Bdy;
    
  PetscScalar ***phi;
  DMDACoor3d ***B, ***P;
  PetscScalar ***nodetype;
    
  for(PetscInt level=-1;level<ctx.mgLevels;++level){ //level=-1 means that we create objects for the fine grid without using vectors of grids 
    PetscInt level_temp;
    PetscInt nn1_l=(ctx.nx)/pow(2,level)+1;
    PetscInt nn2_l=(ctx.ny)/pow(2,level)+1;
    PetscInt nn3_l=(ctx.nz)/pow(2,level)+1;
    if(level==-1){
        level_temp=0;
        nn1_l=ctx.nnx;
        nn2_l=ctx.nny;
        nn3_l=ctx.nnz;
        ierr = DMCreateGlobalVector(ctx.daField[var::s], &ctx.NODETYPE); CHKERRQ(ierr);
        ierr = DMCreateLocalVector(ctx.daField[var::s], &ctx.local_NodeType); CHKERRQ(ierr);

        //ierr = DMGlobalToLocalBegin(ctx.daField[var::s], Phi, INSERT_VALUES, local_Phi); CHKERRQ(ierr);
        //ierr = DMGlobalToLocalEnd(ctx.daField[var::s], Phi, INSERT_VALUES, local_Phi); CHKERRQ(ierr);
        ierr = DMDAVecGetArrayRead(ctx.daField[var::s], ctx.local_Phi, &phi); CHKERRQ(ierr);

        ierr = DMDAVecGetArrayRead(ctx.daCoord, ctx.coordsLocal, &P); CHKERRQ(ierr);

        ierr = DMDAVecGetArrayRead(ctx.daCoord, ctx.BOUNDARY, &B); CHKERRQ(ierr);
        ierr = DMDAVecGetArrayWrite(ctx.daField[var::s], ctx.NODETYPE, &nodetype); CHKERRQ(ierr);
    }
    else{
      SETERRQ(PETSC_COMM_SELF,1,"not yet implemented"); CHKERRQ(ierr);
        //level_temp=level;
        //ierr = DMCreateGlobalVector(CartesianGrid3D_p[level], &NodeType_p[level]);
        //ierr = DMCreateLocalVector(CartesianGrid3D_p[level], &local_NodeType_p[level]);

        //ierr = DMGlobalToLocalBegin(CartesianGrid3D_p[level], Phi_p[level], INSERT_VALUES, local_Phi_p[level]);
        //ierr = DMGlobalToLocalEnd(CartesianGrid3D_p[level], Phi_p[level], INSERT_VALUES, local_Phi_p[level]);       
        //ierr = DMDAVecGetArrayRead(CartesianGrid3D_p[level], local_Phi_p[level], &phi);
        //ierr = DMDAVecGetArrayRead(CartesianGrid3D_p[level], Xb_p[level], &xb);
        //ierr = DMDAVecGetArrayRead(CartesianGrid3D_p[level], Yb_p[level], &yb);
        //ierr = DMDAVecGetArrayRead(CartesianGrid3D_p[level], Zb_p[level], &zb);

        //ierr = DMDAVecGetArrayWrite(CartesianGrid3D_p[level], NodeType_p[level], &nodetype);
        //ierr = DMDAGetCorners(CartesianGrid3D_p[level], &ys, &xs, &zs, &ym, &xm, &zm);
    }

    for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++){
      for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++){
        for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
          if( (i==0 && phi[k][j][i+1]<0 ) || (i==nn1_l-1 && phi[k][j][i-1]<0) || (j==0 && phi[k][j+1][i]<0) || (j==nn2_l-1 && phi[k][j-1][i]<0) || (k==0 && phi[k+1][j][i]<0) || (k==nn3_l-1 && phi[k-1][j][i]<0) )
            nodetype[k][j][i]=N_GHOSTBDY; //Ghost.Bdy
          else if(i==0 || i==nn1_l-1 || j==0 || j==nn2_l-1 || k==0 || k==nn3_l-1 )
            nodetype[k][j][i]=N_INACTIVE; //inactive
          else if(phi[k][j][i]<0)
            nodetype[k][j][i]=N_INSIDE; //Inside
          else if( phi[k][j][i-1]<0 || phi[k][j][i+1]<0 || phi[k][j-1][i]<0 || phi[k][j+1][i]<0 || phi[k-1][j][i]<0 || phi[k+1][j][i]<0 )
            nodetype[k][j][i]=N_GHOSTPHI1; //Ghost.Phi1
          else 
            nodetype[k][j][i]=N_INACTIVE; //inactive
        }
      }
    }

    if(level==-1){
      ierr = DMDAVecRestoreArrayWrite(ctx.daField[var::s], ctx.NODETYPE, &nodetype); CHKERRQ(ierr);
      ierr = DMGlobalToLocalBegin(ctx.daField[var::s], ctx.NODETYPE, INSERT_VALUES, ctx.local_NodeType); CHKERRQ(ierr);
      ierr = DMGlobalToLocalEnd(ctx.daField[var::s], ctx.NODETYPE, INSERT_VALUES, ctx.local_NodeType); CHKERRQ(ierr);
      ierr = DMDAVecGetArrayWrite(ctx.daField[var::s], ctx.local_NodeType, &nodetype); CHKERRQ(ierr);
    }
    else{
      SETERRQ(PETSC_COMM_SELF,1,"not yet implemented"); CHKERRQ(ierr);
      //ierr = DMDAVecRestoreArrayWrite(CartesianGrid3D_p[level], NodeType_p[level], &nodetype);

      //ierr = DMGlobalToLocalBegin(CartesianGrid3D_p[level], NodeType_p[level], INSERT_VALUES, local_NodeType_p[level]);
      //ierr = DMGlobalToLocalEnd(CartesianGrid3D_p[level], NodeType_p[level], INSERT_VALUES, local_NodeType_p[level]);
      //ierr = DMDAVecGetArrayWrite(CartesianGrid3D_p[level], local_NodeType_p[level], &nodetype);
    }

    PetscPrintf(PETSC_COMM_WORLD,"level=%d\n",level);

    for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++){
      for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++){
        for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
          if(nodetype[k][j][i]==N_GHOSTPHI1){
            //PetscPrintf(PETSC_COMM_WORLD,"(%d,%d,%d) --> %f\n",i,j,k,nodetype[k][j][i]);

            PetscInt kg =  i + j * nn1_l + k * nn1_l * nn2_l;
            current.index=kg;
            current.xb=B[k][j][i].x;
            current.yb=B[k][j][i].y;
            current.zb=B[k][j][i].z;
            //setGhostStencil(kg, phi, nodetype, x_p[level_temp], y_p[level_temp], z_p[level_temp], current.xb, current.yb, current.zb, stencil, coeffsD, coeffs_dx, coeffs_dy, coeffs_dz, nxb, nyb, nzb, 1);
            if (level==-1)
              setGhostStencil(ctx,kg, phi, nodetype, P, current.xb, current.yb, current.zb, stencil, coeffsD, coeffs_dx, coeffs_dy, coeffs_dz, nxb, nyb, nzb, 1, critici);
            else
              {SETERRQ(PETSC_COMM_SELF,1,"not yet implemented"); CHKERRQ(ierr);}
            current.nx=nxb;
            current.ny=nyb;
            current.nz=nzb;
            for(int cont=0;cont<27;++cont){
              current.stencil[cont]=stencil[cont];
              current.coeffsD[cont]=coeffsD[cont];
              current.coeffs_dx[cont]=coeffs_dx[cont];
              current.coeffs_dy[cont]=coeffs_dy[cont];
              current.coeffs_dz[cont]=coeffs_dz[cont];
              }
            current.dtau=0.9;   
            if(level==-1){ 
              ctx.Ghost.Phi1.push_back(current);
              nodetype[k][j][i]=(ctx.Ghost.Phi1.size())-1;
            }
            else{
              {SETERRQ(PETSC_COMM_SELF,1,"not yet implemented"); CHKERRQ(ierr);}
              //Ghost_p[level].Phi1.push_back(current);
              //nodetype[k][j][i]=(Ghost_p[level].Phi1.size())-1;
            }
          }
          else if(nodetype[k][j][i]==N_GHOSTBDY){
            PetscInt kg =  i + j * nn1_l + k * nn1_l * nn2_l;
            current_Bdy.index=kg;
            current_Bdy.type=2*(i==0)+3*(i==nn1_l-1)+0*(j==0)+1*(j==nn2_l-1)+4*(k==0)+5*(k==nn3_l-1);
            current_Bdy.xb=B[k][j][i].x;
            current_Bdy.yb=B[k][j][i].y;
            current_Bdy.zb=B[k][j][i].z;
            int type=current_Bdy.type;
            int ss=((type==0)-(type==1))*nn1_l+((type==2)-(type==3))+((type==4)-(type==5))*nn1_l*nn2_l;
            current_Bdy.stencil[0]=kg;
            current_Bdy.stencil[1]=kg+ss;
            current_Bdy.stencil[2]=kg+2*ss;
            current_Bdy.coeffsD[0]=1;
            current_Bdy.coeffsD[1]=0;
            current_Bdy.coeffsD[2]=0;
            bool k2 = nodetype[i+2*((type==2)-(type==3))][j+2*((type==0)-(type==1))][k+2*((type==4)-(type==5))]>-4;
            //Perché usa sempre dx??
            // dovrebbe usare dx,dy o dz in tutte le 3 formule,
            // a seconda della faccia su cui siamo (vedi type=current_Bdy.type)
            SETERRQ(PETSC_COMM_SELF,1,"Rivedere queste formule");
            current_Bdy.coeffsN[0]=k2*3./2./ctx.dx+(1-k2)*1./ctx.dx;
            current_Bdy.coeffsN[1]=k2*(-4.)/2./ctx.dx+(1-k2)*(-1.)/ctx.dx;
            current_Bdy.coeffsN[2]=k2*1./2./ctx.dx+(1-k2)*0./ctx.dx;
            current.dtau=0.9; 
            if(level==-1){ 
              ctx.Ghost.Bdy.push_back(current_Bdy);
              nodetype[k][j][i]=(ctx.Ghost.Bdy.size())-1+ctx.nn123;
            }
            else{
              {SETERRQ(PETSC_COMM_SELF,1,"not yet implemented");}
              //Ghost_p[level].Bdy.push_back(current_Bdy);
              //nodetype[k][j][i]=(Ghost_p[level].Bdy.size())-1+nn123;
            }
          }
        }
      }
    }

    if(level==-1){ 
      ierr = DMDAVecRestoreArrayWrite(ctx.daField[var::s], ctx.local_NodeType, &nodetype); CHKERRQ(ierr);
      ierr = DMDAVecRestoreArrayRead(ctx.daField[var::s], ctx.local_Phi, &phi); CHKERRQ(ierr);
      ierr = DMDAVecRestoreArrayRead(ctx.daCoord, ctx.coordsLocal, &P); CHKERRQ(ierr);
      ierr = DMDAVecRestoreArrayRead(ctx.daCoord, ctx.BOUNDARY, &B); CHKERRQ(ierr);

      ierr = DMLocalToGlobalBegin(ctx.daField[var::s], ctx.local_NodeType, INSERT_VALUES, ctx.NODETYPE); CHKERRQ(ierr);
      ierr = DMLocalToGlobalEnd(ctx.daField[var::s], ctx.local_NodeType, INSERT_VALUES, ctx.NODETYPE); CHKERRQ(ierr);
    }
    else{
      {SETERRQ(PETSC_COMM_SELF,1,"not yet implemented"); CHKERRQ(ierr);}
      //ierr = DMDAVecRestoreArrayWrite(CartesianGrid3D_p[level], local_NodeType_p[level], &nodetype);
      //ierr = DMDAVecRestoreArrayRead(CartesianGrid3D_p[level], local_Phi_p[level], &phi);
      //ierr = DMDAVecRestoreArrayRead(CartesianGrid3D_p[level], Xb_p[level], &xb);
      //ierr = DMDAVecRestoreArrayRead(CartesianGrid3D_p[level], Yb_p[level], &yb);
      //ierr = DMDAVecRestoreArrayRead(CartesianGrid3D_p[level], Zb_p[level], &zb);

      //ierr = DMLocalToGlobalBegin(CartesianGrid3D_p[level], local_NodeType_p[level], INSERT_VALUES, NodeType_p[level]);
      //ierr = DMLocalToGlobalEnd(CartesianGrid3D_p[level], local_NodeType_p[level], INSERT_VALUES, NodeType_p[level]);
    }

    //ierr = DMRestoreLocalVector(CartesianGrid3D,&local_NodeType);
    }
    PetscPrintf(PETSC_COMM_WORLD,"set ghost stencils ...DONE\n");

    PetscPrintf(PETSC_COMM_SELF,"------- %d punti critici\n",critici.size());
    scriviListaPunti(critici,"critici");

    return ierr;
}

PetscErrorCode setGhostStencil(AppContext & ctx, PetscInt kg,
  PetscScalar ***phi, PetscScalar ***nodetype,
  DMDACoor3d ***P,
  double& xC, double& yC, double& zC,//boundary point
  int stencil[], double coeffsD[], double coeffs_dx[], double coeffs_dy[], double coeffs_dz[],
  double& nxb, double& nyb, double& nzb,
  int upwind,
  std::vector<shiftedStencil> &critici)
{
  PetscInt nn1 = ctx.nnx;
  PetscInt nn2 = ctx.nny;
  PetscInt nn3 = ctx.nnz;
  PetscScalar dx = ctx.dx;
  PetscScalar dy = ctx.dy;
  PetscScalar dz = ctx.dz;

  //int s=3;
  int i,j,k;
  nGlob2IJK(ctx,kg,i,j,k);
  //PetscPrintf(PETSC_COMM_SELF,"#%d] setGhostStencil %d=(%d,%d,%d)\n",ctx.rank,kg,i,j,k);

  double xG=P[k][j][i].x; double yG=P[k][j][i].y;  double zG=P[k][j][i].z;
  int sx=SIGN(xC-xG);  int sy=SIGN(yC-yG);  int sz=SIGN(zC-zG);
  if(sx==0)
      //sx=SGN(Phi[k-nn1]-Phi[k+nn1]);
      sx=(i<nn1/2.)-(i>=nn1/2.);
  if(sy==0)
      //sy=SGN(Phi[k-1]-Phi[k+1]);
      sy=(j<nn2/2.)-(j>=nn2/2.);
  if(sz==0)
      //sz=SGN(Phi[k-nn1*nn2]-Phi[k+nn1*nn2]);
      sz=(k<nn3/2.)-(k>=nn3/2.);
  //PetscPrintf(PETSC_COMM_SELF,"  sx,sy,sz (%d,%d,%d)\n",sx,sy,sz);
  int iii = ((i==1 && sx<0) || (i==nn1-2 && sx>0) || (upwind==0 && i!=1 && i!=nn1-2));
  int jjj = ((j==1 && sy<0) || (j==nn2-2 && sy>0) || (upwind==0 && j!=1 && j!=nn2-2));
  int kkk = ((k==1 && sz<0) || (k==nn3-2 && sz>0) || (upwind==0 && k!=1 && k!=nn3-2));

  if ( (iii!=0) || (jjj!=0) || (kkk!=0) ){
    PetscPrintf(PETSC_COMM_SELF,"%d (%d,%d,%d)\n",kg,i,j,k);
    PetscPrintf(PETSC_COMM_SELF,"  sx ,sy ,sz  (%d,%d,%d)\n",sx,sy,sz);
    PetscPrintf(PETSC_COMM_SELF,"  iii,jjj,kkk (%d,%d,%d)\n",iii,jjj,kkk);
  }

  for(int kk=0;kk<3;++kk)
    for(int jj=0;jj<3;++jj)
      for(int ii=0;ii<3;++ii)
        stencil[ii+3*jj+9*kk]=kg+sx*(ii-iii)+sy*(jj-jjj)*nn1+sz*(kk-kkk)*nn1*nn2;

  //PetscPrintf(PETSC_COMM_SELF,"  Stencil inziale\n");
  //for (int c=0; c<27; ++c){
    //int ic,jc,kc;
    //nGlob2IJK(ctx,stencil[c],ic,jc,kc);
    //PetscPrintf(PETSC_COMM_SELF,"  (%d,%d,%d)%2.0f",ic,jc,kc,nodetype[kc][jc][ic]);
    //if (c%9==8)
      //PetscPrintf(PETSC_COMM_SELF,"\n");
  //}

  int kg1=stencil[0];
  int i1,j1,k1;
  nGlob2IJK(ctx,kg1,i1,j1,k1);
  
  double xG1=P[k1][j1][i1].x; double yG1=P[k1][j1][i1].y;  double zG1=P[k1][j1][i1].z;
  double thtx=fabs(xC-xG1)/2/dx;
  double thty=fabs(yC-yG1)/2/dy;
  double thtz=fabs(zC-zG1)/2/dz;
  //PetscPrintf(PETSC_COMM_SELF,"  B=(%f,%f,%f): theta=(%f,%f,%f)\n",xC,yC,zC,thtx,thty,thtz);

  double weights_x[]={(1-2*thtx)*(1-thtx), 4*thtx*(1-thtx), thtx*(2*thtx-1)};
  double weights_y[]={(1-2*thty)*(1-thty), 4*thty*(1-thty), thty*(2*thty-1)};
  double weights_z[]={(1-2*thtz)*(1-thtz), 4*thtz*(1-thtz), thtz*(2*thtz-1)};
  double weights_dx[]={(-1+(2*thtx-0.5))/dx, (1-2*(2*thtx-0.5))/dx, (2*thtx-0.5)/dx};
  double weights_dy[]={(-1+(2*thty-0.5))/dy, (1-2*(2*thty-0.5))/dy, (2*thty-0.5)/dy};
  double weights_dz[]={(-1+(2*thtz-0.5))/dz, (1-2*(2*thtz-0.5))/dz, (2*thtz-0.5)/dz};

  for(int kk=0;kk<3;++kk)
    for(int jj=0;jj<3;++jj)
      for(int ii=0;ii<3;++ii){
        int ind=ii+3*jj+9*kk;
        coeffsD[ind]  =   weights_x[ii] *weights_y[jj] *weights_z[kk];
        coeffs_dx[ind]=sx*weights_dx[ii]*weights_y[jj] *weights_z[kk];
        coeffs_dy[ind]=sy*weights_x[ii] *weights_dy[jj]*weights_z[kk];
        coeffs_dz[ind]=sz*weights_x[ii] *weights_y[jj] *weights_dz[kk];
      }

  //calcolo normali sul bordo
  nxb=nyb=nzb=0;
  for(int cont=0;cont<27;++cont)
  {
      int kg_temp=stencil[cont];
      int i_temp,j_temp,k_temp;
      nGlob2IJK(ctx,kg_temp,i_temp,j_temp,k_temp);
      nxb+=phi[k_temp][j_temp][i_temp]*coeffs_dx[cont];
      nyb+=phi[k_temp][j_temp][i_temp]*coeffs_dy[cont];
      nzb+=phi[k_temp][j_temp][i_temp]*coeffs_dz[cont];
  }
  double module=sqrt(nxb*nxb+nyb*nyb+nzb*nzb);
  nxb=nxb/module;
  nyb=nyb/module;
  nzb=nzb/module;

  //PetscPrintf(PETSC_COMM_SELF,"  nB=(%f,%f,%f)\n",nxb,nyb,nzb);

  // check whether or not I have to shift the stencil
  int direction, s1,s2, nn1nn2, sxyz;
  double toll=1e-6*(dx+dy+dz)/3.;
  if(fabs(xC-xG)-toll>fabs(yC-yG)+toll && fabs(xC-xG)-toll>fabs(zC-zG)+toll){
    //normale circa come (1,0,0)
    direction=0;
    nn1nn2=1;
    s1=3;
    s2=9;
    sxyz=sx;
  }
  else if(fabs(yC-yG)-toll>fabs(zC-zG)+toll){
    //normale circa come (0,1,0)
    direction=1;
    nn1nn2=nn1;
    s1=1;
    s2=9;
    sxyz=sy;
  }
  else{
    //normale circa come (0,0,1)
    direction=2;
    s1=1;
    s2=3;
    nn1nn2=nn1*nn2;
    sxyz=sz;
  }

  //indici locali faccia da shiftare
  int Face[9];
  for(int j=0; j<3; ++j)
    for(int i=0; i<3; ++i)
      Face[i+3*j]=s1*i+s2*j;

  unsigned int nShifts=0;
  for(int i=0; i<9; ++i){
    int ind_face=Face[i];
    for(int pt=0; pt<2;++pt)
    {
      int ind=ind_face+pt*powI(3,direction);
      int kg_ind=stencil[ind];
      int i_ind,j_ind,k_ind;
      nGlob2IJK(ctx,kg_ind,i_ind,j_ind,k_ind);

      //if(Mask[stencil[ind]])
      if(nodetype[k_ind][j_ind][i_ind]>-4)
        break;
      else
      {
        //PetscPrintf(PETSC_COMM_SELF,"Ghost %d Shift faccia direzione %d a (%f,%f,%f)\n",kg,direction,xC,yC,zC);
        nShifts++;

        stencil[ind]+=sxyz*3*nn1nn2;
        double cD=coeffsD[ind],    cdx=coeffs_dx[ind],    cdy=coeffs_dy[ind],       cdz=coeffs_dz[ind];
        coeffsD[ind]=0;     coeffs_dx[ind]=0;      coeffs_dy[ind]=0;        coeffs_dz[ind]=0;
        int c_extrap[3];
        if(pt){ c_extrap[0]=-3; c_extrap[1]=1; c_extrap[2]=3;}
        else{ c_extrap[0]=1; c_extrap[1]=3; c_extrap[2]=-3;}
        for(int i=0;i<3;++i){
          int ind=ind_face+i*powI(3,direction);
          coeffsD[ind]+=cD*c_extrap[i];
          coeffs_dx[ind]+=cdx*c_extrap[i];
          coeffs_dy[ind]+=cdy*c_extrap[i];
          coeffs_dz[ind]+=cdz*c_extrap[i];
        }
      }
    }
  }

  //PetscScalar interpError = checkInterp(ctx,P,xC,yC,zC,stencil,coeffsD);
  //if (fabs(interpError)>1e-15){
    //PetscPrintf(PETSC_COMM_SELF,"Interpolation error for quadratic function at %d: %e\n",kg,interpError);
  //}

  if (nShifts>0)
    critici.push_back({xC,yC,zC,nShifts});

  //PetscPrintf(PETSC_COMM_SELF,"  Stencil 2\n");
  //for (int c=0; c<27; ++c){
    //int ic,jc,kc;
    //nGlob2IJK(ctx,stencil[c],ic,jc,kc);
    //PetscPrintf(PETSC_COMM_SELF,"  (%d,%d,%d)%2.0f",ic,jc,kc,nodetype[kc][jc][ic]);
    //if (c%9==8)
      //PetscPrintf(PETSC_COMM_SELF,"\n");
  //}

  //controllo se sono rimasti altri punti non interni nè ghost
  std::bitset<27> mask;
  for(int cont=0;cont<27;++cont){
    int kg_cont=stencil[cont];
    int i_cont,j_cont,k_cont;
    nGlob2IJK(ctx,kg_cont,i_cont,j_cont,k_cont);
    //mask.set(cont,Mask[stencil[cont]]);
    mask.set(cont,nodetype[k_cont][j_cont][i_cont]>-4);
  }

  if(!(mask.all()))
  {
    //riduco al primo ordine
    PetscPrintf(PETSC_COMM_SELF,"#%d] riduco: %d (%d,%d,%d)\n",ctx.rank,kg,i,j,k);
    for(int kk=0;kk<3;++kk)
      for(int jj=0;jj<3;++jj)
        for(int ii=0;ii<3;++ii)
          stencil[ii+3*jj+9*kk] = kg + sy*jj*nn1 + sx*ii + sz*kk*nn1*nn2;
    for(int cont=0;cont<27;++cont)
        coeffsD[cont]=coeffs_dx[cont]=coeffs_dy[cont]=coeffs_dz[cont]=0;
    coeffsD[0]=1;//errore? devo usare indice del ghost

    int kg_1=stencil[1];
    int i_1,j_1,k_1;
    nGlob2IJK(ctx,kg_1,i_1,j_1,k_1);
    int kg_3=stencil[3];
    int i_3,j_3,k_3;
    nGlob2IJK(ctx,kg_3,i_3,j_3,k_3);
    int kg_9=stencil[9];
    int i_9,j_9,k_9;
    nGlob2IJK(ctx,kg_9,i_9,j_9,k_9);

    //PetscPrintf(PETSC_COMM_WORLD,"maremmamaiala\n");
    //PetscPrintf(PETSC_COMM_WORLD,"%d (%d,%d,%d)\n",kg_1,i_1,j_1,k_1);
    //PetscPrintf(PETSC_COMM_WORLD,"%d (%d,%d,%d)\n",kg_3,i_3,j_3,k_3);
    //PetscPrintf(PETSC_COMM_WORLD,"%d (%d,%d,%d)\n",kg_9,i_9,j_9,k_9);

    if(nodetype[k_1][j_1][i_1]>-4){//PetscPrintf(PETSC_COMM_WORLD,"maremmamaiala 2a\n");
      coeffs_dx[0]=-sx/dx;  coeffs_dx[1]=sx/dx; }
    else if(phi[k_3][j_3][i_3]<0){//PetscPrintf(PETSC_COMM_WORLD,"maremmamaiala 2b\n");
      coeffs_dx[3]=-sx/dx; coeffs_dx[4]=sx/dx; }
    else if(phi[k_9][j_9][i_9]<0){//PetscPrintf(PETSC_COMM_WORLD,"maremmamaiala 2c\n");
      coeffs_dx[9]=-sx/dx; coeffs_dx[10]=sx/dx;}
    else {SETERRQ(PETSC_COMM_SELF,1,"Error: No stencil available for the x-derivative.");}

    //PetscPrintf(PETSC_COMM_WORLD,"maremmamaiala 3\n");
    if(nodetype[k_3][j_3][i_3]>-4){ coeffs_dy[0]=-sy/dy;  coeffs_dy[3]=sy/dy; }
    else if(phi[k_1][j_1][i_1]<0){ coeffs_dy[1]=-sy/dy; coeffs_dy[4]=sy/dy; }
    else if(phi[k_9][j_9][i_9]<0){coeffs_dy[9]=-sy/dy; coeffs_dy[12]=sy/dy;}
    else {SETERRQ(PETSC_COMM_SELF,1,"Error: No stencil available for the y-derivative.");}

    //PetscPrintf(PETSC_COMM_WORLD,"maremmamaiala 4\n");
    //TODO: controllare queste!
    if(nodetype[k_9][j_9][i_9]>-4){ coeffs_dz[0]=-sz/dz;  coeffs_dz[9]=sz/dz; }
    else if(phi[k_1][j_1][i_1]<0){ coeffs_dz[1]=-sz/dz; coeffs_dz[10]=sz/dz; }
    else if(phi[k_3][j_3][i_3]<0){coeffs_dz[3]=-sz/dz; coeffs_dz[12]=sz/dz;}
    else {SETERRQ(PETSC_COMM_SELF,1,"Error: No stencil available for the z-derivative.");}

    //PetscPrintf(PETSC_COMM_SELF,"  Stencil 3\n");
    //for (int c=0; c<27; ++c){
      //int ic,jc,kc;
      //nGlob2IJK(ctx,stencil[c],ic,jc,kc);
      //PetscPrintf(PETSC_COMM_SELF,"  (%d,%d,%d)",ic,jc,kc);
      //if (c%9==8)
        //PetscPrintf(PETSC_COMM_SELF,"\n");
    //}
  }
  //PetscPrintf(PETSC_COMM_SELF,"Done setGhostStencil %d\n",kg);

  return 0;
}

PetscErrorCode setMatValuesHelmoltz(AppContext &ctx, DM da, Vec Gamma, Vec Sigma, PetscScalar nu, Mat A)
{
  //Calls MatSetValues on A to insert values for the linear operator
  // u -> sigma*u + alpha* Div( gamma Grad u) on inner nodes
  // interpolation of u on xB for ghost nodes
  //
  // The matrix A should already exist and
  // assembly routines should be called afterwards by the caller.
  //
  // gamma should be a local vector with ghost values set correctly:
  // we do not call communication routines on gamma before using it.

  PetscErrorCode ierr;

  //PetscInt xs, ys, zs, xm, ym, zm;
  PetscScalar ***gamma, ***sigma, ***nodetype;

  double dx2=ctx.dx*ctx.dx;
  double dy2=ctx.dy*ctx.dy;
  double dz2=ctx.dz*ctx.dz;

  ierr = DMDAVecGetArrayRead(da, Gamma, &gamma);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da, Sigma, &sigma);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da, ctx.NODETYPE, &nodetype);CHKERRQ(ierr);

  for (PetscInt k=ctx.daInfo.zs; k<ctx.daInfo.zs+ctx.daInfo.zm; k++){
    for (PetscInt j=ctx.daInfo.ys; j<ctx.daInfo.ys+ctx.daInfo.ym; j++){
      for (PetscInt i=ctx.daInfo.xs; i<ctx.daInfo.xs+ctx.daInfo.xm; i++){
        if(nodetype[k][j][i]==N_INACTIVE)
        { //identity matrix
          MatStencil row;
          PetscScalar val=1.0;
          row.i = i; row.j = j; row.k = k; row.c=var::s;
          MatSetValuesStencil(A,1,&row,1,&row,&val,INSERT_VALUES);
        }
        else if(nodetype[k][j][i]==N_INSIDE)
        {
          double sigmaC = sigma[k][j][i];
          double gammaX1 = nu * (gamma[k][j][i] + gamma[k][j][i-1]) / 2.;
          double gammaX2 = nu * (gamma[k][j][i] + gamma[k][j][i+1]) / 2.;
          double gammaY1 = nu * (gamma[k][j][i] + gamma[k][j-1][i]) / 2.;
          double gammaY2 = nu * (gamma[k][j][i] + gamma[k][j+1][i]) / 2.;
          double gammaZ1 = nu * (gamma[k][j][i] + gamma[k-1][j][i]) / 2.;
          double gammaZ2 = nu * (gamma[k][j][i] + gamma[k+1][j][i]) / 2.;
          MatStencil rows = {0}, cols[7] = {{0}};
          PetscScalar vals[7];
          PetscInt ncols = 0;
          rows.i = i; rows.j = j; rows.k = k;  rows.c=var::s;
          cols[ncols].i = i; cols[ncols].j = j; cols[ncols].k = k; cols[ncols].c=var::s;
          vals[ncols++] = sigmaC+ (gammaX1+gammaX2)/dx2+(gammaY1+gammaY2)/dy2+(gammaZ1+gammaZ2)/dz2;
          cols[ncols].i = i-1; cols[ncols].j = j; cols[ncols].k = k; cols[ncols].c=var::s;
          vals[ncols++] = -gammaX1/dx2;
          cols[ncols].i = i+1; cols[ncols].j = j; cols[ncols].k = k; cols[ncols].c=var::s;
          vals[ncols++] = -gammaX2/dx2;
          cols[ncols].i = i; cols[ncols].j = j-1; cols[ncols].k = k; cols[ncols].c=var::s;
          vals[ncols++] = -gammaY1/dy2;
          cols[ncols].i = i; cols[ncols].j = j+1; cols[ncols].k = k; cols[ncols].c=var::s;
          vals[ncols++] = -gammaY2/dy2;                    
          cols[ncols].i = i; cols[ncols].j = j; cols[ncols].k = k-1; cols[ncols].c=var::s;
          vals[ncols++] = -gammaZ1/dz2;
          cols[ncols].i = i; cols[ncols].j = j; cols[ncols].k = k+1; cols[ncols].c=var::s;
          vals[ncols++] = -gammaZ2/dz2;

          MatSetValuesStencil(A,1,&rows,ncols,cols,vals,INSERT_VALUES);
        }
        else if (nodetype[k][j][i] >= 0)
        {
          if (nodetype[k][j][i] < ctx.nn123){ // Ghost.Phi1
            ghost & current = ctx.Ghost.Phi1[nodetype[k][j][i]];
            MatStencil rows = {0}, cols[27] = {{0}};
            PetscScalar vals[27];
            PetscInt ncols = 0;
            rows.i = i; rows.j = j; rows.k = k; rows.c=var::s;
            for(int cont=0;cont<27;++cont){
                int kg_ghost=current.stencil[cont];
                int i_ghost, j_ghost, k_ghost;
                nGlob2IJK(ctx, kg_ghost, i_ghost, j_ghost, k_ghost);
                cols[ncols].i = i_ghost; cols[ncols].j = j_ghost; cols[ncols].k = k_ghost; cols[ncols].c=var::s;
                vals[ncols++] = current.coeffsD[cont];
            }
            MatSetValuesStencil(A,1,&rows,ncols,cols,vals,INSERT_VALUES);
          }
          else{ // Ghost.Bdy
            SETERRQ(PETSC_COMM_SELF,1,"Not (yet) implemented");
            //MatStencil rows = {0}, cols[1] = {{0}};
            //PetscScalar vals[1];
            //rows.i = i; rows.j = j; rows.k = k;
            //cols[0].i = i; cols[0].j = j; cols[0].k = k;
            //vals[0] = 1.;
            //MatSetValuesStencil(M,1,&rows,1,cols,vals,INSERT_VALUES);
            //identity matrix
            MatStencil row;
            PetscScalar val=1.0;
            row.i = i; row.j = j; row.k = k; row.c=var::s;
            MatSetValuesStencil(A,1,&row,1,&row,&val,INSERT_VALUES);
          }
        }
        else
          SETERRQ(PETSC_COMM_SELF,1,"Error: nodetype has values that are not supported.");
      }
    }
  }

  ierr = DMDAVecRestoreArrayRead(da, ctx.POROSloc, &gamma);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da, ctx.Sigma   , &sigma);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da, ctx.NODETYPE, &nodetype);CHKERRQ(ierr);

  return ierr;
}

PetscScalar checkInterp(AppContext &ctx,DMDACoor3d ***P,PetscScalar xC,PetscScalar yC,PetscScalar zC,int stencil[], double coeffsD[]){
  PetscScalar value=0.;
  for (int p=0; p<27;p++){
    int i,j,k;
    nGlob2IJK(ctx,stencil[p],i,j,k);
    const PetscScalar xP=P[k][j][i].x;
    const PetscScalar yP=P[k][j][i].y;
    const PetscScalar zP=P[k][j][i].z;
    const PetscScalar pValue = 1.0 + (xP-xC)*(xP-xC) + (yP-yC)*(yP-yC)+ (zP-zC)*(zP-zC);
    value += coeffsD[p] * pValue;
  }
  return (value-1.0);
}

PetscScalar Phi1_(DMDACoor3d p)
{
  //const PetscScalar x0=sqrt(2.)/30.;
  //const PetscScalar y0=sqrt(3.)/40.;
  //const PetscScalar z0=-sqrt(2.)/50.;
  ////PetscScalar radius=0.786;
  //const PetscScalar aa=0.786;
  //const PetscScalar bb=0.386;
  //const PetscScalar cc=0.586;
  //return pow((p.x-x0)/aa,2)+pow((p.y-y0)/bb,2)+pow((p.z-z0)/cc,2)-1;
  const PetscScalar x0=0.;
  const PetscScalar y0=0.;
  const PetscScalar z0=0.;
  const PetscScalar aa=1.;
  const PetscScalar bb=1.;
  const PetscScalar cc=1.;
  return pow((p.x-x0)/aa,2)+pow((p.y-y0)/bb,2)+pow((p.z-z0)/cc,2)-0.7*0.7;

}
