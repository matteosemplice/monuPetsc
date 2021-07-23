static char help[] = "Test parallel HDF5 output\n";

#include <petscversion.h>
#include <petscdmda.h>
#include <petscviewer.h>
#include <petscsys.h>
#include <petscviewerhdf5.h>

int main(int argc, char **argv) {

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help); CHKERRQ(ierr);
  char petscVersion[255];
  ierr = PetscGetVersion(petscVersion,255);
  PetscPrintf(PETSC_COMM_WORLD,"Compiled with %s\n",petscVersion);
  PetscInt Nx=11;
  PetscInt Ny=11;
  PetscScalar dx = 1.0 / (Nx-1);
  PetscScalar dy = 1.0 / (Ny-1);
  DM dmda;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                      DMDA_STENCIL_STAR,
                      Nx,Ny, //global dim
                      PETSC_DECIDE,PETSC_DECIDE, //n proc on each dim
                      2,1, //dof, stencil width
                      NULL, NULL, //n nodes per direction on each cpu
                      &dmda);      CHKERRQ(ierr);
  ierr = DMSetFromOptions(dmda); CHKERRQ(ierr);
  ierr = DMSetUp(dmda); CHKERRQ(ierr); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dmda, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0); CHKERRQ(ierr);
  ierr = DMDASetFieldName(dmda,0,"s"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(dmda,1,"c"); CHKERRQ(ierr);
  DMDALocalInfo daInfo;
  ierr = DMDAGetLocalInfo(dmda,&daInfo); CHKERRQ(ierr);
  IS *is;
  DM *daField;
  ierr = DMCreateFieldDecomposition(dmda,NULL, NULL, &is, &daField); CHKERRQ(ierr);
  Vec U0;
  ierr = DMCreateGlobalVector(dmda,&U0); CHKERRQ(ierr);

  //Initial data
  typedef struct{ PetscScalar s,c;} data_type;
  data_type **u;
  ierr = DMDAVecGetArray(dmda,U0,&u); CHKERRQ(ierr);
  for (PetscInt j=daInfo.ys; j<daInfo.ys+daInfo.ym; j++){
    PetscScalar y = j*dy;
    for (PetscInt i=daInfo.xs; i<daInfo.xs+daInfo.xm; i++){
      PetscScalar x = i*dx;
      u[j][i].s = x+2.*y;
      u[j][i].c = 10. + 2.*x*x+y*y;
    }
  }
  ierr = DMDAVecRestoreArray(dmda,U0,&u); CHKERRQ(ierr);

  PetscViewer viewer;
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"solutionSC.hdf5",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
  Vec uField;
  ierr = DMCreateGlobalVector(daField[0], &uField); CHKERRQ(ierr);
  ierr = VecStrideGather(U0,0,uField,INSERT_VALUES); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) uField, "S");
  ierr = VecView(uField,viewer); CHKERRQ(ierr);
  ierr = VecStrideGather(U0,1,uField,INSERT_VALUES); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) uField, "C");
  ierr = VecView(uField,viewer); CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(daField[0], &uField); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
