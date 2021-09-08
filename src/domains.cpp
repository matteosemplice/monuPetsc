#include "domains.h"

PetscScalar Phi1_ellipsoid(DMDACoor3d p)
{
  const PetscScalar x0=sqrt(2.)/30.;
  const PetscScalar y0=sqrt(3.)/40.;
  const PetscScalar z0=-sqrt(2.)/50.;
  const PetscScalar aa=0.786;
  const PetscScalar bb=0.386;
  const PetscScalar cc=0.586;
  return pow((p.x-x0)/aa,2)+pow((p.y-y0)/bb,2)+pow((p.z-z0)/cc,2)-1;
}

PetscScalar Phi1_sphere(DMDACoor3d p)
{
  const PetscScalar x0=0.;
  const PetscScalar y0=0.;
  const PetscScalar z0=0.;
  const PetscScalar aa=1.;
  const PetscScalar bb=1.;
  const PetscScalar cc=1.;
  return pow((p.x-x0)/aa,2)+pow((p.y-y0)/bb,2)+pow((p.z-z0)/cc,2)-0.7*0.7;
}

PetscErrorCode getDomainFromOptions(levelSetFPointer &domain){
  PetscErrorCode ierr;
  char domainName[255];
  PetscBool domainGiven;
  ierr = PetscOptionsGetString(NULL,NULL,"-domain",domainName,255,&domainGiven);CHKERRQ(ierr);

  domain = &Phi1_sphere;
  if (domainGiven){
    if (strcmp(domainName,"sphere")==0){
      domain = &Phi1_sphere;
      PetscPrintf(PETSC_COMM_WORLD,"Chosen SPHERE domain\n");
    } else if (strcmp(domainName,"ellipsoid")==0){
      domain = &Phi1_ellipsoid;
      PetscPrintf(PETSC_COMM_WORLD,"Chosen ELLIPSOIDAL domain\n");
    } else
      SETERRQ(PETSC_COMM_SELF,1,"Domain name not recognized");
  }
  return ierr;
}
