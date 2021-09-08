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

inline PetscScalar _flower(PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar A){
  const PetscScalar r=sqrt(x*x+y*y+z*z);
  return r - 0.5
         - A*(pow(z,5)
              +5. * z * pow(x*x+y*y,2)
              -10. * (x*x+y*y) * pow(z,3)
             ) / (3.* pow(r,5));
}

PetscScalar Phi1_flower(DMDACoor3d p){
  return _flower(p.x,p.y,p.z,1.0);
}

PetscScalar Phi1_smoothflower(DMDACoor3d p){
  return _flower(p.x,p.y,p.z,0.5);
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
    } else if (strcmp(domainName,"flower")==0){
      domain = &Phi1_flower;
      PetscPrintf(PETSC_COMM_WORLD,"Chosen SMOOTH FLOWER domain\n");
    } else if (strcmp(domainName,"smoothflower")==0){
      domain = &Phi1_smoothflower;
      PetscPrintf(PETSC_COMM_WORLD,"Chosen HARD FLOWER domain\n");
    } else
      SETERRQ(PETSC_COMM_SELF,1,"Domain name not recognized");
  }
  return ierr;
}
