#include "domains.h"
#include <fstream>

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
inline DMDACoor3d rotate(DMDACoor3d p, DMDACoor3d n, PetscScalar theta){
  DMDACoor3d q;
  const PetscScalar C = std::cos(theta);
  const PetscScalar S = std::sin(theta);
  q.x = (C+n.x*n.x*(1-C))*p.x + (n.x*n.y*(1-C)-n.z*S)*p.y + (n.x*n.z*(1-C)+n.y*S)*p.z;
  q.y = (n.x*n.y*(1-C)+n.z*S)*p.x + (C+n.y*n.y*(1-C))*p.y + (n.y*n.z*(1-C)-n.x*S)*p.z;
  q.z = (n.x*n.z*(1-C)-n.y*S)*p.x + (n.y*n.z*(1-C)+n.x*S)*p.y + (C+n.z*n.z*(1-C))*p.z;
  return q;
}

inline PetscScalar _sphere(PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar x0, PetscScalar y0, PetscScalar z0, PetscScalar R){
  return (x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0)-R*R;
}

PetscScalar Phi1_sphere(DMDACoor3d p)
{
  return _sphere(p.x,p.y,p.z,0.,0.,0.,0.7);
}

inline PetscScalar _flower(PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar A){
  const PetscScalar r=sqrt(x*x+y*y+z*z);
  const PetscScalar app = (pow(z,5) +5. * z * pow(x*x+y*y,2) -10. * (x*x+y*y) * pow(z,3)) 
                          / (3.* pow(r,5));
  return r - 0.5 - A*  (r<1e-12?0.:app);
}

PetscScalar Phi1_flower(DMDACoor3d p){
  return _flower(p.x,p.y,p.z,1.0);
}

PetscScalar Phi1_smoothflower(DMDACoor3d p){
  DMDACoor3d n={1./sqrt(3.),1./sqrt(3.),1./sqrt(3.)};
  DMDACoor3d q=p;//rotate(p,n,M_PI_4);
//  q.x+=0.01;
  return _flower(q.x,q.y,q.z,0.5);
}

inline PetscScalar _cube(PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar L){
  const PetscScalar ax=std::abs(x);
  const PetscScalar ay=std::abs(y);
  const PetscScalar az=std::abs(z);
  PetscScalar v;
  if ( (ax-L/2.>0.)&&(ay-L/2.>0.)&&(az-L/2.>0.))
    v= std::sqrt( (ax-L/2.)*(ax-L/2.) + (ay-L/2.)*(ay-L/2.)+ (az-L/2.)*(az-L/2.));
  else
    v= std::max(ax,std::max(ay,az))-L/2.;
  return v;
}

//inline PetscScalar _cylinder(PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar x0, PetscScalar y0, PetscScalar R, PetscScalar L){
  //PetscScalar disk=(x-x0)*(x-x0)+(y-y0)*(y-y0)-R*R;
  //PetscScalar segment=abs(z)-L/2.;
  //printf("(%f,%f,%f): max(%f , %f)= %f\n",x,y,z,disk,segment,fmax(disk,segment));
  //return fmax(disk,segment);
//}

PetscScalar Phi1_cubosfere(DMDACoor3d p){
  DMDACoor3d n={1./sqrt(3.),1./sqrt(3.),1./sqrt(3.)};
  DMDACoor3d q=rotate(p,n,M_PI_4);
  const PetscScalar cubo = _cube(q.x,q.y,q.z,0.8);
  const PetscScalar s1   = _sphere(q.x,q.y,q.z,0.4,0.4,-0.4,0.15);
  const PetscScalar s2   = _sphere(q.x,q.y,q.z,0.4,0.4,+0.4,0.15);
  const PetscScalar s3   = _sphere(q.x,q.y,q.z,-0.4,-0.4,0.,0.25);

  return std::min(cubo,std::min(s1,std::min(s2,s3)));
}

PetscErrorCode getDomainFromOptions(levelSetFPointer &domain, AppContext & ctx){
  PetscErrorCode ierr;
  PetscBool domainGiven;
  ierr = PetscOptionsGetString(NULL,NULL,"-domain",ctx.domainName,255,&domainGiven);CHKERRQ(ierr);

  domain = &Phi1_sphere;
  if (domainGiven){
    if (strcmp(ctx.domainName,"sphere")==0){
      domain = &Phi1_sphere;
      PetscPrintf(PETSC_COMM_WORLD,"Chosen SPHERE domain\n");
    } else if (strcmp(ctx.domainName,"ellipsoid")==0){
      domain = &Phi1_ellipsoid;
      PetscPrintf(PETSC_COMM_WORLD,"Chosen ELLIPSOIDAL domain\n");
    } else if (strcmp(ctx.domainName,"smoothflower")==0){
      domain = &Phi1_flower;
      PetscPrintf(PETSC_COMM_WORLD,"Chosen SMOOTH FLOWER domain\n");
    } else if (strcmp(ctx.domainName,"flower")==0){
      domain = &Phi1_smoothflower;
      PetscPrintf(PETSC_COMM_WORLD,"Chosen HARD FLOWER domain\n");
    } else if (strcmp(ctx.domainName,"cubespheres")==0){
      domain = &Phi1_cubosfere;
      PetscPrintf(PETSC_COMM_WORLD,"Chosen CUBE&SPHERES domain\n");
    } else {
      domain = NULL;
      PetscPrintf(PETSC_COMM_WORLD,"Will load levelset from %s \n",ctx.domainName);
      //SETERRQ(PETSC_COMM_SELF,1,"Domain name not recognized");
    }
  }
  return ierr;
}
