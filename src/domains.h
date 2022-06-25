#ifndef __DOMAINS_H
#define __DOMAINS_H

#include <petscdmda.h>
#include "appctx.h"

typedef PetscScalar (*levelSetFPointer)(DMDACoor3d);

PetscErrorCode getDomainFromOptions(levelSetFPointer &domain, AppContext & ctx);

//PetscScalar Phi1_ellipsoid(DMDACoor3d p);
//PetscScalar Phi1_sphere(DMDACoor3d p);

#endif
