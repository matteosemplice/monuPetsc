/* Output su file VTK (per ParaView) */

#ifndef __HDF5OUTPUT_H
#define __HDF5OUTPUT_H

#include "appctx.h"

//! HDF5 output of the solution
PetscErrorCode WriteHDF5(AppContext &ctx, const char * basename, Vec U);

#endif

