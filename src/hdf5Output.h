/* Output su file VTK (per ParaView) */

#ifndef __HDF5OUTPUT_H
#define __HDF5OUTPUT_H

#include "appctx.h"

#include <vector>
#include <string>

typedef struct{
  PetscScalar t;
  std::string hdfSnippet;
} hdfTime;

class HDF5output{
public:
  HDF5output(const char * _basename, AppContext &_ctx):
    ctx(_ctx)
    {strncpy(basename,_basename,250);};

  //! HDF5 output of the solution
  //! with singleXDMF=true it will also write a XDMF for this timestep
  PetscErrorCode writeHDF5(Vec U, PetscScalar time, bool singleXDMF=false);

  //! writes XDMF for the entire simulation
  void writeSimulationXDMF();

private:
  AppContext &ctx;
  char basename[250];
  std::vector<hdfTime> stepBuffer;

  //! writes XDMF for a single timestep (the last)
  void writeLastXDMF();
};

#endif

