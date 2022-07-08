/* Output su file VTK (per ParaView) */

#ifndef __HDF5OUTPUT_H
#define __HDF5OUTPUT_H

#include "appctx.h"

#include <vector>
#include <string>

PetscErrorCode writePhi(AppContext &_ctx);

typedef struct{
  PetscScalar t;
  std::string hdfSnippet;
} hdfTime;

class HDF5output{
public:
  HDF5output(const char * _basename, AppContext &_ctx, PetscScalar tFinal, int nSave):
    ctx(_ctx),
    dtSave(tFinal/nSave),
    nextSave(nSave>=0?0.:INFINITY)
    {strncpy(basename,_basename,250);};

  //! HDF5 output of the domain (levelset and NodeTypes)
  PetscErrorCode writeDomain(AppContext &ctx);

  //! HDF5 output of the solution
  //! with singleXDMF=true it will also write a XDMF for this timestep
  PetscErrorCode writeHDF5(Vec U, PetscScalar time, bool singleXDMF=false);

  //! writes XDMF for the entire simulation
  void writeSimulationXDMF();

  void skipNSave(PetscInt nSkip) {
    hdfTime tStep;
    for (PetscInt i=0; i<nSkip; ++i){
      tStep.t = nextSave;
      tStep.hdfSnippet = "SKIPPED";
      stepBuffer.push_back(tStep);
      nextSave += dtSave; 
    }
    PetscPrintf(PETSC_COMM_WORLD,"Next save point (%d) set at %f\n",stepBuffer.size(),nextSave);
  }

private:
  AppContext &ctx;
  char basename[250];
  char hdf5DomainName[256];
  std::vector<hdfTime> stepBuffer;
  PetscScalar dtSave, nextSave;
  //! writes XDMF for a single timestep (the last)
  void writeLastXDMF();
};

#endif

