/* Output su file VTK (per ParaView) */

#include <petscviewerhdf5.h>

#include "appctx.h"
#include "hdf5Output.h"
#include <stdio.h>
//#include <assert.h>

PetscErrorCode WriteHDF5(AppContext &ctx, const char * basename, Vec U){
  FILE *file;
  PetscErrorCode ierr;

  char  hdf5name[256];
  PetscSNPrintf(hdf5name,256,"%s.h5",basename);

  PetscViewer viewer;
  PetscPrintf(PETSC_COMM_WORLD,"Save on file %s\n",hdf5name);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,hdf5name,FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
  Vec uField;
  ierr = VecGetSubVector(U,ctx.is[var::s],&uField); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) uField, "S");
  ierr = VecView(uField,viewer); CHKERRQ(ierr);
  ierr = VecRestoreSubVector(U,ctx.is[var::s],&uField); CHKERRQ(ierr);
  ierr = VecGetSubVector(U,ctx.is[var::c],&uField); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) uField, "C");
  ierr = VecView(uField,viewer); CHKERRQ(ierr);
  ierr = VecRestoreSubVector(U,ctx.is[var::c],&uField); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  if (ctx.rank==0){
    char  xdmfname[256];
    PetscSNPrintf(xdmfname,256,"%s.xdmf",basename);

    file = fopen(xdmfname, "w");

    fprintf(file, "<?xml version=\"1.0\" ?>\n");
    fprintf(file, "<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude\" Version=\"2.0\">\n");
    fprintf(file, "<Domain>\n");
    fprintf(file, "  <Grid GridType=\"Collection\" CollectionType=\"Temporal\">\n");
    fprintf(file, "    <Time TimeType=\"List\">\n");
    fprintf(file, "      <DataItem Dimensions=\"1\">1.0</DataItem>\n");
    fprintf(file, "    </Time>\n");

    fprintf(file, "    <Grid GridType=\"Uniform\" Name=\"domain\">\n");
    switch (ctx.dim){
    case 2:
    fprintf(file, "      <Topology TopologyType=\"2DCoRectMesh\" Dimensions=\"%d %d\"/>\n",ctx.Nx,ctx.Nx);
    fprintf(file, "      <Geometry GeometryType=\"ORIGIN_DXDY\">\n");
    fprintf(file, "        <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"2\">0.0 0.0</DataItem>\n");
    fprintf(file, "        <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"2\">%f %f</DataItem>\n",ctx.dx,ctx.dx);
    fprintf(file, "      </Geometry>\n");
    fprintf(file, "      <Attribute Name=\"SO2\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file, "        <DataItem Format=\"HDF\" Dimensions=\"%d %d\">%s:/S</DataItem>\n",ctx.Nx,ctx.Nx,hdf5name);
    fprintf(file, "      </Attribute>\n");
    fprintf(file, "      <Attribute Name=\"CaCO3\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file, "        <DataItem Format=\"HDF\" Dimensions=\"%d %d\">%s:/C</DataItem>\n",ctx.Nx,ctx.Nx,hdf5name);
    fprintf(file, "      </Attribute>\n");
    break;
    case 3:
    fprintf(file, "      <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\"%d %d %d\"/>\n",ctx.Nx,ctx.Nx,ctx.Nx);
    fprintf(file, "      <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
    fprintf(file, "        <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"3\">0.0 0.0 0.0</DataItem>\n");
    fprintf(file, "        <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"3\">%f %f %f</DataItem>\n",ctx.dx,ctx.dx,ctx.dx);
    fprintf(file, "      </Geometry>\n");
    fprintf(file, "      <Attribute Name=\"SO2\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file, "        <DataItem Format=\"HDF\" Dimensions=\"%d %d %d\">%s:/S</DataItem>\n",ctx.Nx,ctx.Nx,ctx.Nx,hdf5name);
    fprintf(file, "      </Attribute>\n");
    fprintf(file, "      <Attribute Name=\"CaCO3\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file, "        <DataItem Format=\"HDF\" Dimensions=\"%d %d %d\">%s:/C</DataItem>\n",ctx.Nx,ctx.Nx,ctx.Nx,hdf5name);
    fprintf(file, "      </Attribute>\n");
    break;
    default:
    abort();
    }
    fprintf(file, "    </Grid>\n");
    fprintf(file, "  </Grid>\n");
    fprintf(file, "</Domain>\n");
    fprintf(file, "</Xdmf>\n");

    fclose(file);
  }


  return 0;
}
