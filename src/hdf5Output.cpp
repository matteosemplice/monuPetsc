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

  //s
  ierr = DMGetGlobalVector(ctx.daField[var::s], &uField); CHKERRQ(ierr);
  ierr = VecStrideGather(U,var::s,uField,INSERT_VALUES); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) uField, "S");
  ierr = VecView(uField,viewer); CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(ctx.daField[var::s], &uField); CHKERRQ(ierr);

  //c
  ierr = DMGetGlobalVector(ctx.daField[var::c], &uField); CHKERRQ(ierr);
  ierr = VecStrideGather(U,var::c,uField,INSERT_VALUES); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) uField, "C");
  ierr = VecView(uField,viewer); CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(ctx.daField[var::c], &uField); CHKERRQ(ierr);

  //levelset function
  ierr = VecView(ctx.Phi,viewer); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) ctx.NODETYPE, "NodeType");
  ierr = VecView(ctx.NODETYPE,viewer); CHKERRQ(ierr);

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
    fprintf(file, "      <Topology TopologyType=\"2DCoRectMesh\" Dimensions=\"%d %d\"/>\n",ctx.nx,ctx.ny);
    fprintf(file, "      <Geometry GeometryType=\"ORIGIN_DXDY\">\n");
    fprintf(file, "        <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"2\">%f %f</DataItem>\n",ctx.xmin,ctx.ymin);
    fprintf(file, "        <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"2\">%f %f</DataItem>\n",ctx.dx,ctx.dy);
    fprintf(file, "      </Geometry>\n");
    fprintf(file, "      <Attribute Name=\"SO2\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file, "        <DataItem Format=\"HDF\" Dimensions=\"%d %d\">%s:/S</DataItem>\n",ctx.nx,ctx.ny,hdf5name);
    fprintf(file, "      </Attribute>\n");
    fprintf(file, "      <Attribute Name=\"CaCO3\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file, "        <DataItem Format=\"HDF\" Dimensions=\"%d %d\">%s:/C</DataItem>\n",ctx.nx,ctx.ny,hdf5name);
    fprintf(file, "      </Attribute>\n");
    break;
    case 3:
    fprintf(file, "      <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\"%d %d %d\"/>\n",ctx.nx,ctx.ny,ctx.nz);
    fprintf(file, "      <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
    fprintf(file, "        <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"3\">%f %f %f</DataItem>\n",ctx.xmin,ctx.ymin,ctx.zmin);
    fprintf(file, "        <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"3\">%f %f %f</DataItem>\n",ctx.dx,ctx.dy,ctx.dz);
    fprintf(file, "      </Geometry>\n");
    fprintf(file, "      <Attribute Name=\"SO2\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file, "        <DataItem Format=\"HDF\" Dimensions=\"%d %d %d\">%s:/S</DataItem>\n",ctx.nx,ctx.ny,ctx.nz,hdf5name);
    fprintf(file, "      </Attribute>\n");
    fprintf(file, "      <Attribute Name=\"CaCO3\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file, "        <DataItem Format=\"HDF\" Dimensions=\"%d %d %d\">%s:/C</DataItem>\n",ctx.nx,ctx.ny,ctx.nz,hdf5name);
    fprintf(file, "      </Attribute>\n");
    fprintf(file, "      <Attribute Name=\"Phi\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file, "        <DataItem Format=\"HDF\" Dimensions=\"%d %d %d\">%s:/Phi</DataItem>\n",ctx.nx,ctx.ny,ctx.nz,hdf5name);
    fprintf(file, "      </Attribute>\n");
    fprintf(file, "      <Attribute Name=\"NodeType\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file, "        <DataItem Format=\"HDF\" Dimensions=\"%d %d %d\">%s:/NodeType</DataItem>\n",ctx.nx,ctx.ny,ctx.nz,hdf5name);
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
