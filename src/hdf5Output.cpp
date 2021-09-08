/* Output su file HDF5 (e XDMF per ParaView) */

#include <petscviewerhdf5.h>

#include "hdf5Output.h"

#include <sstream>
#include <fstream>

PetscErrorCode HDF5output::writeHDF5(Vec U, PetscScalar time, bool singleXDMF){
  PetscErrorCode ierr;

  char  hdf5name[256];
  PetscSNPrintf(hdf5name,256,"%s_%d.h5",basename,stepBuffer.size());

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

  //levelset function and NodeTypes
  ierr = VecView(ctx.Phi,viewer); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) ctx.NODETYPE, "NodeType");
  ierr = VecView(ctx.NODETYPE,viewer); CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  std::stringstream buffer;

  if (ctx.rank==0){
    char  xdmfname[256];
    PetscSNPrintf(xdmfname,256,"%s_%d.xdmf",basename,stepBuffer.size());

    std::ofstream xdmf;
    xdmf.open(xdmfname,std::ios_base::out);

    xdmf   << "<?xml version=\"1.0\" ?>\n";
    xdmf   << "<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude\" Version=\"2.0\">\n";
    xdmf   << "<Domain>\n";
    xdmf   << "  <Grid GridType=\"Collection\" CollectionType=\"Temporal\">\n";

    buffer << "    <Grid GridType=\"Uniform\" Name=\"domain\">\n";
    buffer << "      <Time Type=\"Single\" Value=\""<<time<<"\" />\n";
    switch (ctx.dim){
    case 2:
    buffer << "      <Topology TopologyType=\"2DCoRectMesh\" Dimensions=\"" << ctx.nnx << " " << ctx.nny << "\"/>\n";
    buffer << "      <Geometry GeometryType=\"ORIGIN_DXDY\">\n";
    buffer << "        <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"2\">" << ctx.xmin << " " << ctx.ymin << "</DataItem>\n";
    buffer << "        <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"2\">" << ctx.dx   << " " << ctx.dy   << "</DataItem>\n";
    buffer << "      </Geometry>\n";
    buffer << "      <Attribute Name=\"SO2\" Center=\"Node\" AttributeType=\"Scalar\">\n";
    buffer << "        <DataItem Format=\"HDF\" Dimensions=\"" << ctx.nnx << " " << ctx.nny << "\">"<<hdf5name<<":/S</DataItem>\n";
    buffer << "      </Attribute>\n";
    buffer << "      <Attribute Name=\"CaCO3\" Center=\"Node\" AttributeType=\"Scalar\">\n";
    buffer << "        <DataItem Format=\"HDF\" Dimensions=\"" << ctx.nnx << " " << ctx.nny << "\">"<<hdf5name<<":/C</DataItem>\n";
    buffer << "      </Attribute>\n";
    break;
    case 3:
    buffer << "      <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\"" << ctx.nnx << " " << ctx.nny << " " << ctx.nnz << "\"/>\n";
    buffer << "      <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
    buffer << "        <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"3\">" << ctx.xmin << " " << ctx.ymin << " " << ctx.zmin << "</DataItem>\n";
    buffer << "        <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"3\">" << ctx.dx   << " " << ctx.dy   << " " << ctx.dz   << "</DataItem>\n";
    buffer << "      </Geometry>\n";
    buffer << "      <Attribute Name=\"SO2\" Center=\"Node\" AttributeType=\"Scalar\">\n";
    buffer << "        <DataItem Format=\"HDF\" Dimensions=\"" << ctx.nnx << " " << ctx.nny << " " << ctx.nnz << "\">"<<hdf5name<<":/S</DataItem>\n";
    buffer << "      </Attribute>\n";
    buffer << "      <Attribute Name=\"CaCO3\" Center=\"Node\" AttributeType=\"Scalar\">\n";
    buffer << "        <DataItem Format=\"HDF\" Dimensions=\"" << ctx.nnx << " " << ctx.nny << " " << ctx.nnz << "\">"<<hdf5name<<":/C</DataItem>\n";
    buffer << "      </Attribute>\n";
    buffer << "      <Attribute Name=\"Phi\" Center=\"Node\" AttributeType=\"Scalar\">\n";
    buffer << "        <DataItem Format=\"HDF\" Dimensions=\"" << ctx.nnx << " " << ctx.nny << " " << ctx.nnz << "\">"<<hdf5name<<":/Phi</DataItem>\n";
    buffer << "      </Attribute>\n";
    buffer << "      <Attribute Name=\"NodeType\" Center=\"Node\" AttributeType=\"Scalar\">\n";
    buffer << "        <DataItem Format=\"HDF\" Dimensions=\"" << ctx.nnx << " " << ctx.nny << " " << ctx.nnz << "\">"<<hdf5name<<":/NodeType</DataItem>\n";
    buffer << "      </Attribute>\n";
    break;
    default:
    abort();
    }
    buffer << "    </Grid>\n";

    xdmf   << buffer.str();
    xdmf   << "  </Grid>\n";
    xdmf   << "</Domain>\n";
    xdmf   << "</Xdmf>\n";

    xdmf.close();
  }

  //Note: ranks>0 have empty buffers, but stepBuffer.size does increase for everybody
  hdfTime tStep;
  tStep.t = time;
  tStep.hdfSnippet = buffer.str();
  stepBuffer.push_back(tStep);

  if (singleXDMF) writeLastXDMF();

  return 0;
}

void HDF5output::writeLastXDMF(){
  if (stepBuffer.empty())
    return;

  if (ctx.rank==0){
    char  xdmfname[256];
    PetscSNPrintf(xdmfname,256,"%s_%d.xdmf",basename,stepBuffer.size());

    std::ofstream xdmf;
    xdmf.open(xdmfname,std::ios_base::out);

    xdmf   << "<?xml version=\"1.0\" ?>\n";
    xdmf   << "<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude\" Version=\"2.0\">\n";
    xdmf   << "<Domain>\n";
    xdmf   << "  <Grid GridType=\"Collection\" CollectionType=\"Temporal\">\n";
    xdmf   << stepBuffer.back().hdfSnippet;
    xdmf   << "  </Grid>\n";
    xdmf   << "</Domain>\n";
    xdmf   << "</Xdmf>\n";

    xdmf.close();
  }
}

void HDF5output::writeSimulationXDMF(){
  if (ctx.rank==0){
    char  xdmfname[256];
    snprintf(xdmfname,256,"%s.xdmf",basename);

    std::ofstream xdmf;
    xdmf.open(xdmfname,std::ios_base::out);

    xdmf << "<?xml version=\"1.0\" ?>\n";
    xdmf << "<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude\" Version=\"2.0\">\n";
    xdmf << "<Domain>\n";
    xdmf << "  <Grid GridType=\"Collection\" CollectionType=\"Temporal\">\n";
    xdmf << "    <Time TimeType=\"List\">\n";
    xdmf << "      <DataItem Dimensions=\""<< stepBuffer.size()<<"\">\n";
    for (auto it = stepBuffer.begin(); it != stepBuffer.end(); it++)
      xdmf << (*it).t << " " ;
    xdmf << "\n";
    xdmf << "      </DataItem>\n";
    xdmf   << "    </Time>\n";
    xdmf   << "\n";

    for (auto it = stepBuffer.begin(); it != stepBuffer.end(); it++)
      xdmf   << (*it).hdfSnippet << "\n";

    xdmf   << "  </Grid>\n";
    xdmf   << "</Domain>\n";
    xdmf   << "</Xdmf>\n";

    xdmf.close();
  }
}
