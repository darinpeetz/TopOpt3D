#include "mpi.h" // Has to precede Petsc includes to use MPI::BOOL
#include "TopOpt.h"
#include <fstream>

using namespace std;

PetscErrorCode TopOpt::PrepLog()
{
  PetscErrorCode ierr = 0;
  ierr = PetscLogEventRegister("Optimization Update", 0, &UpdateEvent); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Functions", 0, &funcEvent); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("FE Analysis", 0, &FEEvent); CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode TopOpt::Clear()
{ 
  PetscErrorCode ierr = 0;
  delete[] B; delete[] G; delete[] GT; delete[] W;
  ierr = VecDestroy(&F); CHKERRQ(ierr);
  ierr = VecDestroy(&U); CHKERRQ(ierr);
  //ierr = MatDestroy(&spK); CHKERRQ(ierr);
  ierr = VecDestroy(&spKVec); CHKERRQ(ierr);
  ierr = MatDestroy(&K); CHKERRQ(ierr);
  ierr = VecDestroy(&MLump); CHKERRQ(ierr);
  ierr = KSPDestroy(&KUF); CHKERRQ(ierr);
  //ierr = KSPDestroy(&dynamicKSP); CHKERRQ(ierr);
  //ierr = KSPDestroy(&bucklingKSP); CHKERRQ(ierr);
  ierr = MatDestroy(&P); CHKERRQ(ierr);
  ierr = VecDestroy(&V); CHKERRQ(ierr);
  ierr = VecDestroy(&dVdy); CHKERRQ(ierr);
  ierr = VecDestroy(&E); CHKERRQ(ierr);
  ierr = VecDestroy(&dEdy); CHKERRQ(ierr);
  ierr = VecDestroy(&Es); CHKERRQ(ierr);
  ierr = VecDestroy(&dEsdy); CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = VecDestroy(&rho); CHKERRQ(ierr);
  for (unsigned int i = 0; i < function_list.size(); i++)
    delete function_list[i];
  return ierr;
}

PetscErrorCode TopOpt::MeshOut ( )
{
  PetscErrorCode ierr = 0;
  ofstream file;
  if (this->myid == 0)
  {
    file.open("Element_Distribution.bin", ios::binary);
    file.write((char*)this->elmdist.data(), this->elmdist.size()*sizeof(PetscInt));
    file.close();
    file.open("Node_Distribution.bin", ios::binary);
    file.write((char*)this->nddist.data(), this->nddist.size()*sizeof(PetscInt));
    file.close();
  }

  // Getting distribution of edges, loads, supports, springs, and masses
  int edgedist = this->edgeElem.rows(), loaddist = this->loadNode.rows();
  int suppdist = this->suppNode.rows(), springdist = this->springNode.rows();
  int massdist = this->massNode.rows();
  MPI_Request edgereq, loadreq, suppreq, springreq, massreq;
  ierr = MPI_Iscan(MPI_IN_PLACE, &edgedist, 1, MPI_INT, MPI_SUM, this->comm,
            &edgereq); CHKERRQ(ierr);
  ierr = MPI_Iscan(MPI_IN_PLACE, &loaddist, 1, MPI_INT, MPI_SUM, this->comm,
            &loadreq); CHKERRQ(ierr);
  ierr = MPI_Iscan(MPI_IN_PLACE, &suppdist, 1, MPI_INT, MPI_SUM, this->comm,
            &suppreq); CHKERRQ(ierr);
  ierr = MPI_Iscan(MPI_IN_PLACE, &springdist, 1, MPI_INT, MPI_SUM, this->comm,
            &springreq); CHKERRQ(ierr);
  ierr = MPI_Iscan(MPI_IN_PLACE, &massdist, 1, MPI_INT, MPI_SUM, this->comm,
            &massreq); CHKERRQ(ierr);

  MPI_File fh;
  int myid = this->myid, nprocs = this->nprocs;
  Eigen::Array<PetscInt, -1, -1, Eigen::RowMajor> global_int;
  Eigen::Array<double, -1, -1, Eigen::RowMajor> global_float;
  // Writing element array
  ierr = MPI_File_open(this->comm, "elements.bin", MPI_MODE_CREATE |
             MPI_MODE_WRONLY | MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
         CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);
  ierr = MPI_File_open(this->comm, "elements.bin", MPI_MODE_CREATE |
                       MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); CHKERRQ(ierr);
  ierr = MPI_File_seek(fh, this->elmdist(myid) * this->element.cols() *
                       sizeof(PetscInt), MPI_SEEK_SET); CHKERRQ(ierr);
  global_int.resize(this->nLocElem, this->element.cols());
  for (int el = 0; el < this->nLocElem; el++)
  {
    for (int nd = 0; nd < this->element.cols(); nd++)
      global_int(el,nd) = this->gNode(this->element(el,nd)); 
  }
  ierr = MPI_File_write_all(fh, global_int.data(), global_int.size(),
                            MPI_PETSCINT, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);

  // Writing node array
  ierr = MPI_File_open(this->comm, "nodes.bin", MPI_MODE_CREATE |
             MPI_MODE_WRONLY | MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
         CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);
  ierr = MPI_File_open(this->comm, "nodes.bin", MPI_MODE_CREATE |
                       MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); CHKERRQ(ierr);
  ierr = MPI_File_seek(fh, this->nddist(myid) * this->node.cols() *
                       sizeof(double), MPI_SEEK_SET); CHKERRQ(ierr);
  ierr = MPI_File_write_all(fh, this->node.data(), this->nLocNode *
             this->node.cols(), MPI_DOUBLE, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);

  // Writing edge array
  ierr = MPI_File_open(this->comm, "edges.bin", MPI_MODE_CREATE |
             MPI_MODE_WRONLY | MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
          CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);
  ierr = MPI_File_open(this->comm, "edges.bin", MPI_MODE_CREATE |
                       MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); CHKERRQ(ierr);
  ierr = MPI_Wait(&edgereq, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  edgedist -= this->edgeElem.rows();
  ierr = MPI_File_seek(fh, 2*edgedist*sizeof(PetscInt), MPI_SEEK_SET); CHKERRQ(ierr);
  global_int.resize(this->edgeElem.rows(), 2);
  for (int el = 0; el < this->edgeElem.rows(); el++)
  {
    global_int(el, 0) = this->gElem(this->edgeElem(el,0));
    if (this->edgeElem(el,1) < this->gElem.size())

      global_int(el, 1) = this->gElem(this->edgeElem(el,1));
    else
      global_int(el, 1) = this->nElem;
  }
  ierr = MPI_File_write_all(fh, global_int.data(), global_int.size(),
                            MPI_PETSCINT, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);

  // Writing edge length array
  ierr = MPI_File_open(this->comm, "edgeLengths.bin", MPI_MODE_CREATE |
             MPI_MODE_WRONLY | MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
         CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);
  ierr = MPI_File_open(this->comm, "edgeLengths.bin", MPI_MODE_CREATE |
                       MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); CHKERRQ(ierr);
  ierr = MPI_File_seek(fh, edgedist*sizeof(double), MPI_SEEK_SET); CHKERRQ(ierr);
  ierr = MPI_File_write_all(fh, this->edgeSize.data(), this->edgeSize.size(),
                            MPI_DOUBLE, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);

  // Writing load node array
  ierr = MPI_File_open(this->comm, "loadNodes.bin", MPI_MODE_CREATE |
             MPI_MODE_WRONLY | MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
         CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);
  ierr = MPI_File_open(this->comm, "loadNodes.bin", MPI_MODE_CREATE |
                       MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); CHKERRQ(ierr);
  ierr = MPI_Wait(&loadreq, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  loaddist -= this->loadNode.rows();
  ierr = MPI_File_seek(fh, loaddist*sizeof(PetscInt), MPI_SEEK_SET); CHKERRQ(ierr);
  global_int.resize(this->loadNode.rows(), 1);
  for (int i = 0; i < this->loadNode.size(); i++)
    global_int(i, 0) = this->gNode(this->loadNode(i));
  ierr = MPI_File_write_all(fh, global_int.data(), global_int.size(),
                            MPI_PETSCINT, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);

  // Writing loads array
  ierr = MPI_File_open(this->comm, "loads.bin", MPI_MODE_CREATE |
             MPI_MODE_WRONLY | MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
         CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);
  ierr = MPI_File_open(this->comm, "loads.bin", MPI_MODE_CREATE |
                       MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); CHKERRQ(ierr);
  ierr = MPI_File_seek(fh, loaddist * this->loads.cols() * sizeof(double),
                       MPI_SEEK_SET); CHKERRQ(ierr);
  ierr = MPI_File_write_all(fh, this->loads.data(), this->loads.size(),
                            MPI_DOUBLE, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);

  // Writing support node array
  ierr = MPI_File_open(this->comm, "supportNodes.bin", MPI_MODE_CREATE |
             MPI_MODE_WRONLY | MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
         CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);
  ierr = MPI_File_open(this->comm, "supportNodes.bin", MPI_MODE_CREATE |
                       MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); CHKERRQ(ierr);
  ierr = MPI_Wait(&suppreq, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  suppdist -= this->suppNode.rows();
  ierr = MPI_File_seek(fh, suppdist*sizeof(PetscInt), MPI_SEEK_SET); CHKERRQ(ierr);
  global_int.resize(this->suppNode.rows(), 1);
  for (int i = 0; i < this->suppNode.size(); i++)
    global_int(i, 0) = this->gNode(this->suppNode(i)); CHKERRQ(ierr);
  ierr = MPI_File_write_all(fh, global_int.data(), global_int.size(),
                            MPI_PETSCINT, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);

  // Writing supports array
  ierr = MPI_File_open(this->comm, "supports.bin", MPI_MODE_CREATE |
             MPI_MODE_WRONLY | MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
         CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);
  ierr = MPI_File_open(this->comm, "supports.bin", MPI_MODE_CREATE |
                       MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); CHKERRQ(ierr);
  ierr = MPI_File_seek(fh, suppdist * this->supports.cols() * sizeof(bool),
                       MPI_SEEK_SET); CHKERRQ(ierr);
  ierr = MPI_File_write_all(fh, this->supports.data(), this->supports.size(),
                            MPI::BOOL, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);

  // Writing spring node array
  ierr = MPI_File_open(this->comm, "springNodes.bin", MPI_MODE_CREATE |
             MPI_MODE_WRONLY | MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
         CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);
  ierr = MPI_File_open(this->comm, "springNodes.bin", MPI_MODE_CREATE |
                       MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); CHKERRQ(ierr);
  ierr = MPI_Wait(&springreq, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  springdist -= this->springNode.rows();
  ierr = MPI_File_seek(fh, springdist*sizeof(PetscInt), MPI_SEEK_SET); CHKERRQ(ierr);
  global_int.resize(this->springNode.rows(), 1);
  for (int i = 0; i < this->springNode.size(); i++)
    global_int(i, 0) = this->gNode(this->springNode(i));
  ierr = MPI_File_write_all(fh, global_int.data(), global_int.size(),
                            MPI_PETSCINT, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);

  // Writing springs array
  ierr = MPI_File_open(this->comm, "springs.bin", MPI_MODE_CREATE |
             MPI_MODE_WRONLY | MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
         CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);
  ierr = MPI_File_open(this->comm, "springs.bin", MPI_MODE_CREATE |
                       MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); CHKERRQ(ierr);
  ierr = MPI_File_seek(fh, springdist * this->springs.cols() * sizeof(double),
                       MPI_SEEK_SET); CHKERRQ(ierr);
  ierr = MPI_File_write_all(fh, this->springs.data(), this->springs.size(),
                            MPI_DOUBLE, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);

  // Writing mass node array
  ierr = MPI_File_open(this->comm, "massNodes.bin", MPI_MODE_CREATE |
             MPI_MODE_WRONLY | MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
         CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);
  ierr = MPI_File_open(this->comm, "massNodes.bin", MPI_MODE_CREATE |
                       MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); CHKERRQ(ierr);
  ierr = MPI_Wait(&massreq, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  massdist -= this->massNode.rows();
  ierr = MPI_File_seek(fh, massdist*sizeof(PetscInt), MPI_SEEK_SET); CHKERRQ(ierr);
  global_int.resize(this->massNode.rows(), 1);
  for (int i = 0; i < this->massNode.size(); i++)
    global_int(i, 0) = this->gNode(this->massNode(i));
  ierr = MPI_File_write_all(fh, global_int.data(), global_int.size(),
                            MPI_PETSCINT, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);

  // Writing masses array
  ierr = MPI_File_open(this->comm, "masses.bin", MPI_MODE_CREATE |
             MPI_MODE_WRONLY | MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
         CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);
  ierr = MPI_File_open(this->comm, "masses.bin", MPI_MODE_CREATE |
                       MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); CHKERRQ(ierr);
  ierr = MPI_File_seek(fh, massdist*this->masses.cols()*sizeof(double),
                       MPI_SEEK_SET); CHKERRQ(ierr);
  ierr = MPI_File_write_all(fh, this->masses.data(), this->masses.size(),
                            MPI_DOUBLE, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);

  // Writing filter
  PetscViewer view;
  ierr = PetscViewerBinaryOpen(this->comm, "Filter.bin", FILE_MODE_WRITE, &view); CHKERRQ(ierr);
  ierr = MatView(this->P, view); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view); CHKERRQ(ierr);

  // Writing projecting matrices
  ArrayXPI lcol(this->nprocs);
  for (unsigned int i = 0; i < this->PR.size(); i++)
  {
    stringstream level; level << i;
    string filename = "P" + level.str() + ".bin";
    ierr = PetscViewerBinaryOpen(this->comm, filename.c_str(), FILE_MODE_WRITE, &view); CHKERRQ(ierr);
    ierr = MatView(this->PR[i], view); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view); CHKERRQ(ierr);
    ierr = MatGetLocalSize(this->PR[i], NULL, lcol.data()+this->myid); CHKERRQ(ierr);
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_PETSCINT, lcol.data(), 1, MPI_PETSCINT, this->comm);
    if (this->myid == 0)
    {
      filename += ".split";
      file.open(filename.c_str(), ios::binary);
      file.write((char*)lcol.data(), lcol.size()*sizeof(PetscInt));
      file.close();
    }
  }
  
  return 0;
}

PetscErrorCode TopOpt::StepOut ( const double &f,
                                 const Eigen::VectorXd &cons, int it )
{
  PetscErrorCode ierr = 0;

  ierr = PetscFPrintf(this->comm, this->output, "Iteration number: %u\tObjective: %1.6g\n",
              it, f); CHKERRQ(ierr);
  ierr = PetscFPrintf(this->comm, this->output, "Constraints:\n"); CHKERRQ(ierr);
  for (short i = 0; i < cons.size(); i++)
  {
    ierr = PetscFPrintf(this->comm, this->output, "%1.12g\t", cons(i)); CHKERRQ(ierr);
  }
  ierr = PetscFPrintf(this->comm, this->output, "\n\n"); CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode TopOpt::ResultOut ( int it )
{
  PetscErrorCode ierr = 0;

  // Output a ratio of stiffness to volume
  PetscScalar Esum, Vsum;
  ierr = VecSum(this->E, &Esum); CHKERRQ(ierr);
  ierr = VecSum(this->V, &Vsum); CHKERRQ(ierr);
  ierr = PetscFPrintf(this->comm, this->output, "************************************************\n");
  ierr = PetscFPrintf(this->comm, this->output, "After %4i iterations with a penalty of %1.4g the\n",
              it, this->penal); CHKERRQ(ierr);
  ierr = PetscFPrintf(this->comm, this->output, "ratio of stiffness sum to volume sum is %1.4g\n",
              Esum/Vsum); CHKERRQ(ierr);
  ierr = PetscFPrintf(this->comm, this->output, "************************************************\n\n");

  stringstream pen;
  pen << this->penal;
  PetscViewer output;
  char filename[30];

  sprintf(filename, "U_pen%1.4g.bin", this->penal);
  ierr = PetscViewerBinaryOpen(this->comm, filename,
      FILE_MODE_WRITE, &output); CHKERRQ(ierr);
  ierr = VecView(this->U, output); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&output); CHKERRQ(ierr);

  sprintf(filename, "x_pen%1.4g.bin", this->penal);
  ierr = PetscViewerBinaryOpen(this->comm, filename,
      FILE_MODE_WRITE, &output); CHKERRQ(ierr);
  ierr = VecView(this->x, output); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&output); CHKERRQ(ierr);

  sprintf(filename, "V_pen%1.4g.bin", this->penal);
  ierr = PetscViewerBinaryOpen(this->comm, filename,
      FILE_MODE_WRITE, &output); CHKERRQ(ierr);
  ierr = VecView(this->V, output); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&output); CHKERRQ(ierr);

  sprintf(filename, "E_pen%1.4g.bin", this->penal);
  ierr = PetscViewerBinaryOpen(this->comm, filename,
      FILE_MODE_WRITE, &output); CHKERRQ(ierr);
  ierr = VecView(this->E, output); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&output); CHKERRQ(ierr);

  for (int i = 0; i < this->bucklingShape.cols(); i++)
  {
    sprintf(filename,"phiB_pen%1.4g_mode%i.bin", this->penal, i);
    Vec phi;
    ierr = VecCreateMPIWithArray(this->comm, 1, this->numDims*this->nLocNode,
        this->numDims*this->nNode, this->bucklingShape.data() +
        this->bucklingShape.rows()*i, &phi); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(this->comm, filename,
        FILE_MODE_WRITE, &output); CHKERRQ(ierr);
    ierr = VecView(phi, output); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&output); CHKERRQ(ierr);
    ierr = VecDestroy(&phi); CHKERRQ(ierr);
  }

  for (int i = 0; i < this->dynamicShape.cols(); i++)
  {
    sprintf(filename,"phiD_pen%1.4g_mode%i.bin", this->penal, i);
    Vec phi;
    ierr = VecCreateMPIWithArray(this->comm, 1, this->numDims*this->nLocNode,
        this->numDims*this->nNode, this->dynamicShape.data() +
        this->dynamicShape.rows()*i, &phi); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(this->comm, filename,
        FILE_MODE_WRITE, &output); CHKERRQ(ierr);
    ierr = VecView(phi, output); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&output); CHKERRQ(ierr);
    ierr = VecDestroy(&phi); CHKERRQ(ierr);
  }

  return 0;
}
