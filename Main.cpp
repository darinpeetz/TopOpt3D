#include "mpi.h"
#include <iostream>
#include <fstream>
#include <Eigen/Eigen>
#include <sstream>
#include "TopOpt.h"
#include "RecMesh.h"
#include "EigLab.h"
#include "MMA.h"
#include "Inputs.h"
#include "EigenPeetz.h"
#include <slepceps.h>
#include "Functions.h"

using namespace std;

static char help[] = "The topology optimization routine we deserve, but not the one we need right now.\n\n";

int MeshOut ( TopOpt *topOpt );
int StepOut ( TopOpt *topOpt, const double &f, const Eigen::VectorXd &cons, int it );
int ResultOut ( TopOpt *topOpt, int it );

int main(int argc, char **args)
{
    /// MPI Variables
    int myid, nproc;
    PetscErrorCode ierr = 0;
    SlepcInitialize(&argc,&args,(char*)0,help);
    ierr = EigenPeetz::Initialize(); CHKERRQ(ierr);
    MPI_Comm Opt_Comm = MPI_COMM_WORLD;
    MPI_Comm_rank(Opt_Comm, &myid);
    MPI_Comm_size(Opt_Comm, &nproc);

    /// Optimization Variables
    TopOpt * topOpt = new TopOpt;
    MMA * optmma = new MMA;
    optmma->Set_Comm(Opt_Comm);

    // Open file for outputs
    ierr = PetscFOpen(topOpt->comm, "Output.txt", "w", &topOpt->output); CHKERRQ(ierr);

    /// Input Parameters
    topOpt->filename = "Standard_Input";
    char input[256]; PetscBool hasInput;
    ierr = PetscOptionsGetString(NULL, NULL, "-Input", input, 256, &hasInput); CHKERRQ(ierr);
    if (hasInput)
      topOpt->filename = input;

    Eigen::VectorXd Dimensions;
    ArrayXPI Nel;
    double R;

    bool Normalization = false, Reorder_Mesh = true;
    PetscInt mg_levels = 2, min_size = -1;
    topOpt->Def_Param(optmma, topOpt, Dimensions, Nel, R, Normalization,
                      Reorder_Mesh, mg_levels, min_size);
    mg_levels = max(mg_levels, 2);
    topOpt->Set_Funcs();

    /// Domain, Boundary Conditions, and initial design variables
    Eigen::VectorXd xIni;
    if (topOpt->folder.length() > 0)
    {
      ierr = topOpt->LoadMesh(xIni); CHKERRQ(ierr);
    }
    else
    {
      ierr = topOpt->CreateMesh(Dimensions, Nel, R, Reorder_Mesh, 
                                mg_levels, min_size); CHKERRQ(ierr);
      topOpt->Def_BC();
      xIni = 0.5*Eigen::VectorXd::Ones(topOpt->nLocElem);
      topOpt->penal = topOpt->pmin;
    }

    // Write out the mesh to file
    MeshOut( topOpt );

    /// Design Variable Initialization
    optmma->Set_Lower_Bound( Eigen::VectorXd::Constant(topOpt->nLocElem, 0) );
    optmma->Set_Upper_Bound( Eigen::VectorXd::Ones(topOpt->nLocElem) );
    optmma->Set_Init_Values( xIni );
    optmma->Set_n( topOpt->nLocElem );

    /// Optimize
    cout.precision(12);
    topOpt->Initialize();
    PetscInt ncon = 0;
    for (unsigned int ii = 0; ii < topOpt->function_list.size(); ii++)
    {
      if (topOpt->function_list[ii]->objective == PETSC_FALSE)
        ncon++;
    }
    double f;
    VectorXPS dfdx(topOpt->nLocElem), g(ncon);
    MatrixXPS dgdx(topOpt->nLocElem, ncon);
    topOpt->bucklingShape.resize(topOpt->node.size(), topOpt->bucklingShape.cols());
    topOpt->dynamicShape.resize(topOpt->node.size(), topOpt->dynamicShape.cols());
    for (unsigned int i = 0; i < topOpt->function_list.size(); i++){
      ierr = topOpt->function_list[i]->Initialize_Arrays(topOpt->nLocElem); CHKERRQ(ierr); }

    for ( ; topOpt->penal <= topOpt->pmax; topOpt->penal += topOpt->pstep )
    {
      ierr = PetscFPrintf(topOpt->comm, topOpt->output, "\nPenalty increased to %1.3g\n",
                  topOpt->penal); CHKERRQ(ierr);

      optmma->Set_It(0);
      topOpt->MatIntFnc( optmma->Get_x() );
      ierr = PetscLogEventBegin(topOpt->FEEvent, 0, 0, 0, 0); CHKERRQ(ierr);
      if (topOpt->needK)
      {
        ierr = topOpt->FEAssemble(); CHKERRQ(ierr);
      }
      if (topOpt->needU)
      {
        ierr = topOpt->FESolve(); CHKERRQ(ierr);
      }
      ierr = PetscLogEventEnd(topOpt->FEEvent, 0, 0, 0, 0); CHKERRQ(ierr);
      ierr = PetscLogEventBegin(topOpt->funcEvent, 0, 0, 0, 0); CHKERRQ(ierr);
      ierr = Function_Base::Function_Call( topOpt, f, dfdx, g, dgdx ); CHKERRQ(ierr);
      ierr = PetscLogEventEnd(topOpt->funcEvent, 0, 0, 0, 0); CHKERRQ(ierr);
      StepOut(topOpt, f, g, optmma->Get_it());

/*ierr = PetscFClose(topOpt->comm, topOpt->output); CHKERRQ(ierr);
    delete topOpt;
    delete optmma;

    ierr = SlepcFinalize(); CHKERRQ(ierr);

    return ierr; */
      do
      {
        ierr = PetscLogEventBegin(topOpt->UpdateEvent, 0, 0, 0, 0); CHKERRQ(ierr);
        optmma->Update( dfdx, g, dgdx );
        ierr = PetscLogEventEnd(topOpt->UpdateEvent, 0, 0, 0, 0); CHKERRQ(ierr);
        topOpt->MatIntFnc( optmma->Get_x() );
        ierr = PetscLogEventBegin(topOpt->FEEvent, 0, 0, 0, 0); CHKERRQ(ierr);
        if (topOpt->needK)
        {
          ierr = topOpt->FEAssemble(); CHKERRQ(ierr);
        }
        if (topOpt->needU)
        {
          ierr = topOpt->FESolve(); CHKERRQ(ierr);
        }
        ierr = PetscLogEventEnd(topOpt->FEEvent, 0, 0, 0, 0); CHKERRQ(ierr);
        ierr = PetscLogEventBegin(topOpt->funcEvent, 0, 0, 0, 0); CHKERRQ(ierr);
        ierr = Function_Base::Function_Call( topOpt, f, dfdx, g, dgdx ); CHKERRQ(ierr);
        ierr = PetscLogEventEnd(topOpt->funcEvent, 0, 0, 0, 0); CHKERRQ(ierr);

        StepOut(topOpt, f, g, optmma->Get_it());

      } while ( !optmma->Check() );

      /// Print result after this penalization
      ResultOut(topOpt, optmma->Get_it());
    }

    /// Print out all function values if desired
    if (Normalization)
    {
      ierr = PetscFPrintf(topOpt->comm, topOpt->output, "***Final Values***\n"); CHKERRQ(ierr);
      ierr = Function_Base::Normalization(topOpt); CHKERRQ(ierr);
    }

    /// Wrap up and finish
    ierr = PetscFClose(topOpt->comm, topOpt->output); CHKERRQ(ierr);
    delete topOpt;
    delete optmma;

    ierr = EigenPeetz::Finalize(); CHKERRQ(ierr);
    ierr = SlepcFinalize(); CHKERRQ(ierr);

    return ierr;
}

int MeshOut ( TopOpt *topOpt )
{
  ofstream file;
  if (topOpt->myid == 0)
  {
    file.open("Element_Distribution.bin", ios::binary);
    file.write((char*)topOpt->elmdist.data(), topOpt->elmdist.size()*sizeof(PetscInt));
    file.close();
    file.open("Node_Distribution.bin", ios::binary);
    file.write((char*)topOpt->nddist.data(), topOpt->nddist.size()*sizeof(PetscInt));
    file.close();
  }

  // Getting distribution of edges, loads, supports, springs, and masses
  int edgedist = topOpt->edgeElem.rows(), loaddist = topOpt->loadNode.rows();
  int suppdist = topOpt->suppNode.rows(), springdist = topOpt->springNode.rows();
  int massdist = topOpt->massNode.rows();
  MPI_Request edgereq, loadreq, suppreq, springreq, massreq;
  MPI_Iscan(MPI_IN_PLACE, &edgedist, 1, MPI_INT, MPI_SUM, topOpt->comm, &edgereq);
  MPI_Iscan(MPI_IN_PLACE, &loaddist, 1, MPI_INT, MPI_SUM, topOpt->comm, &loadreq);
  MPI_Iscan(MPI_IN_PLACE, &suppdist, 1, MPI_INT, MPI_SUM, topOpt->comm, &suppreq);
  MPI_Iscan(MPI_IN_PLACE, &springdist, 1, MPI_INT, MPI_SUM, topOpt->comm, &springreq);
  MPI_Iscan(MPI_IN_PLACE, &massdist, 1, MPI_INT, MPI_SUM, topOpt->comm, &massreq);

  MPI_File fh;
  int myid = topOpt->myid, nprocs = topOpt->nprocs;
  Eigen::Array<PetscInt, -1, -1, Eigen::RowMajor> global_int;
  Eigen::Array<double, -1, -1, Eigen::RowMajor> global_float;
  // Writing element array
  MPI_File_open(topOpt->comm, "elements.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY | 
                                              MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
  MPI_File_close(&fh);
  MPI_File_open(topOpt->comm, "elements.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_File_seek(fh, topOpt->elmdist(myid)*topOpt->element.cols()*sizeof(PetscInt), MPI_SEEK_SET);
  global_int.resize(topOpt->nLocElem, topOpt->element.cols());
  for (int el = 0; el < topOpt->nLocElem; el++)
  {
    for (int nd = 0; nd < topOpt->element.cols(); nd++)
      global_int(el,nd) = topOpt->gNode(topOpt->element(el,nd)); 
  }
  MPI_File_write_all(fh, global_int.data(), global_int.size(), MPI_PETSCINT, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);

  // Writing node array
  MPI_File_open(topOpt->comm, "nodes.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY | 
                                           MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
  MPI_File_close(&fh);
  MPI_File_open(topOpt->comm, "nodes.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_File_seek(fh, topOpt->nddist(myid)*topOpt->node.cols()*sizeof(double), MPI_SEEK_SET);
  MPI_File_write_all(fh, topOpt->node.data(), topOpt->nLocNode*topOpt->node.cols(), MPI_DOUBLE, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);

  // Writing edge array
  MPI_File_open(topOpt->comm, "edges.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY | 
                                           MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
  MPI_File_close(&fh);
  MPI_File_open(topOpt->comm, "edges.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_Wait(&edgereq, MPI_STATUS_IGNORE);
  edgedist -= topOpt->edgeElem.rows();
  MPI_File_seek(fh, edgedist*2*sizeof(PetscInt), MPI_SEEK_SET);
  global_int.resize(topOpt->edgeElem.rows(), 2);
  for (int el = 0; el < topOpt->edgeElem.rows(); el++)
  {
    global_int(el, 0) = topOpt->gElem(topOpt->edgeElem(el,0));
    if (topOpt->edgeElem(el,1) < topOpt->gElem.size())

      global_int(el, 1) = topOpt->gElem(topOpt->edgeElem(el,1));
    else
      global_int(el, 1) = topOpt->nElem;
  }
  MPI_File_write_all(fh, global_int.data(), global_int.size(), MPI_PETSCINT, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);

  // Writing edge length array
  MPI_File_open(topOpt->comm, "edgeLengths.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY | 
                                                 MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
  MPI_File_close(&fh);
  MPI_File_open(topOpt->comm, "edgeLengths.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_File_seek(fh, edgedist*sizeof(double), MPI_SEEK_SET);
  MPI_File_write_all(fh, topOpt->edgeSize.data(), topOpt->edgeSize.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);

  // Writing load node array
  MPI_File_open(topOpt->comm, "loadNodes.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY | 
                                               MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
  MPI_File_close(&fh);
  MPI_File_open(topOpt->comm, "loadNodes.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_Wait(&loadreq, MPI_STATUS_IGNORE);
  loaddist -= topOpt->loadNode.rows();
  MPI_File_seek(fh, loaddist*sizeof(PetscInt), MPI_SEEK_SET);
  global_int.resize(topOpt->loadNode.rows(), 1);
  for (int i = 0; i < topOpt->loadNode.size(); i++)
    global_int(i, 0) = topOpt->gNode(topOpt->loadNode(i));
  MPI_File_write_all(fh, global_int.data(), global_int.size(), MPI_PETSCINT, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);

  // Writing loads array
  MPI_File_open(topOpt->comm, "loads.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY |
                                                 MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
  MPI_File_close(&fh);
  MPI_File_open(topOpt->comm, "loads.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_File_seek(fh, loaddist*topOpt->loads.cols()*sizeof(double), MPI_SEEK_SET);
  MPI_File_write_all(fh, topOpt->loads.data(), topOpt->loads.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);

  // Writing support node array
  MPI_File_open(topOpt->comm, "supportNodes.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY |
                                               MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
  MPI_File_close(&fh);
  MPI_File_open(topOpt->comm, "supportNodes.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_Wait(&suppreq, MPI_STATUS_IGNORE);
  suppdist -= topOpt->suppNode.rows();
  MPI_File_seek(fh, suppdist*sizeof(PetscInt), MPI_SEEK_SET);
  global_int.resize(topOpt->suppNode.rows(), 1);
  for (int i = 0; i < topOpt->suppNode.size(); i++)
    global_int(i, 0) = topOpt->gNode(topOpt->suppNode(i));
  MPI_File_write_all(fh, global_int.data(), global_int.size(), MPI_PETSCINT, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);

  // Writing supports array
  MPI_File_open(topOpt->comm, "supports.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY |
                                                 MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
  MPI_File_close(&fh);
  MPI_File_open(topOpt->comm, "supports.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_File_seek(fh, suppdist*topOpt->supports.cols()*sizeof(bool), MPI_SEEK_SET);
  MPI_File_write_all(fh, topOpt->supports.data(), topOpt->supports.size(), MPI::BOOL, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);

  // Writing spring node array
  MPI_File_open(topOpt->comm, "springNodes.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY |
                                               MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
  MPI_File_close(&fh);
  MPI_File_open(topOpt->comm, "springNodes.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_Wait(&springreq, MPI_STATUS_IGNORE);
  springdist -= topOpt->springNode.rows();
  MPI_File_seek(fh, springdist*sizeof(PetscInt), MPI_SEEK_SET);
  global_int.resize(topOpt->springNode.rows(), 1);
  for (int i = 0; i < topOpt->springNode.size(); i++)
    global_int(i, 0) = topOpt->gNode(topOpt->springNode(i));
  MPI_File_write_all(fh, global_int.data(), global_int.size(), MPI_PETSCINT, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);

  // Writing springs array
  MPI_File_open(topOpt->comm, "springs.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY |
                                                 MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
  MPI_File_close(&fh);
  MPI_File_open(topOpt->comm, "springs.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_File_seek(fh, springdist*topOpt->springs.cols()*sizeof(double), MPI_SEEK_SET);
  MPI_File_write_all(fh, topOpt->springs.data(), topOpt->springs.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);

  // Writing mass node array
  MPI_File_open(topOpt->comm, "massNodes.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY |
                                               MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
  MPI_File_close(&fh);
  MPI_File_open(topOpt->comm, "massNodes.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_Wait(&massreq, MPI_STATUS_IGNORE);
  massdist -= topOpt->massNode.rows();
  MPI_File_seek(fh, massdist*sizeof(PetscInt), MPI_SEEK_SET);
  global_int.resize(topOpt->massNode.rows(), 1);
  for (int i = 0; i < topOpt->massNode.size(); i++)
    global_int(i, 0) = topOpt->gNode(topOpt->massNode(i));
  MPI_File_write_all(fh, global_int.data(), global_int.size(), MPI_PETSCINT, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);

  // Writing masses array
  MPI_File_open(topOpt->comm, "masses.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY |
                                                 MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &fh);
  MPI_File_close(&fh);
  MPI_File_open(topOpt->comm, "masses.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_File_seek(fh, massdist*topOpt->masses.cols()*sizeof(double), MPI_SEEK_SET);
  MPI_File_write_all(fh, topOpt->masses.data(), topOpt->masses.size(), MPI_DOUBLE, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);

  // Writing filter
  PetscErrorCode ierr;
  PetscViewer view;
  ierr = PetscViewerBinaryOpen(topOpt->comm, "Filter.bin", FILE_MODE_WRITE, &view); CHKERRQ(ierr);
  ierr = MatView(topOpt->P, view); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view); CHKERRQ(ierr);

  // Writing projecting matrices
  ArrayXPI lcol(topOpt->nprocs);
  for (unsigned int i = 0; i < topOpt->PR.size(); i++)
  {
    stringstream level; level << i;
    string filename = "P" + level.str() + ".bin";
    ierr = PetscViewerBinaryOpen(topOpt->comm, filename.c_str(), FILE_MODE_WRITE, &view); CHKERRQ(ierr);
    ierr = MatView(topOpt->PR[i], view); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view); CHKERRQ(ierr);
    ierr = MatGetLocalSize(topOpt->PR[i], NULL, lcol.data()+topOpt->myid); CHKERRQ(ierr);
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_PETSCINT, lcol.data(), 1, MPI_PETSCINT, topOpt->comm);
    if (topOpt->myid == 0)
    {
      filename += ".split";
      file.open(filename.c_str(), ios::binary);
      file.write((char*)lcol.data(), lcol.size()*sizeof(PetscInt));
      file.close();
    }
  }
  
  return 0;
}

int StepOut ( TopOpt *topOpt, const double &f, const Eigen::VectorXd &cons, int it )
{
  PetscErrorCode ierr = 0;

  ierr = PetscFPrintf(topOpt->comm, topOpt->output, "Iteration number: %u\tObjective: %1.6g\n",
              it, f); CHKERRQ(ierr);
  ierr = PetscFPrintf(topOpt->comm, topOpt->output, "Constraints:\n"); CHKERRQ(ierr);
  for (short i = 0; i < cons.size(); i++)
  {
    ierr = PetscFPrintf(topOpt->comm, topOpt->output, "%1.12g\t", cons(i)); CHKERRQ(ierr);
  }
  ierr = PetscFPrintf(topOpt->comm, topOpt->output, "\n\n"); CHKERRQ(ierr);

  return 0;
}

int ResultOut ( TopOpt *topOpt, int it )
{
  PetscErrorCode ierr = 0;

  // Output a ratio of stiffness to volume
  PetscScalar Esum, Vsum;
  ierr = VecSum(topOpt->E, &Esum); CHKERRQ(ierr);
  ierr = VecSum(topOpt->V, &Vsum); CHKERRQ(ierr);
  ierr = PetscFPrintf(topOpt->comm, topOpt->output, "************************************************\n");
  ierr = PetscFPrintf(topOpt->comm, topOpt->output, "After %4i iterations with a penalty of %1.4g the\n",
              it, topOpt->penal); CHKERRQ(ierr);
  ierr = PetscFPrintf(topOpt->comm, topOpt->output, "ratio of stiffness sum to volume sum is %1.4g\n",
              Esum/Vsum); CHKERRQ(ierr);
  ierr = PetscFPrintf(topOpt->comm, topOpt->output, "************************************************\n\n");

  stringstream pen;
  pen << topOpt->penal;
  PetscViewer output;

  string filename = "U_pen" + pen.str() + ".bin";
  ierr = PetscViewerBinaryOpen(topOpt->comm, filename.c_str(),
      FILE_MODE_WRITE, &output); CHKERRQ(ierr);
  ierr = VecView(topOpt->U, output); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&output); CHKERRQ(ierr);

  filename = "x_pen" + pen.str() + ".bin";
  ierr = PetscViewerBinaryOpen(topOpt->comm, filename.c_str(),
      FILE_MODE_WRITE, &output); CHKERRQ(ierr);
  ierr = VecView(topOpt->x, output); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&output); CHKERRQ(ierr);

  filename = "V_pen" + pen.str() + ".bin";
  ierr = PetscViewerBinaryOpen(topOpt->comm, filename.c_str(),
      FILE_MODE_WRITE, &output); CHKERRQ(ierr);
  ierr = VecView(topOpt->V, output); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&output); CHKERRQ(ierr);

  filename = "E_pen" + pen.str() + ".bin";
  ierr = PetscViewerBinaryOpen(topOpt->comm, filename.c_str(),
      FILE_MODE_WRITE, &output); CHKERRQ(ierr);
  ierr = VecView(topOpt->E, output); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&output); CHKERRQ(ierr);

  for (int i = 0; i < topOpt->bucklingShape.cols(); i++)
  {
    stringstream mode;  mode << i;
    filename = "phiB" + mode.str() + "_pen" + pen.str() + ".bin";
    Vec phi;
    ierr = VecCreateMPIWithArray(topOpt->comm, 1, topOpt->numDims*topOpt->nLocNode,
        topOpt->numDims*topOpt->nNode, topOpt->bucklingShape.data() +
        topOpt->bucklingShape.rows()*i, &phi); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(topOpt->comm, filename.c_str(),
        FILE_MODE_WRITE, &output); CHKERRQ(ierr);
    ierr = VecView(phi, output); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&output); CHKERRQ(ierr);
    ierr = VecDestroy(&phi); CHKERRQ(ierr);
  }

  for (int i = 0; i < topOpt->dynamicShape.cols(); i++)
  {
    stringstream mode;  mode << i;
    filename = "phiD" + mode.str() + "_pen" + pen.str() + ".bin";
    Vec phi;
    ierr = VecCreateMPIWithArray(topOpt->comm, 1, topOpt->numDims*topOpt->nLocNode,
        topOpt->numDims*topOpt->nNode, topOpt->dynamicShape.data() +
        topOpt->dynamicShape.rows()*i, &phi); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(topOpt->comm, filename.c_str(),
        FILE_MODE_WRITE, &output); CHKERRQ(ierr);
    ierr = VecView(phi, output); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&output); CHKERRQ(ierr);
    ierr = VecDestroy(&phi); CHKERRQ(ierr);
  }

  return 0;
}
