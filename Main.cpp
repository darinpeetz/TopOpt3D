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
    PetscErrorCode ierr;
    SlepcInitialize(&argc,&args,(char*)0,help);
    MPI_Comm Opt_Comm = MPI_COMM_WORLD;
    MPI_Comm_rank(Opt_Comm, &myid);
    MPI_Comm_size(Opt_Comm, &nproc);

    /// Optimization Variables
    TopOpt * topOpt = new TopOpt;
    MMA * optmma = new MMA;
    optmma->Set_Comm(Opt_Comm);

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
    int mg_levels = 1;
    topOpt->Def_Param(optmma, topOpt, Dimensions, Nel, R, Normalization, Reorder_Mesh, mg_levels);
    topOpt->Set_Funcs();

    /// Domain and Boundary Conditions
    ierr = topOpt->CreateMesh(topOpt, Dimensions, Nel, R, Reorder_Mesh, mg_levels); CHKERRQ(ierr);
    
    /// Revaluate number of elements and size in each dimension
    Eigen::VectorXd newDims(2*topOpt->numDims);
    for (int i = 0; i < topOpt->numDims; i++)
    {
      newDims(i) = topOpt->node.col(i).minCoeff();
      newDims(topOpt->numDims+i) = topOpt->node.col(i).maxCoeff();
    }
    MPI_Allreduce(MPI_IN_PLACE, newDims.data(), topOpt->numDims,
                  MPI_DOUBLE, MPI_MIN, Opt_Comm);
    MPI_Allreduce(MPI_IN_PLACE, newDims.data()+topOpt->numDims,
                  topOpt->numDims, MPI_DOUBLE, MPI_MAX, Opt_Comm);
    for (int i = 0; i < topOpt->numDims; i++)
      Nel(i) *= (newDims(topOpt->numDims+i)-newDims(i))/(Dimensions(2*i+1)-Dimensions(2*i));
    topOpt->Def_BC();

    MeshOut( topOpt );
    /// Design Variable Initialization
    optmma->Set_Lower_Bound( Eigen::VectorXd::Constant(topOpt->nLocElem, 0) );
    optmma->Set_Upper_Bound( Eigen::VectorXd::Ones(topOpt->nLocElem) );
    Eigen::VectorXd zIni = 0.5*Eigen::VectorXd::Ones(topOpt->nLocElem);
    optmma->Set_Init_Values( zIni );
    optmma->Set_n( topOpt->nLocElem );

    /// Optimize
    cout.precision(12);
    topOpt->Initialize();
    double f;
    Eigen::VectorXd dfdx, g;
    Eigen::MatrixXd dgdx;
    FILE *values;
    ierr = PetscFOpen(topOpt->comm, "Values.txt", "w", &values); CHKERRQ(ierr);
    ierr = PetscFClose(topOpt->comm, values); CHKERRQ(ierr);

    for ( topOpt->penal = topOpt->pmin; topOpt->penal <= topOpt->pmax; topOpt->penal += topOpt->pstep )
    {
      ierr = PetscFOpen(topOpt->comm, "Values.txt", "a", &values); CHKERRQ(ierr);
      ierr = PetscFPrintf(topOpt->comm, values, "\nPenalty increased to %1.3g\n",
                  topOpt->penal); CHKERRQ(ierr);
      ierr = PetscFClose(topOpt->comm, values); CHKERRQ(ierr);

      optmma->Set_It(0);
      topOpt->MatIntFnc( optmma->Get_x() );
      ierr = PetscLogEventBegin(topOpt->FEEvent, 0, 0, 0, 0); CHKERRQ(ierr);
      topOpt->FESolve();
      ierr = PetscLogEventEnd(topOpt->FEEvent, 0, 0, 0, 0); CHKERRQ(ierr);
      ierr = PetscLogEventBegin(topOpt->funcEvent, 0, 0, 0, 0); CHKERRQ(ierr);
      ierr = Functions::FunctionCall( topOpt, f, dfdx, g, dgdx ); CHKERRQ(ierr);
      ierr = PetscLogEventEnd(topOpt->funcEvent, 0, 0, 0, 0); CHKERRQ(ierr);
      StepOut(topOpt, f, g, optmma->Get_it());

      do
      {
        ierr = PetscLogEventBegin(topOpt->UpdateEvent, 0, 0, 0, 0); CHKERRQ(ierr);
        optmma->Update( dfdx, g, dgdx );
        ierr = PetscLogEventBegin(topOpt->UpdateEvent, 0, 0, 0, 0); CHKERRQ(ierr);
        topOpt->MatIntFnc( optmma->Get_x() );
        ierr = PetscLogEventBegin(topOpt->FEEvent, 0, 0, 0, 0); CHKERRQ(ierr);
        topOpt->FESolve();
        ierr = PetscLogEventEnd(topOpt->FEEvent, 0, 0, 0, 0); CHKERRQ(ierr);
        ierr = PetscLogEventBegin(topOpt->funcEvent, 0, 0, 0, 0); CHKERRQ(ierr);
        ierr = Functions::FunctionCall( topOpt, f, dfdx, g, dgdx ); CHKERRQ(ierr);
        ierr = PetscLogEventEnd(topOpt->funcEvent, 0, 0, 0, 0); CHKERRQ(ierr);
        StepOut(topOpt, f, g, optmma->Get_it());
      } while ( !optmma->Check() );

      /// Print result after this penalization
      ResultOut(topOpt, optmma->Get_it());
    }

    /// Print out all function values if desired
    if (Normalization)
    {
      double value; double *grad = NULL;
      PetscPrintf(topOpt->comm, "***Final Values***\n");

      ierr = Functions::Compliance( topOpt, value, grad ); CHKERRQ(ierr);
      PetscPrintf(topOpt->comm, "\tCompliance: %1.12g\n", value);

      ierr = Functions::Perimeter( topOpt, value, grad ); CHKERRQ(ierr);
      PetscPrintf(topOpt->comm, "\tPerimeter: %1.12g\n", value);

      ierr = Functions::Volume( topOpt, value, grad ); CHKERRQ(ierr);
      PetscPrintf(topOpt->comm, "\tVolume: %1.12g\n", value);

      PetscInt nevals = 1;
      ierr = Functions::Buckling( topOpt, &value, grad, nevals ); CHKERRQ(ierr);
      PetscPrintf(topOpt->comm, "\tBuckling: %1.12g\n", value);

      nevals = 1;
      ierr = Functions::Dynamic( topOpt, &value, grad, nevals ); CHKERRQ(ierr);
      PetscPrintf(topOpt->comm, "\tFrequency: %1.12g\n", value);
    }

    /// Wrap up and finish
    delete topOpt;
    delete optmma;

    ierr = SlepcFinalize(); CHKERRQ(ierr);

    return ierr;
}

int MeshOut ( TopOpt *topOpt )
{
  stringstream strmid;
  strmid << topOpt->myid;

  string filename = "elements" + strmid.str() + ".bin";
  ofstream file(filename.c_str(), ios::binary);
  for (int el = 0; el < topOpt->nLocElem; el++)
  {
    for (int nd = 0; nd < topOpt->element.cols(); nd++)
      file.write((char*)(topOpt->gNode.data()+topOpt->element(el,nd)), sizeof(PetscInt));
  }
  file.close();

  filename = "nodes" + strmid.str() + ".bin";
  file.open(filename.c_str(), ios::binary);
  file.write((char*)topOpt->node.data(), topOpt->nLocNode*topOpt->node.cols()*sizeof(double));
  file.close();

  filename = "edges" + strmid.str() + ".bin";
  file.open(filename.c_str(), ios::binary);
  for (int el = 0; el < topOpt->edgeElem.rows(); el++)
  {
    if (topOpt->edgeElem(el,1) < topOpt->gElem.size())
    {
      file.write((char*)(topOpt->gElem.data()+topOpt->edgeElem(el,0)), sizeof(PetscInt));
      file.write((char*)(topOpt->gElem.data()+topOpt->edgeElem(el,1)), sizeof(PetscInt));
    }
    else
    {
      file.write((char*)(topOpt->gElem.data()+topOpt->edgeElem(el,0)), sizeof(PetscInt));
      file.write((char*)&topOpt->nElem, sizeof(PetscInt));
    }
  }
  file.close();
  filename = "edgeLengths" + strmid.str() + ".bin";
  file.open(filename.c_str(), ios::binary);
  file.write((char*)topOpt->edgeSize.data(), topOpt->edgeElem.rows()*sizeof(double));
  file.close();

  filename = "loadNodes" + strmid.str() + ".bin";
  file.open(filename.c_str(), ios::binary);
  for (int i = 0; i < topOpt->loadNode.size(); i++)
    file.write((char*)(topOpt->gNode.data()+topOpt->loadNode(i)), sizeof(PetscInt));
  file.close();
  filename = "loads" + strmid.str() + ".bin";
  file.open(filename.c_str(), ios::binary);
  file.write((char*)topOpt->loads.data(), topOpt->loads.size()*sizeof(double));
  file.close();

  filename = "supportNodes" + strmid.str() + ".bin";
  file.open(filename.c_str(), ios::binary);
  for (int i = 0; i < topOpt->suppNode.size(); i++)
    file.write((char*)(topOpt->gNode.data()+topOpt->suppNode(i)), sizeof(PetscInt));
  file.close();
  filename = "supports" + strmid.str() + ".bin";
  file.open(filename.c_str(), ios::binary);
  file.write((char*)topOpt->supports.data(), topOpt->supports.size()*sizeof(bool));
  file.close();

  filename = "springNodes" + strmid.str() + ".bin";
  file.open(filename.c_str(), ios::binary);
  for (int i = 0; i < topOpt->springNode.size(); i++)
    file.write((char*)(topOpt->gNode.data()+topOpt->springNode(i)), sizeof(PetscInt));
  file.close();
  filename = "springs" + strmid.str() + ".bin";
  file.open(filename.c_str(), ios::binary);
  file.write((char*)topOpt->springs.data(), topOpt->springs.size()*sizeof(double));
  file.close();

  filename = "massNodes" + strmid.str() + ".bin";
  file.open(filename.c_str(), ios::binary);
  for (int i = 0; i < topOpt->massNode.size(); i++)
    file.write((char*)(topOpt->gNode.data()+topOpt->massNode(i)), sizeof(PetscInt));
  file.close();
  filename = "masses" + strmid.str() + ".bin";
  file.open(filename.c_str(), ios::binary);
  file.write((char*)topOpt->masses.data(), topOpt->masses.size()*sizeof(double));
  file.close();

  return 0;
}

int StepOut ( TopOpt *topOpt, const double &f, const Eigen::VectorXd &cons, int it )
{
  PetscErrorCode ierr;

  FILE *values;
  ierr = PetscFOpen(topOpt->comm, "Values.txt", "a", &values); CHKERRQ(ierr);
  ierr = PetscFPrintf(topOpt->comm, values, "Iteration number: %u\tObjective: %1.6g\n",
              it, f); CHKERRQ(ierr);
  ierr = PetscFPrintf(topOpt->comm, values, "Constraints:\n"); CHKERRQ(ierr);
  for (short i = 0; i < cons.size(); i++)
  {
    ierr = PetscFPrintf(topOpt->comm, values, "%1.12g\t", cons(i)); CHKERRQ(ierr);
  }
  ierr = PetscFPrintf(topOpt->comm, values, "\n\n"); CHKERRQ(ierr);
  ierr = PetscFClose(topOpt->comm, values); CHKERRQ(ierr);

  return 0;
}

int ResultOut ( TopOpt *topOpt, int it )
{
  PetscErrorCode ierr;

  // Output a ratio of stiffness to volume
  PetscScalar Esum, Vsum;
  ierr = VecSum(topOpt->E, &Esum); CHKERRQ(ierr);
  ierr = VecSum(topOpt->V, &Vsum); CHKERRQ(ierr);
  FILE *values;
  ierr = PetscFOpen(topOpt->comm, "Values.txt", "a", &values); CHKERRQ(ierr);
  ierr = PetscFPrintf(topOpt->comm, values, "*********************************************\n");
  ierr = PetscFPrintf(topOpt->comm, values, "After %4i iterations with a penalty of %5.4g the\n",
              it, topOpt->penal); CHKERRQ(ierr);
  ierr = PetscFPrintf(topOpt->comm, values, "ratio of stiffness sum to volume sum is %5.4g\n",
              Esum/Vsum); CHKERRQ(ierr);
  ierr = PetscFPrintf(topOpt->comm, values, "*********************************************\n\n");
  ierr = PetscFClose(topOpt->comm, values); CHKERRQ(ierr);

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
