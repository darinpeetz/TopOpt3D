#include <iostream>
#include <cmath>
#include <fstream>
#include <ctime>
#include <Eigen/Eigen>
#include <numeric>
#include <unsupported/Eigen/KroneckerProduct>
#include "Inputs.h"
#include "EigLab.h"

using namespace std;

typedef Eigen::Array<PetscInt, -1, -1> ArrayXXPI;
typedef Eigen::Array<PetscInt, -1,  1> ArrayXPI;
typedef Eigen::Array<PetscInt,  1, -1> RowArrayXPI;
typedef Eigen::Array<PetscScalar, -1,  1> ArrayXPS;
typedef Eigen::Matrix<PetscScalar, -1, -1> MatrixXPS;
typedef Eigen::Array<PetscInt, -1, -1, Eigen::RowMajor> ArrayXXPIRM;
typedef Eigen::Matrix<double,-1,-1,Eigen::RowMajor> MatrixXdRM;


/*****************************************************************/
/**                    Load Mesh for Restart                    **/
/*****************************************************************/
PetscErrorCode TopOpt::LoadMesh(Eigen::VectorXd &xIni)
{
  PetscErrorCode ierr = 0;

  ifstream input;
  PetscInt filesize = 0;
  string filename;

  // Read in element distribution
  filename = folder + "/Element_Distribution.bin";
  input.open(filename.c_str(), ios::ate | ios::binary);
  if (!input.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to open element distribution file");
  filesize = input.tellg();
  input.seekg(0);
  elmdist.resize(filesize/sizeof(PetscInt));
  input.read((char*)elmdist.data(), filesize);
  input.close();

  // Read in node distribution
  filename = folder + "/Node_Distribution.bin";
  input.open(filename.c_str(), ios::ate | ios::binary);
  if (!input.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to open node distribution file");
  filesize = input.tellg();
  input.seekg(0);
  nddist.resize(filesize/sizeof(PetscInt));
  input.read((char*)nddist.data(), filesize);
  input.close();

  // Read in elements
  filename = folder + "/elements.bin";
  input.open(filename.c_str(), ios::ate | ios::binary);
  if (!input.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to open elements file");
  filesize = input.tellg();
  SetDimension(log2(filesize/sizeof(PetscInt)/elmdist(elmdist.size()-1)));
  input.seekg(elmdist(myid)*pow(numDims,2)*sizeof(PetscInt));
  element.resize(elmdist(myid+1)-elmdist(myid), pow(numDims,2));
  input.read((char*)element.data(), element.size()*sizeof(PetscInt));
  input.close();

  // Read in nodes
  filename = folder + "/nodes.bin";
  input.open(filename.c_str(), ios::ate | ios::binary);
  if (!input.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to open nodes file");
  filesize = input.tellg();
  input.seekg(nddist(myid)*numDims*sizeof(PetscScalar));
  node.resize(nddist(myid+1)-nddist(myid), numDims);
  input.read((char*)node.data(), node.size()*sizeof(PetscScalar));
  input.close();

  // Create constitutive matrix
  switch (this->numDims)
  {
    case 1:
      this->d.resize(1,1);
      this->d << this->E0;
      break;
    case 2:
      this->d.resize(3,3);
      this->d << 1, this->Nu0, 0 , this->Nu0, 1, 0 , 0, 0, (1-this->Nu0)/2;
      this->d = this->E0/(1-this->Nu0*Nu0)*this->d;
      break;
    case 3:
      this->d.resize(6,6);
      double c = this->E0/((1+this->Nu0)*(1-2*this->Nu0));
      double G = this->E0/(2*(1+this->Nu0));
      double t1 = (1-this->Nu0)*c;
      double t2 = G;
      double t3 = this->Nu0*c;
      this->d << t1, t3, t3, 0, 0, 0, t3, t1, t3, 0, 0, 0, t3, t3, t1, 0, 0, 0,
                0, 0, 0, t2, 0, 0, 0, 0, 0, 0, t2, 0, 0, 0, 0, 0, 0, t2;
      break;
  }

  // Read in edge information
  // First read in all edge element information
  filename = folder + "/edges.bin";
  input.open(filename.c_str(), ios::ate | ios::binary);
  if (!input.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to open edges file");
  filesize = input.tellg();
  input.seekg(0);
  nEdges = filesize/sizeof(PetscInt)/2;
  ArrayXXPIRM temp_edge(nEdges, 2);
  input.read((char*)temp_edge.data(), filesize);
  input.close();
  // Now find what part is local and extract it
  PetscInt begin = 0, finish = temp_edge.rows();
  for (int i = 0; i < temp_edge.rows(); i++)
  {
    if (temp_edge(i,0) < elmdist(myid))
      begin++;
    if (temp_edge(i,0) >= elmdist(myid+1))
    {
      finish = i;
      break;
    }
  }
  edgeElem = temp_edge.block(begin, 0, finish-begin, 2);
  // Now get edge lengths
  filename = folder + "/edgeLengths.bin";
  input.open(filename.c_str(), ios::ate | ios::binary);
  if (!input.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to open edge lengths file");
  filesize = input.tellg();
  input.seekg(begin*sizeof(PetscScalar));
  edgeSize.resize(edgeElem.rows());
  input.read((char*)edgeSize.data(), edgeSize.size()*sizeof(PetscScalar));
  input.close();

  // Read in BC's
  // Similar to edge information, have to read it all in and parse it later
  // Loads first
  filename = folder + "/loadNodes.bin";
  input.open(filename.c_str(), ios::ate | ios::binary);
  if (!input.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to load nodes file");
  filesize = input.tellg();
  input.seekg(0);
  ArrayXPI temp_node(filesize/sizeof(PetscInt));
  input.read((char*)temp_node.data(), filesize);
  input.close();
  // Now find what part is local and extract it
  begin = 0; finish = temp_node.rows();
  for (int i = 0; i < temp_node.rows(); i++)
  {
    if (temp_node(i) < nddist(myid))
      begin++;
    if (temp_node(i) >= nddist(myid+1))
    {
      finish = i;
      break;
    }
  }
  loadNode = temp_node.segment(begin, finish-begin) -= nddist(myid);
  // Now the load quantities
  filename = folder + "/loads.bin";
  input.open(filename.c_str(), ios::ate | ios::binary);
  if (!input.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to open loads file");
  filesize = input.tellg();
  input.seekg(begin*sizeof(PetscScalar)*numDims);
  loads.resize(loadNode.size(), numDims);
  input.read((char*)loads.data(), loads.size()*sizeof(PetscScalar));
  input.close();

  // Masses
  filename = folder + "/massNodes.bin";
  input.open(filename.c_str(), ios::ate | ios::binary);
  if (!input.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to mass nodes file");
  filesize = input.tellg();
  input.seekg(0);
  temp_node.resize(filesize/sizeof(PetscInt));
  input.read((char*)temp_node.data(), filesize);
  input.close();
  // Now find the local part
  begin = 0; finish = temp_node.rows();
  for (int i = 0; i < temp_node.rows(); i++)
  {
    if (temp_node(i) < nddist(myid))
      begin++;
    if (temp_node(i) >= nddist(myid+1))
    {
      finish = i;
      break;
    }
  }
  massNode = temp_node.segment(begin, finish-begin) -= nddist(myid);
  // Load the masses
  filename = folder + "/masses.bin";
  input.open(filename.c_str(), ios::ate | ios::binary);
  if (!input.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to open masses file");
  filesize = input.tellg();
  input.seekg(begin*sizeof(PetscScalar)*numDims);
  masses.resize(massNode.size(), numDims);
  input.read((char*)masses.data(), masses.size()*sizeof(PetscScalar));
  input.close();

  // Springs
  filename = folder + "/springNodes.bin";
  input.open(filename.c_str(), ios::ate | ios::binary);
  if (!input.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to spring nodes file");
  filesize = input.tellg();
  input.seekg(0);
  temp_node.resize(filesize/sizeof(PetscInt));
  input.read((char*)temp_node.data(), filesize);
  input.close();
  // Now find the local part
  begin = 0; finish = temp_node.rows();
  for (int i = 0; i < temp_node.rows(); i++)
  {
    if (temp_node(i) < nddist(myid))
      begin++;
    if (temp_node(i) >= nddist(myid+1))
    {
      finish = i;
      break;
    }
  }
  springNode = temp_node.segment(begin, finish-begin) -= nddist(myid);
  // Load the springs
  filename = folder + "/springs.bin";
  input.open(filename.c_str(), ios::ate | ios::binary);
  if (!input.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to open springs file");
  filesize = input.tellg();
  input.seekg(begin*sizeof(PetscScalar)*numDims);
  springs.resize(springNode.size(), numDims);
  input.read((char*)springs.data(), springs.size()*sizeof(PetscScalar));
  input.close();

  // Fixed Supports
  filename = folder + "/supportNodes.bin";
  input.open(filename.c_str(), ios::ate | ios::binary);
  if (!input.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to support nodes file");
  filesize = input.tellg();
  input.seekg(0);
  temp_node.resize(filesize/sizeof(PetscInt));
  input.read((char*)temp_node.data(), filesize);
  input.close();
  // Now find the local part
  begin = 0; finish = temp_node.rows();
  for (int i = 0; i < temp_node.rows(); i++)
  {
    if (temp_node(i) < nddist(myid))
      begin++;
    if (temp_node(i) >= nddist(myid+1))
    {
      finish = i;
      break;
    }
  }
  suppNode = temp_node.segment(begin, finish-begin) -= nddist(myid);
  // Load the supports
  filename = folder + "/supports.bin";
  input.open(filename.c_str(), ios::ate | ios::binary);
  if (!input.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to open supports file");
  filesize = input.tellg();
  input.seekg(begin*sizeof(bool)*numDims);
  supports.resize(suppNode.size(), numDims);
  input.read((char*)supports.data(), supports.size()*sizeof(bool));
  input.close();

  /// Establish Global Numbering
  nLocElem = elmdist(myid+1)-elmdist(myid);
  nElem = elmdist(nprocs);
  gElem = ArrayXPI::LinSpaced(nLocElem, elmdist(myid), elmdist(myid+1)-1);
  nLocNode = nddist(myid+1)-nddist(myid);
  nNode = nddist(nprocs);
  gNode = ArrayXPI::LinSpaced(nLocNode, nddist(myid), nddist(myid+1)-1);

  /// Get any needed ghost information
  Expand_Elem();
  Expand_Node();

  /// Assign Ghost Info and create DV vectors
  Initialize_Vectors();

  /// Local Element Numbering
  Localize();

  // Element sizes
  double temp = node(element(0,1),0) - node(element(0,0),0);
  if (numDims > 1)
    temp *= node(element(0,3),1) - node(element(0,0),1);
  if (numDims > 2)
    temp *= node(element(0,7),2) - node(element(0,0),2);
  elemSize.setConstant(nLocElem, temp);
  // Perimeter normalization factor
  PerimNormFactor = 0;
  PerimNormFactor += elemSize(0)/(node(element(0,1),0) - node(element(0,0),0));
  if (numDims > 1)
    PerimNormFactor += elemSize(0)/(node(element(0,3),1) - node(element(0,0),1));
  if (numDims > 2)
    PerimNormFactor += elemSize(0)/(node(element(0,7),2) - node(element(0,0),2));

  // Read in the filter matrix
  PetscViewer view;
  filename = folder + "/Filter.bin";
  input.open(filename.c_str(), ios::ate);
  if (!input.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to open filter file");
  input.close();
  ierr = PetscViewerBinaryOpen(comm, filename.c_str(), FILE_MODE_READ, &view);
  ierr = MatCreate(comm, &this->P); CHKERRQ(ierr);
  ierr = MatSetType(this->P, MATAIJ); CHKERRQ(ierr);
  ierr = MatSetSizes(this->P, nLocElem, nLocElem, nElem, nElem); CHKERRQ(ierr);
  ierr = MatLoad(this->P, view); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);

  // Read in the multigrid hierarchy
  this->PR.resize(0);
  this->MG_comms.resize(0);
  this->MG_comms.push_back(comm);
  PetscInt lrow = 2*nddist(myid+1)-2*nddist(myid), lcol;
  for (int level = 0; ; level++)
  {
    stringstream strmlvl;
    strmlvl << level;
    filename = folder + "/P" + strmlvl.str() + ".bin";
    input.open(filename.c_str(), ios::ate);
    if (!input.is_open())
      break;
    input.close();
    this->PR.resize(level+1);
    ierr = PetscViewerBinaryOpen(comm, filename.c_str(), FILE_MODE_READ, &view); CHKERRQ(ierr);
    ierr = MatCreate(comm, this->PR.data()+level); CHKERRQ(ierr);
    ierr = MatSetType(this->PR[level], MATAIJ); CHKERRQ(ierr);
    // Properly set local matrix dimensions
    filename += ".split";
    input.open(filename.c_str(), ios::binary);
    input.seekg(myid*sizeof(PetscInt));
    input.read((char*)&lcol, sizeof(PetscInt));
    input.close();
    ierr = MatSetSizes(this->PR[level], lrow, lcol, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);
    ierr = MatLoad(this->PR[level], view); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view);
    lrow = lcol;
    this->MG_comms.push_back(comm);
  }
                                                                                                     
  // Initial design values
  xIni.setOnes(nLocElem); xIni *= 0.5;
  ierr = VecPlaceArray(this->x, xIni.data()); CHKERRQ(ierr); 
  for (penal = pmin; penal <= pmax; penal += pstep)
  {
    stringstream strmid; strmid << penal;
    filename = folder + "/x_pen" + strmid.str() + ".bin";
    input.open(filename.c_str(), ios::binary);
    if (!input.is_open())
      break;
    input.close();
    ierr = PetscViewerBinaryOpen(comm, filename.c_str(), FILE_MODE_READ, &view); CHKERRQ(ierr);
    ierr = VecLoad(this->x, view); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view); CHKERRQ(ierr);
  }
  ierr = VecResetArray(this->x); CHKERRQ(ierr);

  // Finally note that the mesh is uniform quadrilaterals
  regular = true;

  return ierr;
}
  
/*****************************************************************/
/**                      Create Base Mesh                       **/
/*****************************************************************/
PetscErrorCode TopOpt::CreateMesh ( Eigen::VectorXd dimensions, ArrayXPI Nel,
                                    double R, bool Reorder_Mesh, int mg_levels,
                                    PetscInt min_size )
{
  PetscErrorCode ierr = 0;

  this->SetDimension(Nel.size());
  Nel.conservativeResize(3);
  for (int i = numDims; i < 3; i++)
    Nel(i) = 1;
  regular = 1;

  /// Constitutive matrix
  this->numDims = dimensions.size()/2; // I think this is redundant to the SetDimensions call above
  switch (this->numDims)
  {
    case 1:
      this->d.resize(1,1);
      this->d << this->E0;
      break;
    case 2:
      this->d.resize(3,3);
      this->d << 1, this->Nu0, 0 , this->Nu0, 1, 0 , 0, 0, (1-this->Nu0)/2;
      this->d = this->E0/(1-this->Nu0*Nu0)*this->d;
      break;
    case 3:
      this->d.resize(6,6);
      double c = this->E0/((1+this->Nu0)*(1-2*this->Nu0));
      double G = this->E0/(2*(1+this->Nu0));
      double t1 = (1-this->Nu0)*c;
      double t2 = G;
      double t3 = this->Nu0*c;
      this->d << t1, t3, t3, 0, 0, 0, t3, t1, t3, 0, 0, 0, t3, t3, t1, 0, 0, 0,
                0, 0, 0, t2, 0, 0, 0, 0, 0, 0, t2, 0, 0, 0, 0, 0, 0, t2;
      break;
  }

  /// Create the elements - Initially distribute by slicing in highest
  /// dimension, therefore it is a good idea to make the highest numbered
  /// dimension be the one with the most elements
  // first and last are the first element on this process and one plus the last 
  // element this process in every dimension
  PetscInt first[3] = {0, 0, 0};
  PetscInt last[3] = {1, 1, 1};
  // Set the last element of lower dimensions
  for (short dim = 0; dim < this->numDims-1; dim++)
  {
    last[dim] = Nel(dim);
  }
  first[this->numDims-1] = ((double)myid/nprocs)*Nel(this->numDims-1);
  last[this->numDims-1] = ((double)(myid+1)/nprocs)*Nel(this->numDims-1);
  // Number of nodes in each dimension;
  ArrayXPI Nnd = Nel;
  for (int i = 0; i < this->numDims; i++)
    Nnd(i)++;

  // Initialize element array
  this->nElem = Nel(0)*Nel(1)*Nel(2);
  this->nLocElem = (last[0]-first[0])*(last[1]-first[1])*(last[2]-first[2]);
  this->element.resize( this->nLocElem, pow(2,this->numDims) );
  for (PetscInt layer = first[2]; layer < last[2]; layer++)
  { // Loop through z-dimension
    for (PetscInt row = first[1]; row < last[1]; row++)
    { // Loop through y-dimension
      // Add first node to element
      this->element.block((layer-first[2])*(Nel(0)*Nel(1)) +
          (row-first[1])*(Nel(0)), 0, last[0]-first[0], 1) =
        ArrayXPI::LinSpaced(last[0]-first[0],
          layer*(Nnd(0)*Nnd(1)) + row*(Nnd(0)) + first[0],
          layer*(Nnd(0)*Nnd(1)) + row*(Nnd(0)) + last[0]-1);
      // Add second node to element
      this->element.block((layer-first[2])*(Nel(0)*Nel(1)) +
          (row-first[1])*(Nel(0)), 1, last[0]-first[0], 1) =
        ArrayXPI::LinSpaced(last[0]-first[0],
          layer*(Nnd(0)*Nnd(1)) + row*(Nnd(0)) + first[0]+1,
          layer*(Nnd(0)*Nnd(1)) + row*(Nnd(0)) + last[0]);
    }
  }

  // Add third and fourth nodes to all elements for 2D or 3D analysis
  if (this->numDims > 1)
  {
    this->element.col(2) = this->element.col(1) + Nnd(0);
    this->element.col(3) = this->element.col(0) + Nnd(0);
  }
  // Add nodes 5 through 8 to all elements for 3D analysis
  if (this->numDims > 2)
  {
    this->element.block(0, 4, this->element.rows(), 4) =
      this->element.block(0, 0, this->element.rows(), 4) + (Nnd(0)*Nnd(1));
  }

  /// Element Distribution Information
  this->elmdist.setZero(nprocs+1);
  this->elmdist(myid) = first[0] + first[1]*Nel(0) + first[2]*Nel(0)*Nel(1);
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_PETSCINT, elmdist.data(), 1, MPI_PETSCINT, comm);
  elmdist(nprocs) = nElem;

  /// Create the filter
  // dx is edgelength of elements in each direction
  double dx[3] = {0, 0, 0};
  elemSize.setOnes(1);
  for (int i = 0; i < this->numDims; i++)
  {
    dx[i] = (dimensions(2*i+1) - dimensions(2*i))/Nel(i);
    elemSize *= dx[i];
  }
  FilterArrays filterArrays;
  RecFilter ( first, last, dx, R, Nel, filterArrays );

  // Create the geometric coarse-grid restrictions
  ArrayXPI I[mg_levels-1], J[mg_levels-1], cList[mg_levels-1];
  Eigen::Array<PetscScalar, -1, 1> K[mg_levels-1];
  ierr = Create_Interpolations(first, last, Nel, I, J, K, cList, mg_levels);
  
  /// Create the nodes
  // start by changing ranges to reflect which nodes go on each process
  if (myid != 0)
    first[this->numDims-1]++;

  // Tracking how the nodes are distributed
  this->nNode = Nnd(0)*Nnd(1)*Nnd(2);
  this->nLocNode = 1;
  for (int i = 0; i < this->numDims; i++)
    this->nLocNode *= (last[i] - first[i] + 1);
  this->nddist.setZero(nprocs+1);
  this->nddist(myid) = first[0] + first[1]*Nnd[0] + first[2]*Nnd(0)*Nnd(1);
  MPI_Allreduce(MPI_IN_PLACE, this->nddist.data()+1, nprocs, MPIU_INT, MPI_MAX, comm);
  this->nddist(nprocs) = this->nNode;

  for (int i = this->numDims; i < 3; i++)
    first[i]++;
  PetscInt nLocNode[3] = {last[0]-first[0]+1, last[1]-first[1]+1,
                        last[2]-first[2]+1}; // Local node ranges

  // Creating the nodal coordinates, assign to processes similar to elements
  this->node.resize( this->nLocNode, this->numDims );
  this->node.col(0) = Eigen::VectorXd::LinSpaced(nLocNode[0],dx[0]*first[0]
                        +dimensions(0),dx[0]*last[0]+dimensions(0))
                        .replicate(nLocNode[1],1).replicate(nLocNode[2],1);
  if (this->numDims > 1)
  {
    Eigen::MatrixXd temp = Eigen::RowVectorXd::LinSpaced(nLocNode[(1)],
                           dx[1]*first[1]+dimensions(2),dx[1]*last[1]+dimensions(2))
                           .replicate(nLocNode[0],1);
    temp.resize(nLocNode[0]*nLocNode[1],1);
    this->node.col(1) = temp.replicate(nLocNode[2],1);
  }
  if (this->numDims > 2)
  {
    Eigen::MatrixXd temp = Eigen::RowVectorXd::LinSpaced(nLocNode[2],
                           dx[2]*first[2]+dimensions(4),dx[2]*last[2]+dimensions(4))
                           .replicate(nLocNode[0],1).replicate(nLocNode[1],1);
    temp.resize(this->nLocNode,1);
    this->node.col(2) = temp;
  }

  // Undo changes to range information
  if (myid != 0)
      first[this->numDims-1]--;
  for (int i = this->numDims; i < 3; i++)
    first[i]--;

  // Assemble arrays containing perimeter information
  Edge_Info( first, last, dx );

  /// Apply shape functions to base mesh
  // Start by determining center of every element
  Eigen::ArrayXXd elemCenters = Eigen::ArrayXXd::Zero(this->nLocElem,this->numDims);
  elemCenters.col(0) = (Eigen::VectorXd::LinSpaced(last[0]-first[0],
                       dx[0]*first[0]+dimensions(0),dx[0]*last[0]+dimensions(0)-dx[0])
                       .replicate(last[1]-first[1],1)
                       .replicate(last[2]-first[2],1).array() + dx[0]/2).matrix();
  if (this->numDims > 1)
  {
    Eigen::MatrixXd temp = (Eigen::RowVectorXd::LinSpaced(last[1]-first[1],
                           dx[1]*first[1]+dimensions(2),dx[1]*last[1]+dimensions(2)-dx[1])
                           .replicate(last[0]-first[0],1).array() + dx[1]/2).matrix();
    temp.resize(temp.size(),1);
    elemCenters.col(1) = temp.replicate(last[2]-first[2],1);
  }
  if (this->numDims > 2)
  {
    Eigen::MatrixXd temp = (Eigen::RowVectorXd::LinSpaced(last[2]-first[2],
                           dx[2]*first[2]+dimensions(4),dx[2]*last[2]+dimensions(4)-dx[2])
                           .replicate(last[0]-first[0],1)
                           .replicate(last[1]-first[1],1).array() + dx[2]/2).matrix();
    temp.resize(temp.size(),1);
    elemCenters.col(2) = temp;
  }

  /// Remove unwanted elements
  // Number of ghost elements long processor boundaries
  int padding = 1;
  for (short dim = 0; dim < numDims-1; dim++)
    padding *= Nel(dim);
  // Global validity/numbering array
  Eigen::Array<bool, -1, 1> elemValidity = Eigen::Array<bool, -1, 1>::Ones(nLocElem);
  Domain(elemCenters, dimensions, elemValidity);

  // Check if any process wants to remove elements
  short reductions = (short)elemValidity.all();
  MPI_Allreduce(MPI_IN_PLACE, &reductions, 1, MPI_SHORT, MPI_MIN, comm);

  if (reductions == 0)
  {
    int nInterfaceNodes = 1;
    for (int dim = 1; dim < numDims; dim++)
      nInterfaceNodes *= Nel(dim-1)+1;
    ApplyDomain(elemValidity, padding, nInterfaceNodes, filterArrays, I, J, cList, mg_levels);
  }

  /// Get a better distribution of elements
  ReorderParMetis(filterArrays, Reorder_Mesh);
  filterArrays.Reset(0);
  double temp = elemSize(0);
  elemSize.setConstant(nLocElem, temp);

  /// Node Distribution and Interpolation reordering
  NodeDist(I, J, K, cList, mg_levels);

  /// Interpolation matrix assembly
  ierr = Assemble_Interpolation ( I, J, K, cList, mg_levels, min_size );

  /// Establish Global Numbering
  gElem = ArrayXPI::LinSpaced(this->nLocElem, elmdist(myid), elmdist(myid+1)-1);

  // nLocNode was locally overwritten earlier
  gNode = ArrayXPI::LinSpaced(this->nLocNode, nddist(myid), nddist(myid+1)-1);

  /// Get any needed ghost information
  Expand_Elem();
  Expand_Node();

  /// Assign Ghost Info and create DV vectors
  ierr = Initialize_Vectors(); CHKERRQ(ierr);

  /// Local Element Numbering
  Localize();

  return 0;
}

/*****************************************************************/
/**                          Edge Info                          **/
/*****************************************************************/
PetscErrorCode TopOpt::Edge_Info ( PetscInt *first, PetscInt *last, double *dx )
{
    PetscErrorCode ierr = 0;

    PetscInt Nel[3] = {last[0]-first[0], last[1]-first[1], last[2]-first[2]};
    PerimNormFactor = 0;
    for (short dim = 0; dim < numDims; dim++)
      PerimNormFactor += elemSize(0)/dx[dim];

    if (Nel[0] == 0 || Nel[1] == 0 || Nel[2] == 0)
      return ierr;
    /// dim == 0 => Edges with a normal in the x-direction
    /// dim == 1 => Edges with a normal in the y-direction
    /// dim == 2 => Edges with a normal in the z-direction
    // Element spacing between elements on each side of edge
    int spacing = 1;
    // Curent size of edgeElem
    int numExisting = 0;
    // Size of an element
    for (short dim = 0; dim < numDims; dim++)
    {
      if (dim > 0)
        spacing *= Nel[dim-1];
      if (dim == numDims-1 && myid > 0)
      {
        Nel[dim]--;
        first[dim]++;
      }
      // Edge info along row/column/layer starting at origin
      ArrayXXPI side1 = RowArrayXPI::LinSpaced(
                    Nel[dim]+1, (first[dim]-1)*spacing, (last[dim]-1)*spacing);
      ArrayXXPI side2 = RowArrayXPI::LinSpaced(
                    Nel[dim]+1, first[dim]*spacing, last[dim]*spacing);
      if (dim < numDims-1 || myid == 0)
        side1(0) = nElem;
      if (dim < numDims-1 || myid == nprocs-1)
        side2(Nel[dim]) = nElem;

      if (dim == numDims-1 && myid > 0)
      {
        Nel[dim]++;
        first[dim]--;
      }
      // Spacing between this row/column and the adjacent one in otherDim
      int otherSpacing = 1;
      for (short otherDim = 0; otherDim < numDims; otherDim++)
      {
        if (otherDim == dim)
        {
          side1.resize(side1.size(),1);
          side2.resize(side2.size(),1);
        }
        else
        { // Offset in otherDim direction
          ArrayXXPI offset = Eigen::KroneckerProduct<RowArrayXPI, ArrayXXPI>
                (RowArrayXPI::LinSpaced(
                Nel[otherDim], first[otherDim]*otherSpacing, (last[otherDim]-1)*otherSpacing),
                ArrayXXPI::Ones(side1.rows(),side1.cols()));
          side1 = side1.replicate(1,Nel[otherDim]).eval();
          side2 = side2.replicate(1,Nel[otherDim]).eval();
          side1 += offset;
          side2 += offset;
        }
        otherSpacing *= Nel[otherDim];
      }
      side1.resize(side1.size(),1);
      side2.resize(side2.size(),1);
      edgeElem.conservativeResize(numExisting+side1.rows(),2);
      edgeElem.block(numExisting,0,side1.rows(),1) = side1;
      edgeElem.block(numExisting,1,side2.rows(),1) = side2;
      edgeSize.conservativeResize(numExisting+side1.rows());
      edgeSize.segment(numExisting, side1.rows()) =
          elemSize(0)/dx[dim]*Eigen::VectorXd::Ones(side1.rows());
      numExisting = edgeElem.rows();
    }
    // Restrict maximum element number to nElem;
    edgeElem = edgeElem.min(nElem*ArrayXXPI::Ones(edgeElem.rows(),edgeElem.cols()));

    return ierr;
}

/*****************************************************************/
/**                  Cut Mesh if necessary                      **/
/*****************************************************************/
PetscErrorCode TopOpt::ApplyDomain( Eigen::Array<bool, -1, 1> elemValidity,
                 int padding, int nInterfaceNodes, FilterArrays &filterArrays,
                 ArrayXPI *I, ArrayXPI *J, ArrayXPI *cList, int mg_levels )
{
    PetscErrorCode ierr = 0;
    /// elem validity should be of size nLocElem, padding is the number of
    /// elements along the interfaces between processes, and nInterfaceNodes is
    /// the number of nodes on those interfaces

    /// Renumber local elements
    // Array for new element number and total remaining elements on each process
    ArrayXPI newElemNumber;
    ArrayXPI number = ArrayXPI::Zero(nprocs);
    // Setting the range of local elements amongst all relevant elements
    PetscInt start, finish;
    if (myid == 0)
    {
      start = 0; finish = nLocElem;
      newElemNumber.setZero(nLocElem + padding);
    }
    else if (myid == nprocs-1)
    {
      start = padding; finish = nLocElem+padding;
      newElemNumber.setZero(nLocElem + padding);
    }
    else
    {
      start = padding; finish = nLocElem+padding;
      newElemNumber.setZero(nLocElem + 2*padding);
    }

    for (PetscInt el = 0; el < nLocElem; el++)
    {
      if (elemValidity(el))
      {
        newElemNumber(el+start) = ++number(myid);
      }
    }

    // Share how many are stored locally on this process
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_PETSCINT, number.data(), 1, MPI_PETSCINT, comm);
    PetscInt newnElem = number.sum();
    // Calculate first number on this process (first process starts at 1)
    if (myid > 0)
    {
      newElemNumber.segment(padding, nLocElem) +=
         number.segment(0, myid).sum()*
         (newElemNumber.segment(padding, nLocElem)>0).cast<PetscInt>();
    }

    /// Send and receive new numbers of edge elements to adjacent processes
    MPI_Request sendReq1 = MPI_REQUEST_NULL, sendReq2 = MPI_REQUEST_NULL,
                recReq1 = MPI_REQUEST_NULL, recReq2 = MPI_REQUEST_NULL;
    int sR1 = true, sR2 = true, rR1 = true, rR2 = true;
    if (nLocElem > 0)
    {
      if (myid > 0 && myid != nprocs-1)
      {
        // Upward send
        MPI_Isend(newElemNumber.data()+nLocElem, padding, MPI_PETSCINT,
                    myid+1, 0, comm, &sendReq1);
        // Downward send
        MPI_Isend(newElemNumber.data()+padding, padding, MPI_PETSCINT,
                    myid-1, 1, comm, &sendReq2);
        // Receive from below
        MPI_Irecv(newElemNumber.data(), padding, MPI_PETSCINT,
                    myid-1, 0, comm, &recReq1);
        // Receive from above
        MPI_Irecv(newElemNumber.data()+nLocElem+padding, padding, MPI_PETSCINT,
                    myid+1, 1, comm, &recReq2);
      }
      else if (myid == 0)
      {
        // Upward send
        MPI_Isend(newElemNumber.data()+nLocElem-padding, padding, MPI_PETSCINT,
                    myid+1, 0, comm, &sendReq1);
        // Receive from above
        MPI_Irecv(newElemNumber.data()+nLocElem, padding, MPI_PETSCINT,
                    myid+1, 1, comm, &recReq2);
      }
      else
      {
        // Downward send
        MPI_Isend(newElemNumber.data()+padding, padding, MPI_PETSCINT,
                    myid-1, 1, comm, &sendReq2);
        // Receive from below
        MPI_Irecv(newElemNumber.data(), padding, MPI_PETSCINT,
                    myid-1, 0, comm, &recReq1);
      }

      // Terminate communications before advancing
      do {
        MPI_Test(&sendReq1, &sR1, MPI_STATUS_IGNORE);
        MPI_Test(&sendReq2, &sR2, MPI_STATUS_IGNORE);
        MPI_Test(&recReq1, &rR1, MPI_STATUS_IGNORE);
        MPI_Test(&recReq2, &rR2, MPI_STATUS_IGNORE);
      } while (!(sR1 && sR2 && rR1 && rR2));
    }
    else
    {
      // this process currently owns zero elements - pass information along
      ArrayXPI zeros = ArrayXPI::Zero(padding);
      if (myid == 0)
      {
        rR1 = 1; sR1 = 0; rR2 = 0; sR2 = 1;
        // Upward send
        MPI_Isend(zeros.data(), padding,
                    MPI_PETSCINT, myid+1, 0, comm, &sendReq1);
        // Receive from above
        MPI_Irecv(newElemNumber.data(), padding, MPI_PETSCINT,
                    myid+1, 1, comm, &recReq2);
      }
      else if (myid == nprocs-1)
      {
        rR2 = 1; sR2 = 0; rR1 = 0; sR1 = 1;
        // Downward send
        MPI_Isend(zeros.data(), padding, MPI_PETSCINT,
                    myid-1, 1, comm, &sendReq2);
        // Receive from below
        MPI_Irecv(newElemNumber.data(), padding, MPI_PETSCINT,
                myid-1, 0, comm, &recReq1);
      }
      else
      {
        rR2 = 0; sR2 = 0; rR1 = 0; sR1 = 0;
        // Receive from above
        MPI_Irecv(newElemNumber.data()+padding, padding, MPI_PETSCINT,
                    myid+1, 1, comm, &recReq2);
        // Receive from below
        MPI_Irecv(newElemNumber.data(), padding, MPI_PETSCINT,
            myid-1, 0, comm, &recReq1);
      }
      do {
        if (rR1 == 0)
        {
          MPI_Test(&recReq1, &rR1, MPI_STATUS_IGNORE);
          if (rR1 == 1) // Just got the message from below
          {
            // Upward send
            MPI_Isend(newElemNumber.data(), padding,
                        MPI_PETSCINT, myid+1, 0, comm, &sendReq1);
            sR1 = 0;
          }
        }
        else if (sR1 == 0)
        {
          MPI_Test(&sendReq1, &sR1, MPI_STATUS_IGNORE);
        }
        if (rR2 == 0)
        {
          MPI_Test(&recReq2, &rR2, MPI_STATUS_IGNORE);
          if (rR2 == 1) // Just got the message from above
          {
            // Downward send
            MPI_Isend(newElemNumber.data()+padding, padding,
                      MPI_PETSCINT, myid-1, 1, comm, &sendReq2);
            sR2 = 0;
          }
        }
        else if (sR2 == 0)
        {
          MPI_Test(&sendReq2, &sR2, MPI_STATUS_IGNORE);
        }
      } while (!(sR1 && sR2 && rR1 && rR2));
    }

    /// Edit the edge information
    start = elmdist(myid) - start; // start is now the first element in elemValidity
    Eigen::Array<bool, -1, 1> edgeValidity(edgeElem.rows());
    for (int edge = 0; edge < edgeElem.rows(); edge++)
    {
      edgeValidity(edge) = false;
      if (edgeElem(edge,0) == nElem)
        edgeElem(edge,0) = newnElem;
      else
      {
        edgeElem(edge,0) = newElemNumber(edgeElem(edge,0)-start)-1;
        if (edgeElem(edge,0) == -1)
          edgeElem(edge,0) = newnElem;
        else
          edgeValidity(edge) = true;
      }

      if (edgeElem(edge,1) == nElem)
        edgeElem(edge,1) = newnElem;
      else
      {
        edgeElem(edge,1) = newElemNumber(edgeElem(edge,1)-start)-1;
        if (edgeElem(edge,1) == -1)
          edgeElem(edge,1) = newnElem;
        else
          edgeValidity(edge) = true;
      }
    }
    EigLab::RemoveSlices(edgeElem, edgeValidity, 1);
    EigLab::RemoveSlices(edgeSize, edgeValidity, 1);

    /// Prepare to check validity of nodes
    // first and last+1 node needed by elements on this process
    // start is now the first node used by this process, finish is the last
    if (element.size() > 0)
    {
      start = element.minCoeff();
      finish = element.maxCoeff()+1;
    }
    else
    {
      start = nddist(myid); finish = nddist(myid);
    }
    // Array to first check if node is needed and then track renumbering
    ArrayXPI newNodeNumber;
    if (myid > 0 && myid < nprocs-1)
      newNodeNumber.setZero(nLocNode+2*nInterfaceNodes);
    else
      newNodeNumber.setZero(nLocNode+nInterfaceNodes);

    /// Loop over elements to see which nodes are needed by this process' elements
    for (int el = 0; el < nLocElem; el++)
    {
      if (newElemNumber(el + (myid>0)*padding) > 0)
      {
        for (int nd = 0; nd < element.cols(); nd++)
          newNodeNumber(element(el,nd)-start) = 1;
      }
    }

    // Container to use for communications with adjacent processes
    ArrayXPI Receptacle = ArrayXPI::Zero(2*nInterfaceNodes);
    // Share validity of edge nodes if this process has any, pass through otherwise
    MPI_Status sendStat1, sendStat2, recStat1, recStat2;
    if (nLocNode > 0)
    {
      if (myid > 0 && myid != nprocs-1)
      {
        // Upward send
        MPI_Isend(newNodeNumber.data()+nLocNode, nInterfaceNodes,
                    MPI_PETSCINT, myid+1, 0, comm, &sendReq1);
        sR1 = 0;
        // Downward send
        MPI_Isend(newNodeNumber.data(), nInterfaceNodes, MPI_PETSCINT,
                    myid-1, 1, comm, &sendReq2);
        sR2 = 0;
        // Receive from below
        MPI_Irecv(Receptacle.data(), nInterfaceNodes, MPI_PETSCINT,
                    myid-1, 0, comm, &recReq1);
        rR1 = 0;
        // Receive from above
        MPI_Irecv(Receptacle.data()+nInterfaceNodes, nInterfaceNodes, MPI_PETSCINT,
                    myid+1, 1, comm, &recReq2);
        rR2 = 0;
      }
      else if (myid == 0)
      {
        // Upward send
        MPI_Isend(newNodeNumber.data()+nLocNode, nInterfaceNodes,
                    MPI_PETSCINT, myid+1, 0, comm, &sendReq1);
        sR1 = 0;
        // Receive from above
        MPI_Irecv(Receptacle.data()+nInterfaceNodes, nInterfaceNodes, MPI_PETSCINT,
                    myid+1, 1, comm, &recReq2);
        rR2 = 0;

      }
      else
      {
        // Downward send
        MPI_Isend(newNodeNumber.data(), nInterfaceNodes, MPI_PETSCINT,
                    myid-1, 1, comm, &sendReq2);
        sR2 = 0;
        // Receive from below
        MPI_Irecv(Receptacle.data(), nInterfaceNodes, MPI_PETSCINT,
                    myid-1, 0, comm, &recReq1);
        rR1 = 0;
      }

      do {
        if (sR1 == 0)
        {
          MPI_Test(&sendReq1, &sR1, &sendStat1);
        }
        if (sR2 == 0)
        {
          MPI_Test(&sendReq2, &sR2, &sendStat2);
        }
        if (rR1 == 0)
        {
          MPI_Test(&recReq1, &rR1, &recStat1);
          if (rR1 == 1) // Just got the message from below
          {
            // Combine indicators from both processes
            newNodeNumber.segment(nInterfaceNodes, nInterfaceNodes) =
              newNodeNumber.segment(nInterfaceNodes, nInterfaceNodes).max(
              Receptacle.segment(0,nInterfaceNodes) );
          }
        }
        if (rR2 == 0)
        {
          MPI_Test(&recReq2, &rR2, &recStat2);
          if (rR2 == 1) // Just got the message from above
          {
            // Combine indicators from both processes
            newNodeNumber.segment(newNodeNumber.size()-2*nInterfaceNodes, nInterfaceNodes) =
              newNodeNumber.segment(newNodeNumber.size()-2*nInterfaceNodes, nInterfaceNodes).max(
              Receptacle.segment(nInterfaceNodes,nInterfaceNodes) );
          }
        }
      } while (!(sR1 && sR2 && rR1 && rR2));
    }
    else
    {
      // this process owns no nodes currently - receive from adjacent
      // processes and pass through
      ArrayXPI zeros = ArrayXPI::Zero(nInterfaceNodes);
      if (myid == 0)
      {
        rR1 = 1; sR1 = 0; rR2 = 0; sR2 = 1;
        // Upward send
        MPI_Isend(zeros.data(), nInterfaceNodes,
                    MPI_PETSCINT, myid+1, 0, comm, &sendReq1);
        // Receive from above
        MPI_Irecv(Receptacle.data(), nInterfaceNodes, MPI_PETSCINT,
                    myid+1, 1, comm, &recReq2);
      }
      else if (myid == nprocs-1)
      {
        rR2 = 1; sR2 = 0; rR1 = 0; sR1 = 1;
        // Downward send
        MPI_Isend(zeros.data(), nInterfaceNodes, MPI_PETSCINT,
                    myid-1, 1, comm, &sendReq2);
        // Receive from below
        MPI_Irecv(Receptacle.data(), nInterfaceNodes, MPI_PETSCINT,
                myid-1, 0, comm, &recReq1);
      }
      else
      {
        rR2 = 0; sR2 = 0; rR1 = 0; sR1 = 0;
        // Receive from above
        MPI_Irecv(Receptacle.data()+nInterfaceNodes, nInterfaceNodes, MPI_PETSCINT,
                    myid+1, 1, comm, &recReq2);
        // Receive from below
        MPI_Irecv(Receptacle.data(), nInterfaceNodes, MPI_PETSCINT,
            myid-1, 0, comm, &recReq1);
      }

      do {
        if (rR1 == 0)
        {
          MPI_Test(&recReq1, &rR1, &recStat1);
          if (rR1 == 1) // Just got the message from below
          {
            // Upward send
            MPI_Isend(Receptacle.data(), nInterfaceNodes,
                        MPI_PETSCINT, myid+1, 0, comm, &sendReq1);
            sR1 = 0;
          }
        }
        else if (sR1 == 0)
        {
          MPI_Test(&sendReq1, &sR1, &sendStat1);
        }
        if (rR2 == 0)
        {
          MPI_Test(&recReq2, &rR2, &recStat2);
          if (rR2 == 1) // Just got the message from above
          {
            // Downward send
            MPI_Isend(Receptacle.data()+nInterfaceNodes, nInterfaceNodes,
                      MPI_PETSCINT, myid-1, 1, comm, &sendReq2);
            sR2 = 0;
          }
        }
        else if (sR2 == 0)
        {
          MPI_Test(&sendReq2, &sR2, &sendStat2);
        }
      } while (!(sR1 && sR2 && rR1 && rR2));
    }

    /// Validity of all nodes and elements has been determined
    /// Renumber local nodes
    number(myid) = 0;
    for (PetscInt nd = nddist(myid)-start; nd < nddist(myid+1)-start; nd++)
    {
      if (newNodeNumber(nd) > 0)
      {
        newNodeNumber(nd) = ++number(myid);
      }
    }
    // Share how many nodes are stored locally on this process
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_PETSCINT, number.data(), 1, MPI_PETSCINT, comm);

    /// Renumber nonlocal nodes
    // Nodes on higher-numbered processes
    for (PetscInt nd = nddist(myid+1)-start; nd < finish-start; nd++)
    {
      if (newNodeNumber(nd) > 0)
      {
        newNodeNumber(nd) = ++number(myid);
      }
    }
    number(myid) = 0;
    // Nodes on lower-numbered processes
    for (PetscInt nd = nddist(myid)-start-1; nd >= 0; nd--)
    {
      if (newNodeNumber(nd) > 0)
      {
        newNodeNumber(nd) = --number(myid);
      }
    }

    // Calculate first number on this process (first process starts at 1)
    number(myid) = number.segment(0, myid).sum();
    // Apply this offset to already calculated node numberings
    newNodeNumber += number(myid)*(newNodeNumber!=0).cast<PetscInt>();

    // Remove unwanted elements
    if (myid > 0)
      newElemNumber = newElemNumber.segment(padding, nLocElem).eval();
    else
      newElemNumber = newElemNumber.segment(0, nLocElem).eval();
    EigLab::RemoveSlices(element, newElemNumber, 1);

    // Reassign node numbers to remaining Elements
    for (int el = 0; el < element.rows(); el++)
    {
      for (int nd = 0; nd < element.cols(); nd++)
      {
        PetscInt newNum = newNodeNumber(element(el,nd)-start);
        if (newNum >= number(myid))
          element(el,nd) = newNum-1;
        else
          element(el,nd) = newNum;
      }
    }

    /// Reassign node numbers in projection matrices
    // Start by getting a global list of the new node numbers
    ArrayXPI allNodeNumber = ArrayXPI::Zero(nNode);
    ArrayXPI displs = ArrayXPI::Zero(nprocs);
    partial_sum(nddist.data(), nddist.data()+nprocs-1, displs.data()+1);
    ArrayXPI sizes = nddist.segment(1,nprocs)-nddist.segment(0,nprocs);
    allNodeNumber(nddist(myid),nddist(myid+1)-nddist(myid)) = 
        newNodeNumber(nddist(myid)-start,nddist(myid+1)-nddist(myid));
    MPI_Allgatherv(newNodeNumber.data()+nddist(myid)-start, nddist(myid+1)-nddist(myid), 
         MPI_PETSCINT, allNodeNumber.data(), sizes.data(), displs.data(), MPI_PETSCINT, comm);
    MPI_Allreduce(MPI_IN_PLACE, allNodeNumber.data(), nNode, MPI_PETSCINT, MPI_SUM, comm);
    // Now update numbers in the lists
    for (int level = 0; level < mg_levels-1; level++)
    { 
      int Iind = 0, Jind = 0, cind = 0;
      for (int j = 0; j < I[level].size(); j++)
      {
        if (allNodeNumber(I[level](j)) != nNode)
          I[level](Iind++) = allNodeNumber(I[level](j))-1;
        if (allNodeNumber(J[level](j)) != nNode)
          J[level](Jind++) = allNodeNumber(J[level](j))-1;
      }
      for (int j = 0; j < cList[level].size(); j++)
        if (allNodeNumber(cList[level](j)) != nNode)
         cList[level](cind++) = allNodeNumber(cList[level](j))-1;
      I[level].conservativeResize(Iind); J[level].conservativeResize(Jind);
      cList[level].conservativeResize(cind);
    }

    /// Remove elements from the filter
    // By row first
    int ind = 0;
    for (PetscInt el = 0; el < filterArrays.nElem; el++)
    {
      PetscInt elem = filterArrays.elements(el,0)-elmdist(myid);
      // Overwrite rows that have been removed
      if ( elemValidity(elem) )
      {
        filterArrays.elements.row(ind) << newElemNumber(elem)-1, filterArrays.elements(el,1);
        filterArrays.distances(ind) = filterArrays.distances(el);
        ind++;
      }
    }
    filterArrays.Truncate(ind);

    // Now renumber and remove invalid column elements
    // Loop through all arrays owned by other processes
    for (int proc = 0; proc < nprocs; proc++)
    {
      int activeproc = (proc+myid) % nprocs;
      int ind = 0;
      // Loop through each row of the filter
      for (PetscInt row = 0; row < filterArrays.nElem; row++)
      {

        PetscInt element = filterArrays.elements(row, 1);
        // If this column's element hasn't been updated and
        // was originally created on activeproc
        if (element >= elmdist(activeproc) &&
            element < elmdist(activeproc+1) &&
            !filterArrays.modified(row))
        {
          // If that element is invalid, remove that column, otherwise
          // update the column number
          if ( newElemNumber(element-elmdist(activeproc)) > 0 )
          {
            filterArrays.modified(ind) = true;
            filterArrays.elements.row(ind) << filterArrays.elements(row,0),
                            newElemNumber(element-elmdist(activeproc))-1;
            filterArrays.distances(ind++) = filterArrays.distances(row);
          }
        }
        else
        {
          filterArrays.modified(ind) = filterArrays.modified(row);
          filterArrays.elements.row(ind) = filterArrays.elements.row(row);
          filterArrays.distances(ind++) = filterArrays.distances(row);
        }

      }
      filterArrays.Truncate(ind);

      // Share the validity information with the next process
      MPI_Request request = MPI_REQUEST_NULL;
      MPI_Status status;
      if (myid > 0)
      {
        MPI_Issend(newElemNumber.data(), newElemNumber.size(), MPI_PETSCINT,
                   myid-1, 0, comm, &request);
      }
      else
      {
        MPI_Issend(newElemNumber.data(), newElemNumber.size(), MPI_PETSCINT,
                   nprocs-1, 0, comm, &request);
      }

      // Check the incoming message and recieve the new information
      if (myid < nprocs-1)
      {
        MPI_Probe(myid+1, 0, comm, &status);
        int count;
        MPI_Get_count(&status, MPI_PETSCINT, &count);
        ArrayXPI newNumber(count);
        MPI_Recv(newNumber.data(), count, MPI_PETSCINT, myid+1, 0, comm, &status);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        newElemNumber = newNumber;
      }
      else
      {
        MPI_Probe(0, 0, comm, &status);
        int count;
        MPI_Get_count(&status, MPI_PETSCINT, &count);
        ArrayXPI newNumber(count);
        MPI_Recv(newNumber.data(), count, MPI_PETSCINT, 0, 0, comm, &status);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        newElemNumber = newNumber;
      }
    }

    /// Reset element distribution array
    number = elmdist; // Need this to fix filter matrix later
    elmdist.setZero(nprocs+1);
    elmdist(myid+1) = element.rows();
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_PETSCINT, elmdist.data()+1, 1, MPI_PETSCINT, comm);
    for (int id = 1; id <= nprocs; id++)
      elmdist(id) += elmdist(id-1);
    nLocElem = element.rows();
    nElem = elmdist(nprocs);

    /// Remove unwanted Nodes
    newNodeNumber = newNodeNumber.segment(nddist(myid)-start,nLocNode).eval();
    EigLab::RemoveSlices(node, newNodeNumber, 1);

    /// Reset the node distribution array
    nddist.setZero(nprocs+1);
    nddist(myid+1) = node.rows();
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_PETSCINT, nddist.data()+1, 1, MPI_PETSCINT, comm);
    for (int id = 1; id <= nprocs; id++)
      nddist(id) += nddist(id-1);
    nLocNode = node.rows();
    nNode = nddist(nprocs);

    return ierr;
}

/*****************************************************************/
/**               Get partitioning with ParMetis                **/
/*****************************************************************/
idx_t TopOpt::ReorderParMetis(FilterArrays &filterArrays, bool Reorder_Mesh,
            idx_t nparts, idx_t ncommonnodes, double *tpwgts, double *ubvec,
            idx_t *opts, idx_t ncon, idx_t *elmwgt, idx_t wgtflag, idx_t numflag )
{
  /// ParMetis won't work if some processors have zero elements, so perform
  /// an initial redistribution
  Eigen::Array<idx_t, -1, 1> partition =
    myid*Eigen::Array<idx_t, -1, 1>::Ones(nLocElem);
  ArrayXPI checkpoints = ArrayXPI::LinSpaced(nprocs+1, 0, nElem);

  for (int i = 0; i < nprocs; i++)
  {
    if (checkpoints(i+1) < elmdist(myid))
      continue;
    else if (checkpoints(i) >= elmdist(myid+1))
      break;
    else
      partition.segment(max((PetscInt)0, checkpoints(i)-elmdist(myid)),
                    min(min(min(checkpoints(i+1) - checkpoints(i),
                                elmdist(myid+1) - checkpoints(i)),
                                checkpoints(i+1) - elmdist(myid)),
                                elmdist(myid+1) - elmdist(myid)) ).setConstant(i);
  }
  ElemDist(filterArrays, partition);

  /// Verify Inputs
  if (nparts <= 0)
    nparts = nprocs;
  if (ncommonnodes <= 0)
    ncommonnodes = pow(2, numDims-1);

  // Local Element Descriptions - element contains the nodes,
  // eptr specifies where each element starts
  short elementSize = pow(2,numDims);
  Eigen::Array<idx_t, -1, -1> eptr =
    Eigen::Array<idx_t, -1, 1>::LinSpaced(nLocElem+1,0,nLocElem*elementSize);
  partition = myid*Eigen::Array<idx_t, -1, 1>::Ones(nLocElem);

  /// Initialize ParMETIS Variables
  if (ubvec == NULL)                            //Imbalance tolerance
  {
    ubvec = new real_t[ncon];
    for (int i = 0; i < ncon; i++)
      ubvec[i] = 1.05+(double)nparts/nElem;
  }

  if (opts == NULL)                         //0 for default options
  { opts = new idx_t; opts[0] = 0;}

  if (tpwgts == NULL)                //Vertex weight in each subdomain
  {
    tpwgts = new double[ncon*nparts];
    for (int i = 0; i < nparts; i++)
    {
      for (int j = 0; j < ncon; j++)
        tpwgts[i*ncon+j] = 1.0/nparts;
    }
  }
  idx_t edgecut;

  // Call ParMETIS
  idx_t METIS;
  if (sizeof(idx_t) != sizeof(PetscInt))
    cout << "WARNING, PetscInt and Parmetis int (idx_t) are of different sizes, " <<
            "skipping reordering with parmetis.\n";
  if (Reorder_Mesh && (sizeof(idx_t) == sizeof(PetscInt)) )
    METIS = ParMETIS_V3_PartMeshKway((idx_t*)elmdist.data(), (idx_t*)eptr.data(),
            (idx_t*)element.data(), elmwgt, &wgtflag, &numflag, &ncon,
            &ncommonnodes, &nparts, tpwgts, ubvec, opts,
            &edgecut, partition.data(), &comm);
  else
    partition.setConstant(myid); METIS = METIS_OK;
  delete[] ubvec;
  delete[] tpwgts;
  delete opts;

  if (METIS != METIS_OK)
  {
    std::cout << "Error partitioning matrix! Error code: " << METIS << "\n";
    return METIS;
  }
    
  ArrayXPI permute;
  ElemDist(filterArrays, partition);

  // TODO: Move filter matrix assembly to its own function
  PetscErrorCode ierr = 0;
  /// Assemble the filter matrix
  ierr = MatCreate(comm, &P); CHKERRQ(ierr);
  ierr = MatSetSizes(P, nLocElem, nLocElem, nElem, nElem); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(P,"P_"); CHKERRQ(ierr);
  ierr = MatSetFromOptions(P); CHKERRQ(ierr);

  // Set preallocation
  ArrayXPI onDiag = ArrayXPI::Zero(nLocElem);
  ArrayXPI offDiag = ArrayXPI::Zero(nLocElem);
  for (int el = 0; el < filterArrays.nElem; el++)
  {
    if (filterArrays.elements(el,1) >= elmdist(myid) &&
        filterArrays.elements(el,1) < elmdist(myid+1) )
    {
      onDiag(filterArrays.elements(el,0)-elmdist(myid))++;
    }
    else
    {
      offDiag(filterArrays.elements(el,0)-elmdist(myid))++;
    }
  }

  // Set the preallocation
  ierr = MatXAIJSetPreallocation(P, 1, onDiag.data(), offDiag.data(), 0, 0); CHKERRQ(ierr);

  // Insert values into matrix
  for (int el = 0; el < filterArrays.nElem; el++)
  {
    ierr = MatSetValue(P, filterArrays.elements(el,0), filterArrays.elements(el,1),
                filterArrays.distances(el), ADD_VALUES); CHKERRQ(ierr);
  }

  // Begin assembly (finish just before returning from function)
  MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  // Finish assembly of the matrix before continuing
  MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  // Scale Rows
  Vec rowSum, Ones;
  ierr = VecCreateMPI(comm, nLocElem, nElem, &rowSum); CHKERRQ(ierr);
  ierr = VecDuplicate(rowSum, &Ones); CHKERRQ(ierr);
  ierr = VecSet(Ones, 1.0); CHKERRQ(ierr);
  ierr = MatGetRowSum(P, rowSum); CHKERRQ(ierr);
  ierr = VecPointwiseDivide(rowSum, Ones, rowSum); CHKERRQ(ierr);
  ierr = MatDiagonalScale(P, rowSum, NULL); CHKERRQ(ierr);
  ierr = VecDestroy(&rowSum); CHKERRQ(ierr);
  ierr = VecDestroy(&Ones); CHKERRQ(ierr);

  return 0;
}

/*****************************************************************/
/**                    Redistribute elements                    **/
/*****************************************************************/
PetscErrorCode TopOpt::ElemDist(FilterArrays &filterArrays,
                      Eigen::Array<idx_t, -1, 1> &partition)
{
    PetscErrorCode ierr = 0;
    /// Reallocate elements
    /// Note abbreviations: senddisp = first element in array sent to each process
    /// sendcnt = how many elements sent to each process - TO BE REMOVED
    /// transferSize = how many elements each process is sending to the other processes
    /// recvcnt = how many elements received from each process
    /// recvdsp = beginning location of buffer to receive elements from each process
    /// elmcpy = a copy of element reordered for continguous send buffers
    /// where = after initial sorting, the local number of each element
    /// permute = permutation vector for filter matrix (global)
    // Initialize transfer Variables
    short elementSize = pow(2,numDims);
    ArrayXPI where = EigLab::gensort(partition).cast<PetscInt>();
    ArrayXXPI transferSize = ArrayXXPI::Zero(nprocs,nprocs);
    ArrayXXPIRM elmcpy(element.rows(),element.cols());
    for (PetscInt i = 0; i < partition.rows(); i++)
    {
      elmcpy.row(i) = element.row(where(i));
      transferSize(partition(i),myid)++;
    }

    // How many elements are transferred between each pair of processes
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_PETSCINT, transferSize.data(),
                  nprocs, MPI_PETSCINT, comm);
    Eigen::ArrayXi sendcnt = elementSize*transferSize.col(myid).cast<int>();
    Eigen::ArrayXi recvcnt = elementSize*transferSize.row(myid).cast<int>();

    // Offsets in sent messages
    Eigen::ArrayXi senddsp = Eigen::ArrayXi::Zero(nprocs);
    for (short i = 1; i < nprocs; i++)
        senddsp(i) = sendcnt(i-1) + senddsp(i-1);

    // Offsets in received messages
    Eigen::ArrayXi recvdsp = Eigen::ArrayXi::Zero(nprocs);
    for (short i = 1; i < nprocs; i++)
        recvdsp(i) = recvcnt(i-1) + recvdsp(i-1);

    // The element transfer
    element.resize(recvcnt.sum()/elementSize, elementSize);
    MPI_Alltoallv(elmcpy.data(), sendcnt.data(), senddsp.data(),
                  MPI_PETSCINT, element.data(), recvcnt.data(),
                  recvdsp.data(), MPI_PETSCINT, comm);

    // Update distribution across processes
    elmdist(myid+1) = element.rows();
    nLocElem = element.rows();
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, elmdist.data()+1,
                  1, MPI_PETSCINT, comm);
    for (short i = 1; i <= nprocs; i++)
        elmdist(i) += elmdist(i-1);

    // Create global permutation array after sharing Elements
    // This is currently assembling a global vector on all processes and reducing
    // it.  The performance could possibly be improved by sharing the local parts
    // and then assembling after transfer, thereby reducing communications.
    // Permute = permutation vector, permute(i) = newi
    // Indices = vector indicating where this process can start assigning Elements
    //            on each process (i.e. global locations in the permute vector)
    ArrayXPI permute = ArrayXPI::Zero(nElem);
    ArrayXPI indices = ArrayXPI::Zero(nprocs);
    indices.segment(1,nprocs-1) = transferSize.block(0, 0, nprocs-1, nprocs)
                                  .rowwise().sum();
    partial_sum(indices.data(), indices.data()+nprocs, indices.data());
    indices += transferSize.block(0, 0, nprocs, myid).rowwise().sum();
    int permuteStart = transferSize.block(0, 0, nprocs, myid).sum();
    for (PetscInt i = 0; i < partition.rows(); i++)
    {
      permute(where(i)+permuteStart) = indices(partition(i))++;
    }
    MPI_Allreduce(MPI_IN_PLACE, permute.data(), nElem, MPI_PETSCINT, MPI_SUM, comm);

    /// Apply permutations to edge information
    for (int edge = 0; edge < edgeElem.rows(); edge++)
    {
      if (edgeElem(edge,0) < nElem)
        edgeElem(edge,0) = permute(edgeElem(edge,0));
      if (edgeElem(edge,1) < nElem)
        edgeElem(edge,1) = permute(edgeElem(edge,1));
      // Arrange so largest actual element is first
      if ( (edgeElem(edge,0) < edgeElem(edge,1) && edgeElem(edge,1) < nElem) ||
            edgeElem(edge,0) == nElem )
      {
        PetscInt temp = edgeElem(edge,0);
        edgeElem(edge,0) = edgeElem(edge,1);
        edgeElem(edge,1) = temp;
      }
    }
    // sort edges by element numbers for redistribution
    where = EigLab::sort(edgeElem, 1).cast<PetscInt>();
    // reorder sizes as well
    Eigen::ArrayXd copyDouble = edgeSize;
    for (int i = 0; i < edgeSize.rows(); i++)
    {
      edgeSize(i) = copyDouble(where(i));
    }

    /// Apply permutations to filter information
    for (int el = 0; el < filterArrays.nElem; el++)
    {
      if (filterArrays.elements(el,0) < nElem)
        filterArrays.elements(el,0) = permute(filterArrays.elements(el,0));
      if (filterArrays.elements(el,1) < nElem)
        filterArrays.elements(el,1) = permute(filterArrays.elements(el,1));
    }
    // sort filter elements for redistribution
    where = EigLab::sort(filterArrays.elements, 1).cast<PetscInt>();
    // reorder distances as well
    copyDouble = filterArrays.distances;
    for (int i = 0; i < filterArrays.nElem; i++)
    {
      filterArrays.distances(i) = copyDouble(where(i));
    }

    /// Redistribute edges
    transferSize.setZero();
    for (int edge = 0; edge < edgeElem.rows(); edge++)
    {
      int location = (elmdist <= edgeElem(edge,0)).cast<int>().sum() - 1;
      transferSize(location, myid)++;
    }

    // How many elements are transferred between each pair of processes
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_PETSCINT, transferSize.data(),
                  nprocs, MPI_PETSCINT, comm);
    sendcnt = 2*transferSize.col(myid).cast<int>();
    recvcnt = 2*transferSize.row(myid).cast<int>();

    // Offsets in sent messages
    senddsp.setZero(nprocs);
    for (short i = 1; i < nprocs; i++)
        senddsp(i) = sendcnt(i-1) + senddsp(i-1);

    // Offsets in received messages
    recvdsp.setZero(nprocs);
    for (short i = 1; i < nprocs; i++)
        recvdsp(i) = recvcnt(i-1) + recvdsp(i-1);

    // The edge element transfer
    elmcpy = edgeElem;
    edgeElem.resize(recvcnt.sum()/2, 2);
    MPI_Alltoallv(elmcpy.data(), sendcnt.data(), senddsp.data(),
                  MPI_PETSCINT, edgeElem.data(), recvcnt.data(),
                  recvdsp.data(), MPI_PETSCINT, comm);

    // The size transfer
    sendcnt /= 2; senddsp /= 2;
    recvcnt /= 2; recvdsp /= 2;
    copyDouble = edgeSize;
    edgeSize.resize(recvcnt.sum());
    MPI_Alltoallv(copyDouble.data(), sendcnt.data(), senddsp.data(),
                  MPI_DOUBLE, edgeSize.data(), recvcnt.data(),
                  recvdsp.data(), MPI_DOUBLE, comm);

    /// Reallocate filter information
    transferSize.setZero();
    for (int filter = 0; filter < filterArrays.nElem; filter++)
    {
      int location = (elmdist <= filterArrays.elements(filter,0)).cast<int>().sum() - 1;
      transferSize(location, myid)++;
    }

    // How many elements are transferred between each pair of processes
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_PETSCINT, transferSize.data(),
                  nprocs, MPI_PETSCINT, comm);
    sendcnt = 2*transferSize.col(myid).cast<int>();
    recvcnt = 2*transferSize.row(myid).cast<int>();

    // Offsets in sent messages
    senddsp.setZero(nprocs);
    for (short i = 1; i < nprocs; i++)
        senddsp(i) = sendcnt(i-1) + senddsp(i-1);

    // Offsets in received messages
    recvdsp.setZero(nprocs);
    for (short i = 1; i < nprocs; i++)
        recvdsp(i) = recvcnt(i-1) + recvdsp(i-1);

    // The actual transfer
    elmcpy = filterArrays.elements;
    copyDouble = filterArrays.distances;
    filterArrays.Reset(recvcnt.sum()/2);
    MPI_Alltoallv(elmcpy.data(), sendcnt.data(), senddsp.data(),
                  MPI_PETSCINT, filterArrays.elements.data(), recvcnt.data(),
                  recvdsp.data(), MPI_PETSCINT, comm);

    sendcnt /= 2; senddsp /= 2;
    recvcnt /= 2; recvdsp /= 2;
    MPI_Alltoallv(copyDouble.data(), sendcnt.data(), senddsp.data(),
                  MPI_DOUBLE, filterArrays.distances.data(), recvcnt.data(),
                  recvdsp.data(), MPI_DOUBLE, comm);
    return ierr;
}

/*****************************************************************/
/**                      Node Distribution                      **/
/*****************************************************************/
PetscErrorCode TopOpt::NodeDist(ArrayXPI *I, ArrayXPI *J, ArrayXPS *K, ArrayXPI *cList, int mg_levels)
{
    PetscErrorCode ierr = 0;

    /// Find which nodes each process interacts with
    Eigen::Array<short,-1,1> pckproc = Eigen::Array<short,-1,1>::Zero(nNode);
    for (PetscInt el = 0; el < element.rows(); el++)
    {
        for (short nd = 0; nd < pow(2, numDims); nd++)
            pckproc(element(el,nd)) = myid;
    }

    // Assign nodes to the highest numbered processor that uses them
    MPI_Allreduce(MPI_IN_PLACE, pckproc.data(), nNode, MPI_SHORT, MPI_MAX, comm);

    /// Sort nodes into chunks to go to each process
    Eigen::Array<short,-1,1> locpart = pckproc.segment(nddist(myid),nLocNode);
    ArrayXPI reorder = EigLab::gensort(locpart).cast<PetscInt>();
    /// Package the nodes into a new array for sending to each process
    /// And track how many are being sent to each process
    MatrixXdRM ndcpy(node.rows(),node.cols());
    Eigen::ArrayXi sendcnt = Eigen::ArrayXi::Zero(nprocs);
    for (PetscInt i = 0; i < locpart.rows(); i++)
    {
        ndcpy.row(i) = node.row(reorder(i)).transpose();
        sendcnt(locpart(i))++;
    }

    // How much to receive from every process
    Eigen::ArrayXi recvcnt(nprocs);
    MPI_Alltoall(sendcnt.data(), 1, MPI_INT, recvcnt.data(), 1, MPI_INT, comm);

    /// Offsets in sent messages
    Eigen::ArrayXi senddsp = Eigen::ArrayXi::Zero(nprocs);
    for (short i = 1; i < nprocs; i++)
        senddsp(i) = numDims*sendcnt(i-1) + senddsp(i-1);

    /// Offsets in recieved messages
    Eigen::ArrayXi recvdsp = Eigen::ArrayXi::Zero(nprocs);
    for (short i = 1; i < nprocs; i++)
        recvdsp(i) = numDims*recvcnt(i-1) + recvdsp(i-1);

    /// The node transfer
    node.resize(recvcnt.sum(),numDims);
    recvcnt *= numDims; sendcnt *= numDims;
    MPI_Alltoallv(ndcpy.data(), sendcnt.data(), senddsp.data(),
                  MPI_DOUBLE, node.data(), recvcnt.data(),
                  recvdsp.data(), MPI_DOUBLE, comm);

    /// Update the distribution of nodes
    nddist.setZero(nprocs+1);
    nLocNode = node.rows();
    nddist(myid+1) = node.rows();
    MPI_Allreduce(MPI_IN_PLACE, nddist.data()+1, nprocs, MPI_PETSCINT, MPI_MAX, comm);
    for (short i = 1; i <= nprocs; i++)
        nddist(i) += nddist(i-1);

    /// Renumber nodes in element array
    reorder = EigLab::gensort(pckproc).cast<PetscInt>();
    ArrayXPI invreorder(reorder.rows());
    for (PetscInt i = 0; i < reorder.rows(); i++)
      invreorder(reorder(i)) = i;
    for (PetscInt el = 0; el < element.rows(); el++)
    {
        for (short nd = 0; nd < pow(2,numDims); nd++)
        {
            element(el,nd) = invreorder(element(el,nd));
        }
    }

    // Give each node in the interpolation it's new number after redistribution
    for (int i = mg_levels-2; i >= 0; i--)
    {
      for (int j = 0; j < I[i].size(); j++)
      {
        I[i](j) = invreorder(I[i](j));
        J[i](j) = invreorder(J[i](j));
      }
      for (int j = 0; j < cList[i].size(); j++)
        cList[i](j) = invreorder(cList[i](j));
    }

    return ierr;
}

/*****************************************************************/
/**      Capture surrounding elements on other processes        **/
/*****************************************************************/
PetscErrorCode TopOpt::Expand_Elem()
{
  PetscErrorCode ierr = 0;

  /// Make a list of all elements with non-local nodes and where to send them
  /// nonLocal contains the element number followed by each of its nodes, listed
  /// consecutively for each process, so that it is ready to send upon completion
  /// of the for loop
  std::vector<PetscInt> *nonLocalElems = new std::vector<PetscInt>[nprocs];
  std::vector<PetscInt> *nonLocalNums = new std::vector<PetscInt>[nprocs];

  /// Make sure each element isn't sent multiple times to each other process
  std::vector<Eigen::Array<bool,-1,1> > elSent(nprocs);
  for (int proc = 0; proc < nprocs; proc++)
    elSent[proc].setZero(nLocElem);
  short elemSize = element.cols();
  for (int el = 0; el < nLocElem; el++)
  {
    for (int nd = 0; nd < element.cols(); nd++)
    {
      PetscInt node = element(el,nd);
      if ( node < nddist(myid) || node >= nddist(myid+1) )
      {
        // Find the destination process
        short proc = 0;
        while (node >= nddist(proc+1))
          proc++;
        // If it hasn't been shared y-direction
        if (!elSent[proc](el))
        {
          /// Package the element number and nodes separately for sending later
          nonLocalNums[proc].push_back(gElem(el));
          nonLocalElems[proc].insert(nonLocalElems[proc].end(), element.data() +
                      elemSize*el, element.data() + elemSize*(el+1));
          elSent[proc](el) = true;
        }
      }
    }
  }

  /// Perform all the sends
  MPI_Request *requests = new MPI_Request[nprocs];
  requests[myid] = MPI_REQUEST_NULL;
  for (short proc = 0; proc < nprocs; proc++)
  {
    if (proc != myid)
    {
      MPI_Issend(nonLocalNums[proc].data(), nonLocalNums[proc].size(),
                 MPI_PETSCINT, proc, 0, comm, requests + proc);
      MPI_Issend(nonLocalElems[proc].data(), nonLocalElems[proc].size(),
                 MPI_PETSCINT, proc, 1, comm, requests + proc);
     }
  }

  /// Receive all the Information
  // First probe all messsages to see how much the elements array needs to be
  // expanded by, then receive all messages in the new buffer
  Eigen::ArrayXi flags = Eigen::ArrayXi::Zero(nprocs);
  MPI_Status *statuses = new MPI_Status[nprocs];
  Eigen::ArrayXi recvCount(nprocs);
  flags(myid) = 1; recvCount(myid) = 0;
  while (flags.sum() < nprocs)
  {
    for (short proc = 0; proc < nprocs; proc++)
    {
      if (proc != myid && flags(proc) == 0)
      {
        // Receive either the number or node message
        MPI_Iprobe(proc, MPI_ANY_TAG, comm, flags.data()+proc, statuses+proc);
        // Determine number of elements contained in this message
        MPI_Get_count(statuses+proc, MPI_PETSCINT, recvCount.data() + proc);
        recvCount[proc] /= (statuses[proc].MPI_TAG == 0) ? 1 : elemSize;
      }
    }
  }

  // Now do the receives
  gElem.conservativeResize(nLocElem + recvCount.sum());
  element.conservativeResize(nLocElem + recvCount.sum(), elemSize);
  int ind = nLocElem;
  for (short proc = 0; proc < nprocs; proc++)
  {
    if (proc == myid)
      continue;
    MPI_Recv(gElem.data() + ind, recvCount(proc), MPI_PETSCINT,
             proc, 0, comm, MPI_STATUS_IGNORE);
    MPI_Recv(element.data() + ind*elemSize, elemSize*recvCount(proc),
             MPI_PETSCINT, proc, 1, comm, MPI_STATUS_IGNORE);
    ind += recvCount(proc);
  }

  /// All messages should be completed by now, but make sure anyway
  MPI_Waitall(nprocs, requests, MPI_STATUSES_IGNORE);
  delete[] statuses;
  delete[] requests;
  delete[] nonLocalNums;
  delete[] nonLocalElems;

  return ierr;
}

/*****************************************************************/
/**       Capture surrounding nodes on other processes          **/
/*****************************************************************/
PetscErrorCode TopOpt::Expand_Node()
{
  PetscErrorCode ierr = 0;

  /// List of all the nodes the local elements need
  ArrayXPI ndlist = Eigen::Map<ArrayXPI>(element.data(),element.size());
  EigLab::Unique(ndlist, 1);

  /// Pull out already owned nodes
  PetscInt ind = 0, nind = 0;
  for (PetscInt i = 0; i < ndlist.rows(); i++)
  {
      if (ind == gNode.size())
      {
          ndlist.segment(nind, ndlist.rows()-i) = ndlist.segment(i, ndlist.rows()-i);
          nind += ndlist.rows()-i;
          break;
      }
      if (gNode(ind) != ndlist(i))
          ndlist(nind++) = ndlist(i);
      else
          ind++;
  }
  ndlist.conservativeResize(nind);

  /// Find where all those nodes are and how many nodes are neede from each process
  ArrayXPI where( ndlist.rows() );
  Eigen::ArrayXi perproc = Eigen::ArrayXi::Zero( nprocs );
  short proc = 0;
  for (PetscInt i = 0; i < ndlist.rows(); i++)
  {
      while( ndlist(i) >= nddist(proc+1) )
          proc++;
      where(i) = proc;
      perproc(proc)++;
  }

  /// Tell each process how many nodes you need sent over
  Eigen::ArrayXi sendcnt(nprocs);
  MPI_Alltoall(perproc.data(), 1, MPI_INT, sendcnt.data(), 1, MPI_INT, comm);

  /// Offsets in recieved messages regarding which nodes are requested
  Eigen::ArrayXi senddsp = Eigen::ArrayXi::Zero(nprocs);
  for (short i = 1; i < nprocs; i++)
      senddsp(i) = sendcnt(i-1) + senddsp(i-1);

  /// Get offsets in sent messages requesting nodes
  Eigen::ArrayXi perprocdisp = Eigen::ArrayXi::Zero(nprocs);
  for (short i = 1; i < nprocs; i++)
      perprocdisp(i) = perprocdisp(i-1)+perproc(i-1);

  /// Send the nodes you want to each process
  ArrayXPI sendnd(sendcnt.sum());
  MPI_Alltoallv(ndlist.data(), perproc.data(), perprocdisp.data(),
                MPIU_INT, sendnd.data(), sendcnt.data(),
                senddsp.data(), MPIU_INT, comm);

  /// Pack up all the nodes for sending
  MatrixXdRM ndpack(sendcnt.sum(),numDims);
  for (int i = 0; i < sendcnt.sum(); i++)
      ndpack.row(i) = node.row(sendnd(i)-nddist(myid));

  /// Ship the nodes
  node.conservativeResize(nLocNode+perproc.sum(), numDims);
  perprocdisp += nLocNode; perprocdisp *= numDims;
  sendcnt *= numDims; senddsp *= numDims; perproc *= numDims;
  MPI_Alltoallv(ndpack.data(), sendcnt.data(), senddsp.data(),
                MPI_DOUBLE, node.data(), perproc.data(),
                perprocdisp.data(), MPI_DOUBLE, comm);

  /// Update the global node list
  gNode.conservativeResize(node.rows());
  gNode.segment(nLocNode, ndlist.rows()) = ndlist;

  return ierr;
}

/*****************************************************************/
/**       Set up ghost communications for PEtSc vectors         **/
/*****************************************************************/
PetscErrorCode TopOpt::Initialize_Vectors()
{
    PetscErrorCode ierr = 0;
    // Nodal ghost info
    ArrayXXPIRM ghosts( gNode.size()-nLocNode, numDims );
    ghosts.col(0) = numDims*gNode.segment( nLocNode,gNode.size()-nLocNode );
    for (short i = 1; i < numDims; i++)
      ghosts.col(i) = ghosts.col(i-1) + 1;

    ierr = VecCreateGhost(comm, numDims*nLocNode, numDims*nNode,
                          ghosts.size(), ghosts.data(), &U); CHKERRQ(ierr);
    ierr = VecSet(U, 0.0); CHKERRQ(ierr);
    ierr = VecDuplicate(U, &F); CHKERRQ(ierr);
    ierr = VecSet(F, 0.0); CHKERRQ(ierr);

    // Element ghost info
    ierr = VecCreateGhost(comm, nLocElem, nElem, gElem.size()-nLocElem,
                          gElem.data()+nLocElem, &V); CHKERRQ(ierr);
    ierr = VecDuplicate(V, &E); CHKERRQ(ierr);
    ierr = VecDuplicate(V, &Es); CHKERRQ(ierr);
    ierr = VecDuplicate(V, &dVdy); CHKERRQ(ierr);
    ierr = VecDuplicate(V, &dEdy); CHKERRQ(ierr);
    ierr = VecDuplicate(V, &dEsdy); CHKERRQ(ierr);

    /// Create design variable and density vectors to work with the filter
    ierr = VecCreateMPI(comm, nLocElem, nElem, &x); CHKERRQ(ierr);
    ierr = VecDuplicate(x, &rho); CHKERRQ(ierr);

    return ierr;
}

/*****************************************************************/
/**       Convert global numberings to local numberings         **/
/*****************************************************************/
PetscErrorCode TopOpt::Localize()
{
    PetscErrorCode ierr = 0;

    /// Convert elements to local node numbers
    PetscInt *start = gNode.data(), *finish = gNode.data()+gNode.size();
    for (PetscInt i = 0; i < element.rows(); i++)
    {
        for (short j = 0; j < element.cols(); j++)
            element(i,j) = std::find(start, finish, element(i,j)) - start;
    }

    /// Convert elements to local node numbers
    start = gElem.data()+nLocElem; finish = gElem.data()+gElem.size();
    for (PetscInt i = 0; i < edgeElem.rows(); i++)
    {
      // Renumber first element
      edgeElem(i,0) -= elmdist(myid);
      // Check if second element is local or not and act accordingly
      if (edgeElem(i,1) == nElem)
        edgeElem(i,1) = gElem.rows();
      else if (edgeElem(i,1) > elmdist(myid))
        edgeElem(i,1) -= elmdist(myid);
      else
        edgeElem(i,1) = std::find(start, finish, edgeElem(i,1)) - gElem.data();
    }

    return ierr;
}
