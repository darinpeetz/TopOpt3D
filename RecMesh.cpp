#include <iostream>
#include <cmath>
#include <fstream>
#include <Eigen/Eigen>
#include <numeric>
#include <unsupported/Eigen/KroneckerProduct>
#include <mpi.h> // Has to precede Petsc includes to use MPI::BOOL
#include "TopOpt.h"
#include "EigLab.h"

//extern "C"
//{
//  #include <parmetis.h>
//}

using namespace std;

typedef Eigen::Array<PetscInt, -1, -1> ArrayXXPI;
typedef Eigen::Array<PetscInt, -1,  1> ArrayXPI;
typedef Eigen::Array<PetscInt,  1, -1> RowArrayXPI;
typedef Eigen::Array<PetscScalar, -1,  1> ArrayXPS;
typedef Eigen::Matrix<PetscScalar, -1, -1> MatrixXPS;
typedef Eigen::Array<PetscInt, -1, -1, Eigen::RowMajor> ArrayXXPIRM;
typedef Eigen::Matrix<double,-1,-1,Eigen::RowMajor> MatrixXdRM;

/********************************************************************
 * Load Mesh for Restart
 * 
 * @param xIni: The initial design values (output)
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
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
  input.seekg(elmdist(myid)*pow(2, numDims)*sizeof(PetscInt));
  element.resize(elmdist(myid+1)-elmdist(myid), pow(2, numDims));
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

  // Read in BC's
  // Have to read it all in and parse it later
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
  PetscInt begin = 0, finish = temp_node.rows();
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

  // Eigen Analysis Fixed Supports
  filename = folder + "/eigenSupportNodes.bin";
  input.open(filename.c_str(), ios::ate | ios::binary);
  if (input.is_open()) {
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
    eigenSuppNode = temp_node.segment(begin, finish-begin) -= nddist(myid);
    // Load the eigen analysis supports
    filename = folder + "/eigenSupports.bin";
    input.open(filename.c_str(), ios::ate | ios::binary);
    if (!input.is_open())
      SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to open eigen supports file");
    filesize = input.tellg();
    input.seekg(begin*sizeof(bool)*numDims);
    eigenSupports.resize(eigenSuppNode.size(), numDims);
    input.read((char*)eigenSupports.data(), eigenSupports.size()*sizeof(bool));
    input.close();
  }
  else { // assume there are no eigen fixed dofs
    eigenSuppNode.resize(0);
    eigenSupports.resize(0, numDims);
  }

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
  PetscScalar temp = node(element(0,1),0) - node(element(0,0),0);
  if (numDims > 1)
    temp *= node(element(0,3),1) - node(element(0,0),1);
  if (numDims > 2)
    temp *= node(element(0,7),2) - node(element(0,0),2);
  elemSize.setConstant(nLocElem, temp);

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

  filename = folder + "/Max_Filter.bin";
  input.open(filename.c_str(), ios::ate);
  if (!input.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to open max feature filter file");
  input.close();
  ierr = PetscViewerBinaryOpen(comm, filename.c_str(), FILE_MODE_READ, &view);
  ierr = MatCreate(comm, &this->R); CHKERRQ(ierr);
  ierr = MatSetType(this->R, MATAIJ); CHKERRQ(ierr);
  ierr = MatSetSizes(this->R, nLocElem, nLocElem, nElem, nElem); CHKERRQ(ierr);
  ierr = MatLoad(this->R, view); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);

  filename = folder + "/Void_Edge_Volume.bin";
  input.open(filename.c_str(), ios::ate);
  if (!input.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to open max feature filter file");
  input.close();
  ierr = MatCreateVecs(this->R, NULL, &this->REdge); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm, filename.c_str(), FILE_MODE_READ, &view);
  ierr = VecLoad(this->REdge, view); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view); CHKERRQ(ierr);

  // Read in the multigrid hierarchy
  this->PR.resize(0);
  this->MG_comms.resize(0);
  this->MG_comms.push_back(comm);
  PetscInt lrow = this->numDims*nddist(myid+1)-this->numDims*nddist(myid), lcol;
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

  // Read which elements are active
  MPI_File fh;
  this->active.resize(this->nLocElem);
  ierr = MPI_File_open(this->comm, (folder + "/active.bin").c_str(), MPI_MODE_RDONLY,
                       MPI_INFO_NULL, &fh); CHKERRQ(ierr);
  ierr = MPI_File_seek(fh, this->elmdist(myid) * sizeof(bool), MPI_SEEK_SET); CHKERRQ(ierr);
  ierr = MPI_File_read_all(fh, this->active.data(), this->nLocElem,
                            MPI::BOOL, MPI_STATUS_IGNORE); CHKERRQ(ierr);
  ierr = MPI_File_close(&fh); CHKERRQ(ierr);
                                                                                                     
  // Initial design values
  xIni.setOnes(nLocElem); xIni *= 0.5;
  ierr = VecPlaceArray(this->x, xIni.data()); CHKERRQ(ierr); 
  for (unsigned int pi = 0; pi < this->penalties.size(); pi++)
  {
    this->penal = this->penalties[pi];
    stringstream strmid; strmid << this->penal;
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
  
/********************************************************************
 * Create Base Mesh
 * 
 * @param dimensions: Lower and upper limit for each dimension of domain
 * @param Nel: Number of elements in each dimension
 * @param Rmin: Minimum filter radius
 * @param Rmax: Maximum filter radius
 * @param Reorder_Mesh: flag indicating whether to use ParMETIS or not
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode TopOpt::CreateMesh(VectorXPS dimensions, ArrayXPI Nel,
                                  double Rmin, double Rmax,
                                  bool Reorder_Mesh)
{
  PetscErrorCode ierr = 0;

  if (this->verbose >= 3) {
    ierr = PetscFPrintf(this->comm, this->output, "Generating mesh\n"); CHKERRQ(ierr);
  }

  this->SetDimension(Nel.size());
  Nel.conservativeResize(3);
  for (int i = numDims; i < 3; i++)
    Nel(i) = 1;
  regular = 1;

  /// Constitutive matrix
  switch (this->numDims) {
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
    last[dim] = Nel(dim);
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
  for (PetscInt layer = first[2]; layer < last[2]; layer++) { // Loop through z-dimension
    for (PetscInt row = first[1]; row < last[1]; row++) { // Loop through y-dimension
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
  if (this->numDims > 1) {
    this->element.col(2) = this->element.col(1) + Nnd(0);
    this->element.col(3) = this->element.col(0) + Nnd(0);
  }
  // Add nodes 5 through 8 to all elements for 3D analysis
  if (this->numDims > 2) {
    this->element.block(0, 4, this->element.rows(), 4) =
      this->element.block(0, 0, this->element.rows(), 4) + (Nnd(0)*Nnd(1));
  }

  /// Element Distribution Information
  this->elmdist.setZero(nprocs+1);
  this->elmdist(myid) = first[0] + first[1]*Nel(0) + first[2]*Nel(0)*Nel(1);
  ierr = MPI_Allgather(MPI_IN_PLACE, 0, MPI_PETSCINT, elmdist.data(), 1,
                       MPI_PETSCINT, comm); CHKERRQ(ierr);
  elmdist(nprocs) = nElem;

  if (this->verbose >= 3) {
    ierr = PetscFPrintf(this->comm, this->output, "Successfully generated %i "
                        "elements\n", this->nElem); CHKERRQ(ierr);
  }

  /// Create the filter
  // dx is edgelength of elements in each direction
  double dx[3] = {0, 0, 0};
  elemSize.setOnes(1);
  for (int i = 0; i < this->numDims; i++) {
    dx[i] = (dimensions(2*i+1) - dimensions(2*i))/Nel(i);
    elemSize *= dx[i];
  }
  ArrayXPI MinFI, MinFJ, MaxFI, MaxFJ;
  ArrayXPS MinFK, MaxFK;
  ierr = RecFilter(first, last, dx, Rmin, Nel, MinFI, MinFJ, MinFK); CHKERRQ(ierr);

  if (this->verbose >= 3) {
    ierr = PetscFPrintf(this->comm, this->output, "Successfully generated "
                        "min scale filter\n"); CHKERRQ(ierr);
  }

  // Maximum length scale filter
  ierr = RecFilter(first, last, dx, Rmax, Nel, MaxFI, MaxFJ, MaxFK, 1); CHKERRQ(ierr);

  if (this->verbose >= 3) {
    ierr = PetscFPrintf(this->comm, this->output, "Successfully generated max "
                        "scale filter\n"); CHKERRQ(ierr);
  }

  // Create the geometric coarse-grid restrictions
  PetscInt mg_levels = (PetscInt)std::log2(Nel.segment(0, this->numDims).minCoeff());
  ierr = PetscOptionsGetInt(NULL, "kuf_", "-pc_mg_levels",
                            &mg_levels, NULL); CHKERRQ(ierr);
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
  if (this->numDims > 1) {
    Eigen::MatrixXd temp = Eigen::RowVectorXd::LinSpaced(nLocNode[(1)],
                           dx[1]*first[1]+dimensions(2),dx[1]*last[1]+dimensions(2))
                           .replicate(nLocNode[0],1);
    temp.resize(nLocNode[0]*nLocNode[1],1);
    this->node.col(1) = temp.replicate(nLocNode[2],1);
  }
  if (this->numDims > 2) {
    Eigen::MatrixXd temp = Eigen::RowVectorXd::LinSpaced(nLocNode[2],
                           dx[2]*first[2]+dimensions(4),dx[2]*last[2]+dimensions(4))
                           .replicate(nLocNode[0],1).replicate(nLocNode[1],1);
    temp.resize(this->nLocNode,1);
    this->node.col(2) = temp;
  }

  if (this->verbose >= 3) {
    ierr = PetscFPrintf(this->comm, this->output, "Successfully generated %i nodes\n",
                        this->nNode); CHKERRQ(ierr);
  }

  // Undo changes to range information
  if (myid != 0)
      first[this->numDims-1]--;
  for (int i = this->numDims; i < 3; i++)
    first[i]--;

  /// Apply shape functions to base mesh
  // Start by determining center of every element
  MatrixXPS elemCenters = Eigen::ArrayXXd::Zero(this->nLocElem,this->numDims);
  elemCenters.col(0) = (Eigen::VectorXd::LinSpaced(last[0]-first[0],
                       dx[0]*first[0]+dimensions(0),dx[0]*last[0]+dimensions(0)-dx[0])
                       .replicate(last[1]-first[1],1)
                       .replicate(last[2]-first[2],1).array() + dx[0]/2).matrix();
  if (this->numDims > 1) {
    Eigen::MatrixXd temp = (Eigen::RowVectorXd::LinSpaced(last[1]-first[1],
                           dx[1]*first[1]+dimensions(2),dx[1]*last[1]+dimensions(2)-dx[1])
                           .replicate(last[0]-first[0],1).array() + dx[1]/2).matrix();
    temp.resize(temp.size(),1);
    elemCenters.col(1) = temp.replicate(last[2]-first[2],1);
  }
  if (this->numDims > 2) {
    Eigen::MatrixXd temp = (Eigen::RowVectorXd::LinSpaced(last[2]-first[2],
                           dx[2]*first[2]+dimensions(4),dx[2]*last[2]+dimensions(4)-dx[2])
                           .replicate(last[0]-first[0],1)
                           .replicate(last[1]-first[1],1).array() + dx[2]/2).matrix();
    temp.resize(temp.size(),1);
    elemCenters.col(2) = temp;
  }

  /// Remove unwanted elements
  // Number of ghost elements along processor boundaries
  int padding = 1;
  for (short dim = 0; dim < numDims-1; dim++)
    padding *= Nel(dim);
  // Global validity/numbering array
  Eigen::Array<bool, -1, 1> elemValidity = Eigen::Array<bool, -1, 1>::Ones(nLocElem);
  Domain(elemCenters, elemValidity, "Domain");

  if (this->verbose >= 3) {
    ierr = PetscFPrintf(this->comm, this->output, "Elements have been marked "
                        "for removal\n"); CHKERRQ(ierr);
  } 

  // Trim domain
  int nInterfaceNodes = 1;
  for (int dim = 1; dim < numDims; dim++)
    nInterfaceNodes *= Nel(dim-1)+1;
  ierr = ApplyDomain(elemValidity, padding, nInterfaceNodes, MinFI, MinFJ, MinFK,
                     MaxFI, MaxFJ, MaxFK, I, J, K, cList, mg_levels); CHKERRQ(ierr);

  if (this->verbose >= 3) {
    ierr = PetscFPrintf(this->comm, this->output, "Successfully trimmed "
                        "domain to %i elements and %i nodes\n",
                        this->nElem, this->nNode); CHKERRQ(ierr);
  }

  /// Get a better distribution of elements
  ierr = ReorderParMetis(Reorder_Mesh, MinFI, MinFJ, MinFK, MaxFI,
                         MaxFJ, MaxFK); CHKERRQ(ierr);
  double temp = elemSize(0);
  elemSize.setConstant(nLocElem, temp);
  if (this->verbose >= 3) {
    ierr = PetscFPrintf(this->comm, this->output, "Successfully "
                        "redistributed elements\n"); CHKERRQ(ierr);
  }

  /// Node Distribution and Interpolation reordering
  NodeDist(I, J, K, cList, mg_levels);
  if (this->verbose >= 3) {
    ierr = PetscFPrintf(this->comm, this->output, "Successfully "
                        "redistributed nodes\n"); CHKERRQ(ierr);
  }

  /// Interpolation matrix assembly
  PetscInt min_size = std::min(cList[mg_levels-2].size(), 5e3);
  ierr = PetscOptionsGetInt(NULL, "kuf_", "-pc_mg_proc_eq_limit",
                            &min_size, NULL); CHKERRQ(ierr);
  ierr = Assemble_Interpolation(I, J, K, cList, mg_levels, min_size);
  if (this->verbose >= 3) {
    ierr = PetscFPrintf(this->comm, this->output, "Successfully assembled "
                        "GMG operators\n"); CHKERRQ(ierr);
  }

  /// Establish Global Numbering
  gElem = ArrayXPI::LinSpaced(this->nLocElem, elmdist(myid), elmdist(myid+1)-1);

  // nLocNode was locally overwritten earlier
  gNode = ArrayXPI::LinSpaced(this->nLocNode, nddist(myid), nddist(myid+1)-1);

  /// Get any needed ghost information
  Expand_Elem();
  Expand_Node();
  if (this->verbose >= 3) {
    ierr = PetscFPrintf(this->comm, this->output, "Successfully added ghost "
                        "nodes/elements\n"); CHKERRQ(ierr);
  }

  /// Assign Ghost Info and create DV vectors
  ierr = Initialize_Vectors(); CHKERRQ(ierr);

  /// Assemble Filter matrices
  ierr = Assemble_Filter(this->P, MinFI, MinFJ, MinFK, true); CHKERRQ(ierr);
  ierr = Assemble_Filter(this->R, MaxFI, MaxFJ, MaxFK, false); CHKERRQ(ierr);

  // Create a vector of how many fewer elements edge elements have in
  // their max length scale radius
  ierr = MatCreateVecs(this->R, NULL, &this->REdge); CHKERRQ(ierr);
  ierr = MatGetRowSum(this->R, this->REdge); CHKERRQ(ierr);
  PetscScalar rowSumMax;
  ierr = VecMax(this->REdge, NULL, &rowSumMax); CHKERRQ(ierr);
  Vec Rtemp;
  ierr = VecDuplicate(this->REdge, &Rtemp); CHKERRQ(ierr);
  ierr = VecSet(Rtemp, rowSumMax); CHKERRQ(ierr);
  ierr = VecAYPX(this->REdge, -1, Rtemp); CHKERRQ(ierr);
  ierr = VecDestroy(&Rtemp); CHKERRQ(ierr);

  /// Local Element Numbering
  Localize();

  if (this->verbose >= 3) {
    ierr = PetscFPrintf(this->comm, this->output, "Mesh generation complete\n"); CHKERRQ(ierr);
  }

  return 0;
}

/********************************************************************
 * Cut elements and nodes from mesh if requested
 * 
 * @param elemValidity: Flag for each element to be included (True) or
 *                      removed (False), set on output
 * @param padding: Product of number of elements in each dimension
 *                 less than n, where n is the dimensionality of the problem
 * @param nInterfaceNodes: Same as padding, but for nodes
 * @param MinFI: row indices of minimum filter matrix triplets
 * @param MinFJ: column indices of minimum filter matrix triplets
 * @param MinFK: values of minimum filter matrix triplets
 * @param MaxFI: row indices of maximum filter matrix triplets
 * @param MaxFJ: column indices of maximum filter matrix triplets
 * @param MaxFK: values of maximum filter matrix triplets
 * @param I: list of row indices of each GMG projection operator
 * @param J: list of column indices of each GMG projection operator
 * @param K: list of values of each GMG projection operator
 * @param cList: List of coarse nodes on each level of GMG hierarchy
 * @param mg_levels: Number of levels in GMG hierarchy
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode TopOpt::ApplyDomain(Eigen::Array<bool, -1, 1> elemValidity,
                                   int padding, int nInterfaceNodes, ArrayXPI &MinFI,
                                   ArrayXPI &MinFJ, ArrayXPS &MinFK, ArrayXPI &MaxFI,
                                   ArrayXPI &MaxFJ, ArrayXPS &MaxFK, ArrayXPI *I,
                                   ArrayXPI *J, ArrayXPS *K, ArrayXPI *cList,
                                   int &mg_levels)
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
  if (myid == 0) {
    start = 0; finish = nLocElem;
    newElemNumber.setZero(nLocElem + padding);
  }
  else if (myid == nprocs-1) {
    start = padding; finish = nLocElem+padding;
    newElemNumber.setZero(nLocElem + padding);
  }
  else {
    start = padding; finish = nLocElem+padding;
    newElemNumber.setZero(nLocElem + 2*padding);
  }

  for (PetscInt el = 0; el < nLocElem; el++) {
    if (elemValidity(el)) {
      newElemNumber(el+start) = ++number(myid);
    }
  }

  if (this->nprocs > 1) {
    // Share how many are stored locally on this process
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_PETSCINT, number.data(), 1, MPI_PETSCINT, comm);
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
        MPI_Isend(zeros.data(), padding, MPI_PETSCINT, myid+1, 0, comm, &sendReq1);
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
  }

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

  if (this->nprocs > 1) {
    /// Send and receive new numbers of edge nodes to adjacent processes
    MPI_Request sendReq1 = MPI_REQUEST_NULL, sendReq2 = MPI_REQUEST_NULL,
                recReq1 = MPI_REQUEST_NULL, recReq2 = MPI_REQUEST_NULL;
    int sR1 = true, sR2 = true, rR1 = true, rR2 = true;
    // Container to use for communications with adjacent processes
    ArrayXPI Receptacle = ArrayXPI::Zero(4*nInterfaceNodes);
    // Share validity of edge nodes if this process has any, pass through otherwise
    MPI_Status sendStat1, sendStat2, recStat1, recStat2;
    if (nLocNode > 0)
    {
      if (myid > 0 && myid != nprocs-1)
      {
        // Upward send
        MPI_Isend(newNodeNumber.data()+nLocNode, 2*nInterfaceNodes,
                  MPI_PETSCINT, myid+1, 0, comm, &sendReq1);
        sR1 = 0;
        // Downward send
        MPI_Isend(newNodeNumber.data(), 2*nInterfaceNodes, MPI_PETSCINT,
                  myid-1, 1, comm, &sendReq2);
        sR2 = 0;
        // Receive from below
        MPI_Irecv(Receptacle.data(), 2*nInterfaceNodes, MPI_PETSCINT,
                  myid-1, 0, comm, &recReq1);
        rR1 = 0;
        // Receive from above
        MPI_Irecv(Receptacle.data()+2*nInterfaceNodes, 2*nInterfaceNodes, MPI_PETSCINT,
                  myid+1, 1, comm, &recReq2);
        rR2 = 0;
      }
      else if (myid == 0)
      {
        // Upward send
        MPI_Isend(newNodeNumber.data()+nLocNode-nInterfaceNodes, 2*nInterfaceNodes,
                  MPI_PETSCINT, myid+1, 0, comm, &sendReq1);
        sR1 = 0;
        // Receive from above
        MPI_Irecv(Receptacle.data()+2*nInterfaceNodes, 2*nInterfaceNodes, MPI_PETSCINT,
                  myid+1, 1, comm, &recReq2);
        rR2 = 0;

      }
      else // last process
      {
        // Downward send
        MPI_Isend(newNodeNumber.data(), 2*nInterfaceNodes, MPI_PETSCINT,
                  myid-1, 1, comm, &sendReq2);
        sR2 = 0;
        // Receive from below
        MPI_Irecv(Receptacle.data(), 2*nInterfaceNodes, MPI_PETSCINT,
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
            newNodeNumber.segment(0, 2*nInterfaceNodes) =
              newNodeNumber.segment(0, 2*nInterfaceNodes).max(
              Receptacle.segment(0, 2*nInterfaceNodes) );
          }
        }
        if (rR2 == 0)
        {
          MPI_Test(&recReq2, &rR2, &recStat2);
          if (rR2 == 1) // Just got the message from above
          {
            // Combine indicators from both processes
            newNodeNumber.segment(newNodeNumber.size()-2*nInterfaceNodes,
                                  2*nInterfaceNodes) = newNodeNumber.segment
              (newNodeNumber.size()-2*nInterfaceNodes, 2*nInterfaceNodes).max
              (Receptacle.segment(2*nInterfaceNodes, 2*nInterfaceNodes) );
          }
        }
      } while (!(sR1 && sR2 && rR1 && rR2));
    }
    else
    {
      // this process owns no nodes currently - receive from adjacent
      // processes and pass through
      ArrayXPI zeros = ArrayXPI::Zero(2*nInterfaceNodes);
      if (myid == 0)
      {
        rR1 = 1; sR1 = 0; rR2 = 0; sR2 = 1;
        // Upward send
        MPI_Isend(zeros.data(), 2*nInterfaceNodes,
                  MPI_PETSCINT, myid+1, 0, comm, &sendReq1);
        // Receive from above
        MPI_Irecv(Receptacle.data(), 2*nInterfaceNodes, MPI_PETSCINT,
                  myid+1, 1, comm, &recReq2);
      }
      else if (myid == nprocs-1)
      {
        rR2 = 1; sR2 = 0; rR1 = 0; sR1 = 1;
        // Downward send
        MPI_Isend(zeros.data(), 2*nInterfaceNodes, MPI_PETSCINT,
                  myid-1, 1, comm, &sendReq2);
        // Receive from below
        MPI_Irecv(Receptacle.data(), 2*nInterfaceNodes, MPI_PETSCINT,
                  myid-1, 0, comm, &recReq1);
      }
      else
      {
        rR2 = 0; sR2 = 0; rR1 = 0; sR1 = 0;
        // Receive from above
        MPI_Irecv(Receptacle.data()+2*nInterfaceNodes, 2*nInterfaceNodes, MPI_PETSCINT,
                  myid+1, 1, comm, &recReq2);
        // Receive from below
        MPI_Irecv(Receptacle.data(), 2*nInterfaceNodes, MPI_PETSCINT,
                  myid-1, 0, comm, &recReq1);
      }

      do {
        if (rR1 == 0)
        {
          MPI_Test(&recReq1, &rR1, &recStat1);
          if (rR1 == 1) // Just got the message from below
          {
            // Upward send
            MPI_Isend(Receptacle.data(), 2*nInterfaceNodes,
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
            MPI_Isend(Receptacle.data()+2*nInterfaceNodes, 2*nInterfaceNodes,
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
  }

  /// Validity of all nodes and elements has been determined
  /// Renumber local nodes
  number(myid) = 0;
  for (PetscInt nd = nddist(myid)-start; nd < nddist(myid+1)-start; nd++) {
    if (newNodeNumber(nd) > 0) {
      newNodeNumber(nd) = ++number(myid);
    }
  }
  // Share how many nodes are stored locally on this process
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_PETSCINT, number.data(), 1, MPI_PETSCINT, comm);

  /// Renumber nonlocal nodes
  // Nodes on higher-numbered processes
  for (PetscInt nd = nddist(myid+1)-start; nd < finish-start; nd++) {
    if (newNodeNumber(nd) > 0) {
      newNodeNumber(nd) = ++number(myid);
    }
  }
  number(myid) = 0;
  // Nodes on lower-numbered processes
  for (PetscInt nd = nddist(myid)-start-1; nd >= 0; nd--) {
    if (newNodeNumber(nd) > 0) {
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
  for (int el = 0; el < element.rows(); el++) {
    for (int nd = 0; nd < element.cols(); nd++) {
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
  allNodeNumber.segment(nddist(myid),nddist(myid+1)-nddist(myid)) = 
      newNodeNumber.segment(nddist(myid)-start,nddist(myid+1)-nddist(myid));
  // ArrayXPI displs = ArrayXPI::Zero(nprocs);
  // partial_sum(nddist.data(), nddist.data()+nprocs-1, displs.data()+1);
  // ArrayXPI sizes = nddist.segment(1,nprocs)-nddist.segment(0,nprocs);
  // MPI_Allgatherv(newNodeNumber.data()+nddist(myid)-start,
  //                nddist(myid+1)-nddist(myid), MPI_PETSCINT, allNodeNumber.data(),
  //                sizes.data(), displs.data(), MPI_PETSCINT, comm);
  MPI_Allreduce(MPI_IN_PLACE, allNodeNumber.data(), nNode, MPI_PETSCINT,
                MPI_SUM, comm);
  // Now update numbers in the lists
  for (int level = 0; level < mg_levels-1; level++)
  { 
    int IJKind = 0, cind = 0;
    for (int j = 0; j < I[level].size(); j++)
    {
      if ((allNodeNumber(I[level](j)) > 0) && (allNodeNumber(J[level](j)) > 0))
      {
        I[level](IJKind) = allNodeNumber(I[level](j))-1;
        J[level](IJKind) = allNodeNumber(J[level](j))-1;
        K[level](IJKind) = K[level](j);
        IJKind++;
      }
    }
    for (int j = 0; j < cList[level].size(); j++)
      if (allNodeNumber(cList[level](j)) > 0)
        cList[level](cind++) = allNodeNumber(cList[level](j))-1;
    I[level].conservativeResize(IJKind); J[level].conservativeResize(IJKind);
    K[level].conservativeResize(IJKind);
    cList[level].conservativeResize(cind);
    if (cind == 0)
      mg_levels = level;
  }
  allNodeNumber.resize(0); // Free up the space

  /// Get new element numbers for each triplet of the filter matrices
  PetscInt ghostStart, ghostEnd;
  if (MinFJ.size() > 0)
  {
    ghostStart = std::min(MinFJ.minCoeff(), MaxFJ.minCoeff());
    ghostEnd   = std::max(MinFJ.maxCoeff(), MaxFJ.maxCoeff());
  }
  else
  {
    ghostStart = elmdist(myid);
    ghostEnd = elmdist(myid+1)-1;
  }
  ArrayXPI allElemNumber = ArrayXPI::Zero(ghostEnd - ghostStart + 1);
  ierr = GetElemNumbers(ghostStart, ghostEnd, newElemNumber,
                        allElemNumber); CHKERRQ(ierr);

  int ind = 0;
  for (int i = 0; i < MinFI.size(); i++)
  {
    PetscInt newI = allElemNumber(MinFI(i) - ghostStart);
    PetscInt newJ = allElemNumber(MinFJ(i) - ghostStart);
    if (newI > 0 and newJ > 0)
    {
      MinFI(ind) = newI-1;
      MinFJ(ind) = newJ-1;
      MinFK(ind) = MinFK(i);
      ind++;
    }
  }
  MinFI.conservativeResize(ind);
  MinFJ.conservativeResize(ind);
  MinFK.conservativeResize(ind);

  ind = 0;
  for (int i = 0; i < MaxFI.size(); i++)
  {
    PetscInt newI = allElemNumber(MaxFI(i) - ghostStart);
    PetscInt newJ = allElemNumber(MaxFJ(i) - ghostStart);
    if (newI > 0 and newJ > 0)
    {
      MaxFI(ind) = newI-1;
      MaxFJ(ind) = newJ-1;
      MaxFK(ind) = MaxFK(i);
      ind++;
    }
  }
  MaxFI.conservativeResize(ind);
  MaxFJ.conservativeResize(ind);
  MaxFK.conservativeResize(ind);

  /// Reset element distribution array
  number(this->myid) = elmdist(this->myid); // Save this value for assembling Proj below
  elmdist.setZero(nprocs+1);
  elmdist(myid+1) = element.rows();
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_PETSCINT, elmdist.data()+1, 1, MPI_PETSCINT, comm);
  for (int id = 1; id <= nprocs; id++)
    elmdist(id) += elmdist(id-1);

  // Reset global and local element counts
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

/********************************************************************
 * Get partitioning with ParMETIS
 * 
 * @param Reorder_Mesh: Flag indicating whether to use ParMETIS or not
 * @param MinFI: row indices of minimum filter matrix triplets
 * @param MinFJ: column indices of minimum filter matrix triplets
 * @param MinFK: values of minimum filter matrix triplets
 * @param MaxFI: row indices of maximum filter matrix triplets
 * @param MaxFJ: column indices of maximum filter matrix triplets
 * @param MaxFK: values of maximum filter matrix triplets
 * @param nparts: Number of partitions to create
 * @param ncommonnodes: See ParMETIS manual for PartMeshKway
 * @param tpwgts: See ParMETIS manual for PartMeshKway
 * @param ubvec: See ParMETIS manual for PartMeshKway
 * @param opts: See ParMETIS manual for PartMeshKway
 * @param ncon: See ParMETIS manual for PartMeshKway
 * @param elmwgt: See ParMETIS manual for PartMeshKway
 * @param wgtflag: See ParMETIS manual for PartMeshKway
 * @param numflag: See ParMETIS manual for PartMeshKway
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
idx_t TopOpt::ReorderParMetis(bool Reorder_Mesh,
                              ArrayXPI &MinFI, ArrayXPI &MinFJ, ArrayXPS &MinFK,
                              ArrayXPI &MaxFI, ArrayXPI &MaxFJ, ArrayXPS &MaxFK,
                              idx_t nparts, idx_t ncommonnodes, real_t *tpwgts,
                              real_t *ubvec, idx_t *opts, idx_t ncon,
                              idx_t *elmwgt, idx_t wgtflag, idx_t numflag)
{
  PetscErrorCode ierr = 0;

  Eigen::Array<idx_t, -1, 1> partition =
    myid*Eigen::Array<idx_t, -1, 1>::Ones(nLocElem);

  if (Reorder_Mesh)
  {
    /// ParMetis won't work if some processors have zero elements, so perform
    /// an initial redistribution
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
    ElemDist(partition, MinFI, MinFJ, MinFK, MaxFI, MaxFJ, MaxFK);
  }
  
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
      ubvec[i] = 1.05+(real_t)nparts/nElem;
  }

  if (opts == NULL)                         //0 for default options
  { opts = new idx_t; opts[0] = 0;}

  if (tpwgts == NULL)                //Vertex weight in each subdomain
  {
    tpwgts = new real_t[ncon*nparts];
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
  {
    partition.setConstant(myid); METIS = METIS_OK;
  }
  delete[] ubvec;
  delete[] tpwgts;
  delete opts;

  if (METIS != METIS_OK)
  {
    std::cout << "Error partitioning matrix! Error code: " << METIS << "\n";
    return METIS;
  }

  ierr = ElemDist(partition, MinFI, MinFJ, MinFK, MaxFI, MaxFJ, MaxFK); CHKERRQ(ierr);

  return ierr;
}

/********************************************************************
 * Redistribute elements
 * 
 * @param partition: Indicates which process each local element 
 *                   should be moved to
 * @param MinFI: row indices of minimum filter matrix triplets
 * @param MinFJ: column indices of minimum filter matrix triplets
 * @param MinFK: values of minimum filter matrix triplets
 * @param MaxFI: row indices of maximum filter matrix triplets
 * @param MaxFJ: column indices of maximum filter matrix triplets
 * @param MaxFK: values of maximum filter matrix triplets
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode TopOpt::ElemDist(Eigen::Array<idx_t, -1, 1> &partition,
                                ArrayXPI &MinFI, ArrayXPI &MinFJ, ArrayXPS &MinFK,
                                ArrayXPI &MaxFI, ArrayXPI &MaxFJ, ArrayXPS &MaxFK)
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

  // Reorder the indices of the filter matrices
  // Permute = permutation vector, permute(i) = newi
  // Indices = vector indicating where this process can start assigning Elements
  //            on each process (i.e. global locations in the permute vector)
  ArrayXPI permute = ArrayXPI::Zero(partition.size());
  ArrayXPI indices = ArrayXPI::Zero(nprocs);
  indices.segment(1,nprocs-1) = transferSize.block(0, 0, nprocs-1, nprocs)
                                .rowwise().sum();
  partial_sum(indices.data(), indices.data()+nprocs, indices.data());
  indices += transferSize.block(0, 0, nprocs, myid).rowwise().sum();
  for (PetscInt i = 0; i < partition.rows(); i++)
  {
    permute(where(i)) = indices(partition(i))++;
  }

  /// Use a global Vec with ghost nodes to renumber filter matrix triplets
  PetscInt ghostStart, ghostEnd;
  if (MinFJ.size() > 0) {
    ghostStart = std::min(MinFJ.minCoeff(), MaxFJ.minCoeff());
    ghostEnd   = std::max(MinFJ.maxCoeff(), MaxFJ.maxCoeff());
  }
  else {
    ghostStart = elmdist(myid);
    ghostEnd  = elmdist(myid+1)-1;
  }
  ArrayXPI allElemNumber = ArrayXPI::Zero(ghostEnd - ghostStart + 1);
  ierr = GetElemNumbers(ghostStart, ghostEnd, permute,
                        allElemNumber); CHKERRQ(ierr);

  for (int i = 0; i < MinFI.size(); i++)
  {
    MinFI(i) = allElemNumber(MinFI(i) - ghostStart);
    MinFJ(i) = allElemNumber(MinFJ(i) - ghostStart);
  }

  for (int i = 0; i < MaxFI.size(); i++)
  {
    MaxFI(i) = allElemNumber(MaxFI(i) - ghostStart);
    MaxFJ(i) = allElemNumber(MaxFJ(i) - ghostStart);
  }

  // Update distribution across processes
  elmdist(myid+1) = element.rows();
  nLocElem = element.rows();
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, elmdist.data()+1,
                1, MPI_PETSCINT, comm);
  for (short i = 1; i <= nprocs; i++)
      elmdist(i) += elmdist(i-1);

  /// Move filter Triplets using same ideas as transferring elements
  /// Minimum length filter first
  // Initialize transfer Variables
  ArrayXPI Fpartition = ArrayXPI::Zero(MinFI.size());
  for (int i = 0; i < this->nprocs; i++)
  {
    Fpartition += (MinFI >= elmdist(i+1)).cast<PetscInt>();
  }
  where = EigLab::gensort(Fpartition).cast<PetscInt>();
  transferSize.setZero(nprocs,nprocs);
  ArrayXPI Icopy(MinFI.size()), Jcopy(MinFJ.size());
  ArrayXPS Kcopy(MinFK.size());
  for (PetscInt i = 0; i < Icopy.rows(); i++)
  {
    Icopy(i) = MinFI(where(i));
    Jcopy(i) = MinFJ(where(i));
    Kcopy(i) = MinFK(where(i));
    transferSize(Fpartition(i),myid)++;
  }

  // How many elements are transferred between each pair of processes
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_PETSCINT, transferSize.data(),
                nprocs, MPI_PETSCINT, comm);
  sendcnt = transferSize.col(myid).cast<int>();
  recvcnt = transferSize.row(myid).cast<int>();

  // Offsets in sent messages
  senddsp.setZero(nprocs);
  for (short i = 1; i < nprocs; i++)
      senddsp(i) = sendcnt(i-1) + senddsp(i-1);

  // Offsets in received messages
  recvdsp.setZero(nprocs);
  for (short i = 1; i < nprocs; i++)
      recvdsp(i) = recvcnt(i-1) + recvdsp(i-1);

  // The element transfer
  MinFI.resize(recvcnt.sum());
  MinFJ.resize(recvcnt.sum());
  MinFK.resize(recvcnt.sum());
  MPI_Alltoallv(Icopy.data(), sendcnt.data(), senddsp.data(),
                MPI_PETSCINT, MinFI.data(), recvcnt.data(),
                recvdsp.data(), MPI_PETSCINT, comm);
  MPI_Alltoallv(Jcopy.data(), sendcnt.data(), senddsp.data(),
                MPI_PETSCINT, MinFJ.data(), recvcnt.data(),
                recvdsp.data(), MPI_PETSCINT, comm);
  MPI_Alltoallv(Kcopy.data(), sendcnt.data(), senddsp.data(),
                MPI_PETSCSCALAR, MinFK.data(), recvcnt.data(),
                recvdsp.data(), MPI_PETSCSCALAR, comm);

  /// Maximum length filter next
  // Initialize transfer Variables
  Fpartition.setZero(MaxFI.size());
  for (int i = 0; i < this->nprocs; i++)
  {
    Fpartition += (MaxFI >= elmdist(i+1)).cast<PetscInt>();
  }
  where = EigLab::gensort(Fpartition).cast<PetscInt>();
  transferSize.setZero(nprocs,nprocs);
  Icopy.setZero(MaxFI.size()), Jcopy.setZero(MaxFJ.size());
  Kcopy.setZero(MaxFK.size());
  for (PetscInt i = 0; i < Icopy.rows(); i++)
  {
    Icopy(i) = MaxFI(where(i));
    Jcopy(i) = MaxFJ(where(i));
    Kcopy(i) = MaxFK(where(i));
    transferSize(Fpartition(i),myid)++;
  }

  // How many elements are transferred between each pair of processes
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_PETSCINT, transferSize.data(),
                nprocs, MPI_PETSCINT, comm);
  sendcnt = transferSize.col(myid).cast<int>();
  recvcnt = transferSize.row(myid).cast<int>();

  // Offsets in sent messages
  senddsp.setZero(nprocs);
  for (short i = 1; i < nprocs; i++)
      senddsp(i) = sendcnt(i-1) + senddsp(i-1);

  // Offsets in received messages
  recvdsp.setZero(nprocs);
  for (short i = 1; i < nprocs; i++)
      recvdsp(i) = recvcnt(i-1) + recvdsp(i-1);

  // The element transfer
  MaxFI.resize(recvcnt.sum());
  MaxFJ.resize(recvcnt.sum());
  MaxFK.resize(recvcnt.sum());
  MPI_Alltoallv(Icopy.data(), sendcnt.data(), senddsp.data(),
                MPI_PETSCINT, MaxFI.data(), recvcnt.data(),
                recvdsp.data(), MPI_PETSCINT, comm);
  MPI_Alltoallv(Jcopy.data(), sendcnt.data(), senddsp.data(),
                MPI_PETSCINT, MaxFJ.data(), recvcnt.data(),
                recvdsp.data(), MPI_PETSCINT, comm);
  MPI_Alltoallv(Kcopy.data(), sendcnt.data(), senddsp.data(),
                MPI_PETSCSCALAR, MaxFK.data(), recvcnt.data(),
                recvdsp.data(), MPI_PETSCSCALAR, comm);

  return ierr;
}

/********************************************************************
 * Get new element numbers for nonlocal elements for reordering filters
 * 
 * @param ghostStart: First column of global filter used by this process
 * @param ghostStart: Last column of global filter used by this process
 * @param newElemNumber: New element numberings for local elements
 * @param allElemNumber: New element numberings for each column of local
 *                       part of filter matrix (output)
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode TopOpt::GetElemNumbers(PetscInt ghostStart, PetscInt ghostEnd,
                                      ArrayXPI &newElemNumber,
                                      ArrayXPI &allElemNumber)
{
  PetscErrorCode ierr = 0;

  /// This function works by passing the new element numbers to adjacent
  /// processes in a continuous fashion until all processes have the
  /// new element numbers for every element in the global range
  /// ghost Start through ghostEnd (inclusive)
  
  // Set up necessary data structures for communication
  allElemNumber.segment(this->elmdist(this->myid) - ghostStart,
                        this->nLocElem) = newElemNumber;
  ArrayXPI lo2hi = newElemNumber, hi2lo = newElemNumber;
  ArrayXPI lo2hi_buf = ArrayXPI::Zero(0), hi2lo_buf = ArrayXPI::Zero(0);
  int lo2hi_tag = 0, hi2lo_tag = 1;
  MPI_Request lo2hi_req, hi2lo_req;
  MPI_Status lo2hi_stat, hi2lo_stat;
  int lo2hi_tgt = this->myid+1, hi2lo_tgt = this->myid-1;
  int lo2hi_src = hi2lo_tgt, hi2lo_src = lo2hi_tgt;
  int lo2hi_cnt, hi2lo_cnt;
  PetscInt lo = this->myid;
  PetscInt hi = this->myid+1;

  int max_pass = 0;
  while (ghostStart < this->elmdist(std::max(0, this->myid-max_pass)) ||
         ghostEnd >= this->elmdist(std::min(this->nprocs, this->myid+max_pass+1)))
    max_pass++;

  ierr = MPI_Allreduce(MPI_IN_PLACE, &max_pass, 1, MPI_INT, MPI_MAX, this->comm); CHKERRQ(ierr);
  for (int i = 0; i < max_pass; i++)
  {
    // Send up
    if (this->myid < this->nprocs-1) {
      ierr = MPI_Issend(lo2hi.data(), lo2hi.size(), MPI_PETSCINT, lo2hi_tgt,
                        lo2hi_tag, this->comm, &lo2hi_req); CHKERRQ(ierr);
    }
    // Send down
    if (this->myid > 0) {
      ierr = MPI_Issend(hi2lo.data(), hi2lo.size(), MPI_PETSCINT, hi2lo_tgt,
                        hi2lo_tag, this->comm, &hi2lo_req); CHKERRQ(ierr);
    }

    // Receive from above
    if (this->myid < this->nprocs-1) {
      ierr = MPI_Probe(hi2lo_src, hi2lo_tag, this->comm, &hi2lo_stat); CHKERRQ(ierr);
      ierr = MPI_Get_count(&hi2lo_stat, MPI_PETSCINT, &hi2lo_cnt); CHKERRQ(ierr);
      hi2lo_buf.resize(hi2lo_cnt);
      ierr = MPI_Recv(hi2lo_buf.data(), hi2lo_cnt, MPI_PETSCINT, hi2lo_src,
                      hi2lo_tag, this->comm, &hi2lo_stat); CHKERRQ(ierr);

      if (hi2lo_buf.size() > 0)
      {
        // Fill in the element numbers
        PetscInt size = std::max(0, std::min((PetscInt)hi2lo_buf.size(),
                                ghostEnd-this->elmdist(hi)+1));
        allElemNumber.segment(std::max(0, std::min(this->elmdist(hi),ghostEnd) - ghostStart), size) =
              hi2lo_buf.segment(0, size);
      }
    }
    // Receive from below
    if (this->myid > 0) {
      ierr = MPI_Probe(lo2hi_src, lo2hi_tag, this->comm, &lo2hi_stat); CHKERRQ(ierr);
      ierr = MPI_Get_count(&lo2hi_stat, MPI_PETSCINT, &lo2hi_cnt); CHKERRQ(ierr);
      lo2hi_buf.resize(lo2hi_cnt);
      ierr = MPI_Recv(lo2hi_buf.data(), lo2hi_cnt, MPI_PETSCINT, lo2hi_src,
                      lo2hi_tag, this->comm, &lo2hi_stat); CHKERRQ(ierr);

      if (lo2hi_buf.size() > 0)
      {
        // Fill in the element numbers
        PetscInt size = std::max(0, std::min((PetscInt)lo2hi_buf.size(),
                                this->elmdist(lo)-ghostStart));
        allElemNumber.segment(std::max(this->elmdist(lo-1)-ghostStart, 0), size) =
              lo2hi_buf.segment(lo2hi_buf.size() - std::max(size, 1), size);
      }
    }


    // Bookkeeping/preparing for the next iteration
    lo = std::max(lo-1, 0);
    hi = std::min(hi+1, this->nprocs);
    if (this->myid < this->nprocs-1) {
      ierr = MPI_Wait(&lo2hi_req, MPI_STATUS_IGNORE); CHKERRQ(ierr);
    }
    if (this->myid > 0) {
      ierr = MPI_Wait(&hi2lo_req, MPI_STATUS_IGNORE); CHKERRQ(ierr);
    }
    hi2lo = hi2lo_buf; lo2hi = lo2hi_buf;
  }

  return ierr;
}

/********************************************************************
 * Redistribute nodes
 * 
 * @param I: list of row indices of each GMG projection operator
 * @param J: list of column indices of each GMG projection operator
 * @param K: list of values of each GMG projection operator
 * @param cList: List of coarse nodes on each level of GMG hierarchy
 * @param mg_levels: Number of levels in GMG hierarchy
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode TopOpt::NodeDist(ArrayXPI *I, ArrayXPI *J, ArrayXPS *K,
                                ArrayXPI *cList, int mg_levels)
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

/********************************************************************
 * Capture surrounding elements on other processes
 * 
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
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

/********************************************************************
 * Capture surrounding nodes on other processes
 * 
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
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

/********************************************************************
 * Set up ghost communications for PETSc vectors
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
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
    ierr = VecDuplicate(V, &dVdrho); CHKERRQ(ierr);
    ierr = VecDuplicate(V, &dEdz); CHKERRQ(ierr);
    ierr = VecDuplicate(V, &dEsdz); CHKERRQ(ierr);

    /// Create design variable and density vectors to work with the filter
    ierr = VecCreateMPI(comm, nLocElem, nElem, &x); CHKERRQ(ierr);
    // V is usually the same as rho, but this allows for interpolation of rho to V
    ierr = VecDuplicate(x, &rho); CHKERRQ(ierr);
    ierr = VecDuplicate(x, &rhoq); CHKERRQ(ierr);
    ierr = VecDuplicate(x, &y); CHKERRQ(ierr);

    return ierr;
}

/********************************************************************
 * Convert global numberings to local numberings
 * 
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
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

    return ierr;
}

/********************************************************************
 * Get centroids of all elements
 * 
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
MatrixXPS TopOpt::GetCentroids()
{
  MatrixXPS elemCenters = Eigen::ArrayXXd::Zero(this->nLocElem,this->numDims);
  for (PetscInt el = 0; el < this->nLocElem; el++) {
    for (PetscInt nd = 0; nd < this->element.cols(); nd++) {
      elemCenters.row(el) += this->node.row(this->element(el, nd));
    }
  }
  elemCenters /= this->element.cols();

  return elemCenters;
}