#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "TopOpt.h"
#include "EigLab.h"
#include "EigenInverse.h"

using namespace std;
typedef Eigen::Map< Eigen::Matrix<PetscScalar, 1, -1>, Eigen::Unaligned,
                    Eigen::InnerStride<-1> > Bmap;
typedef Eigen::Array<PetscScalar, -1, 1> ArrayXPS;

/********************************************************************
 * Set up the stiffness matrix and solver context
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode TopOpt::FEInitialize()
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3) {
    ierr = PetscFPrintf(comm, output, "Setting up FEM structures\n"); CHKERRQ(ierr);
  }

  // Number of nodes and number of dof per element
  Eigen::Array<short, -1, 1> NE(element.rows()), DE(element.rows());
  if (regular) {
    NE.setConstant(pow(2, numDims));
    DE.setConstant(NE(0)*numDims);
    ke.resize(1);
  }
  else {
    DE = numDims*NE;
    ke.resize(element.rows());
    // TODO: Assign number of nodes and dof per element for irregular elements
  }

  // Fixing dofs
  this->fixedDof.resize(this->supports.cast<short>().sum());
  int ind = 0;
  for (PetscInt i = 0; i < this->suppNode.rows(); i++) {
    for (short j = 0; j < this->numDims; j++) {
      if (supports(i,j))
        this->fixedDof(ind++) = this->numDims*this->gNode(this->suppNode(i))+j;
    }
  }
  this->fixedDof.conservativeResize(ind);
  this->nFixDof = ind;
  MPI_Allreduce(MPI_IN_PLACE, &this->nFixDof, 1, MPI_PETSCINT, MPI_SUM, comm);

  // Fixing dofs for eigenvalue analysis
  this->eigenFixedDof.resize(this->eigenSupports.cast<short>().sum());
  ind = 0;
  for (PetscInt i = 0; i < this->eigenSuppNode.rows(); i++) {
    for (short j = 0; j < this->numDims; j++) {
      if (this->eigenSupports(i,j))
        this->eigenFixedDof(ind++) = this->numDims*this->gNode(this->eigenSuppNode(i))+j;
    }
  }
  this->eigenFixedDof.conservativeResize(ind);
  this->nEigFixDof = ind;
  MPI_Allreduce(MPI_IN_PLACE, &this->nEigFixDof, 1, MPI_PETSCINT, MPI_SUM, comm);

  // Get free dof
  ArrayXPI AllDof = ArrayXPI::LinSpaced(numDims*nLocNode, numDims*nddist(myid),
                                        numDims*(nddist(myid+1))-1);
  freeDof = AllDof;
  PetscInt *it = set_difference(AllDof.data(), AllDof.data()+AllDof.size(),
              fixedDof.data(), fixedDof.data()+fixedDof.size(), freeDof.data());
  freeDof.conservativeResize(it - freeDof.data());
  nFreeDof = it - freeDof.data();
  MPI_Allreduce(MPI_IN_PLACE, &nFreeDof, 1, MPI_PETSCINT, MPI_SUM, comm);

  // Get dof with no springs or supports
  springlessDof = freeDof;
  springDof.resize(springs.size());
  ind = 0;
  for (PetscInt i = 0; i < springNode.rows(); i++) {
    for (short j = 0; j < numDims; j++) {
      if (springs(i,j) != 0)
        springDof(ind++) = numDims*gNode(springNode(i))+j;
    }
  }
  springDof.conservativeResize(ind);
  it = set_difference(freeDof.data(), freeDof.data()+freeDof.size(),
      springDof.data(), springDof.data()+springDof.size(), springlessDof.data());
  springlessDof.conservativeResize(it - springlessDof.data());
  nSpringDof = ind;
  MPI_Allreduce(MPI_IN_PLACE, &nSpringDof, 1, MPI_PETSCINT, MPI_SUM, comm);

  // Get Preallocations
  PetscInt nnz = 0; // Total nonzeros from this process
  std::vector<std::vector<PetscInt> > connectivity(nLocNode);
  PetscInt *onDiag  = new PetscInt[nLocNode];
  PetscInt *offDiag = new PetscInt[nLocNode];
  // Loop over all elements (including nonlocal) to determine connectivity of
  // locally-owned nodes
  for (int el = 0; el < element.rows(); el++) {
    for (int nd = 0; nd < NE(el); nd++) {
      PetscInt node = element(el,nd);
      if (node < nLocNode) {
        connectivity[node].insert(connectivity[node].end(),
          element.data()+NE(el)*el, element.data()+NE(el)*(el+1));
      }
    }
  }

  // Remove duplicates from connectivity, and fill in onDiag and offDiag
  for (int nd = 0; nd < nLocNode; nd++) {
    // Remove duplicates
    sort(connectivity[nd].begin(), connectivity[nd].end());
    vector<PetscInt>::iterator it = unique(connectivity[nd].begin(),
              connectivity[nd].end());
    int nCols = distance(connectivity[nd].begin(), it);

    // Determine row preallocation
    PetscInt &on = onDiag[nd];
    for (on = 0; on < nCols; on++) {
      if (connectivity[nd][on] >= nLocNode)
        break;
    }
    offDiag[nd] = nCols - on;
    nnz += numDims*nCols;
  }

  // Initialize K matrix
  ierr = MatCreate(comm, &this->K); CHKERRQ(ierr);
  ierr = MatSetSizes(this->K, numDims*nLocNode, numDims*nLocNode,
              numDims*nNode, numDims*nNode); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(this->K, "K_"); CHKERRQ(ierr);
  ierr = MatSetFromOptions(this->K); CHKERRQ(ierr);
  ierr = MatXAIJSetPreallocation(this->K, this->numDims, onDiag, offDiag, 0, 0);
    CHKERRQ(ierr);
  delete[] onDiag; delete[] offDiag;

  // Allocate space for the sparse matrix assembly values
  this->i.clear(); this->i.reserve(nnz);
  this->j.clear(); this->j.reserve(nnz);
  this->k.clear(); this->k.reserve(nnz);
  this->e.clear(); this->e.reserve(nnz);

  // Assemble element stiffness matrices for each element
  if (regular)
    this->ke[0] = LocalK(0);
  else {
    this->ke.resize(this->nLocElem);
    for (long el = 0; el < element.rows(); el++)
      this->ke[el] = LocalK(el);
  }

  // Create load vector
  PetscScalar *p_F;
  ierr = VecGetArray(this->F, &p_F); CHKERRQ(ierr);

  for (long i = 0; i < loads.rows(); i++) {
    for (short j = 0; j < numDims; j++)
      p_F[numDims*loadNode(i)+j] += loads(i,j);
  }
  ierr = VecRestoreArray(this->F, &p_F); CHKERRQ(ierr);
  // Start ghosting force vector
  ierr = VecGhostUpdateBegin(this->F, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  // Construct secondary K corresponding to any springs
  ierr = VecCreateMPI(comm, numDims*nLocNode, numDims*nNode, &spKVec); CHKERRQ(ierr);
  ierr = VecSet(spKVec, 0.0); CHKERRQ(ierr);
  for (int i = 0; i < springNode.rows(); i++) {
    ArrayXPI where = ArrayXPI::LinSpaced(numDims, numDims*gNode(springNode(i)),
                                         numDims*(gNode(springNode(i))+1)-1);
    ierr = VecSetValues(spKVec, numDims, where.data(),
                      springs.data()+numDims*i, ADD_VALUES); CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(spKVec); CHKERRQ(ierr);

  // Construct secondary (diagonal) K corresponding for fixing void dof
  ierr = VecDuplicate(this->spKVec, &this->MaxStiff); CHKERRQ(ierr);

  // Construct M vector for lumped masses
  ierr = VecCreateMPI(comm, numDims*nLocNode,
               numDims*nNode, &MLump); CHKERRQ(ierr);
  ierr = VecSet(MLump, 0.0); CHKERRQ(ierr);
  for (int i = 0; i < massNode.rows(); i++) {
    ArrayXPI where = ArrayXPI::LinSpaced(numDims, numDims*gNode(massNode(i)),
                                         numDims*(gNode(massNode(i))+1)-1);
    ierr = VecSetValues(MLump, numDims, where.data(),
                      masses.data()+numDims*i, ADD_VALUES); CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(MLump); CHKERRQ(ierr);

  // Create solver context
  ierr = KSPCreate(comm, &KUF); CHKERRQ(ierr);
  ierr = KSPSetType(KUF, KSPGMRES); CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(this->KUF, PETSC_TRUE); CHKERRQ(ierr);
  ierr = KSPSetTolerances(KUF, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    CHKERRQ(ierr);
  // Use unpreconditioned norm for convergence test
  ierr = KSPSetNormType(this->KUF, KSP_NORM_UNPRECONDITIONED); CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(KUF, "kuf_"); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(KUF); CHKERRQ(ierr);
  // Set Preconditioner
  PC pc; PCType pctype;
  ierr = KSPGetPC(KUF, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, PCGAMG); CHKERRQ(ierr); // SWITCH FOR GAMG AND MG!!
  ierr = PCSetOptionsPrefix(pc, "kuf_"); CHKERRQ(ierr);
  ierr = PCSetFromOptions(pc); CHKERRQ(ierr);
  // Set up geometric multigrid hierarchy if desired
  ierr = PCGetType(pc, &pctype); CHKERRQ(ierr);
  if (!strcmp(pctype, PCMG)) {
    PetscInt nlevels = this->PR.size()+1;
    ierr = PCMGSetLevels(pc, nlevels, NULL); CHKERRQ(ierr);
    ierr = PCMGSetType(pc, PC_MG_MULTIPLICATIVE); CHKERRQ(ierr);
    ierr = PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH); CHKERRQ(ierr);
    for (int i = 1; i < nlevels; i++) {
      ierr = PCMGSetInterpolation(pc, i, this->PR[nlevels-i-1]); CHKERRQ(ierr); }
  }

  // Finish ghosting force vector
  ierr = VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(spKVec); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(MLump); CHKERRQ(ierr);

  return 0;
}

/********************************************************************
 * Set up the linear system
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode TopOpt::FEAssemble()
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3) {
    ierr = PetscFPrintf(comm, output, "Assembling Stiffness matrix\n"); CHKERRQ(ierr);
  }

  // Grab element stiffnesses;
  const PetscScalar *p_E;
  ierr = VecGetArrayRead(this->E, &p_E); CHKERRQ(ierr);

  // Reassemble K
  ierr = MatZeroEntries(this->K); CHKERRQ(ierr);
  PetscInt node;
  MatrixXPS ke = p_E[0]*this->ke[0];
  std::vector<PetscInt> cols(element.cols());
  for (long el = 0; el < element.rows(); el++) {
    if (!regular)
      ke = p_E[el]*this->ke[el];
    else
      ke = p_E[el]*this->ke[0];
    for (short nd = 0; nd < element.cols(); nd++)
      cols[nd] = this->gNode(this->element(el, nd));
    for (short nd = 0; nd < element.cols(); nd++) {
      node = element(el,nd);
      if (node < this->nLocNode) {
        ierr = MatSetValuesBlocked(this->K, 1, this->gNode.data()+node,
          this->element.cols(), cols.data(), ke.data() +
          ke.rows()*this->numDims*(nd % this->element.cols()), ADD_VALUES);
        CHKERRQ(ierr);
      }
    }
  }
      
  ierr = MatAssemblyBegin(this->K, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(this->E, &p_E); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(this->K, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  // Apply Spring B.C.'s
  ierr = MatDiagonalSet(this->K, this->spKVec, ADD_VALUES); CHKERRQ(ierr);
  // Apply Dirichlet B.C.'s
  ierr = MatZeroRowsColumns(this->K, fixedDof.size(), fixedDof.data(), 1.0, U, F);
    CHKERRQ(ierr);
  // Put a 1 on the diagonal wherever a node is fully detached from the structure
  Vec Diagonal; PetscScalar *p_Diag;
  ierr = MatCreateVecs(this->K, NULL, &Diagonal); CHKERRQ(ierr);
  ierr = MatGetDiagonal(this->K, Diagonal); CHKERRQ(ierr);
  ierr = VecGetArray(Diagonal, &p_Diag); CHKERRQ(ierr);
  for (PetscInt i = 0; i < this->numDims*this->nLocNode; i++) {
    p_Diag[i] = (p_Diag[i] > 0) ? p_Diag[i] : 1;
  }
  ierr = VecRestoreArray(Diagonal, &p_Diag); CHKERRQ(ierr);
  ierr = MatDiagonalSet(this->K, Diagonal, INSERT_VALUES); CHKERRQ(ierr);
  
  // Set KSP operators
  ierr = KSPSetOperators(this->KUF, this->K, this->K); CHKERRQ(ierr);

  return ierr;
}

/********************************************************************
 * Solve the FEM problem
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode TopOpt::FESolve()
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
  {
    ierr = PetscFPrintf(comm, output, "Solving governing pde\n"); CHKERRQ(ierr);
  }

  // Get the precondtioner and make modifications if necessary
  PC pc; KSPType ksptype; PCType pctype;
  ierr = KSPGetType(KUF, &ksptype); CHKERRQ(ierr);
  ierr = KSPGetPC(KUF, &pc); CHKERRQ(ierr);
  ierr = PCGetType(pc, &pctype); CHKERRQ(ierr);
  if (this->verbose >= 2) {
    ierr = PetscFPrintf(comm, output, "Solving governing PDE with %s solver "
                  "using %s preconditioning\n", ksptype, pctype); CHKERRQ(ierr);
  }

  // Set near nullspace and strength of connection metric for gamg
  if (!strcmp(pctype,PCGAMG)) {
    ierr = PCSetCoordinates(pc, this->numDims, this->nLocNode, this->node.data());
      CHKERRQ(ierr);
    PetscReal threshold = std::pow(0.05, this->numDims);
    ierr = PCGAMGSetThreshold(pc, &threshold, 1); CHKERRQ(ierr);
  }

  // Select the smoothers we're using
  if (!strcmp(pctype,PCGAMG) || !strcmp(pctype,PCMG)) {
    ierr = PCSetUp(pc); CHKERRQ(ierr);
    PetscInt levels;
    KSP smooth_ksp; PC smooth_pc; KSPType smooth_ksp_type; PCType smooth_pc_type;
    ierr = PCMGGetLevels(pc, &levels); CHKERRQ(ierr);

    // Perform a direct solve on the coarse level for GMG (should be on one process)
    if (!strcmp(pctype,PCMG)) {
      KSP *sub_ksp; PC sub_pc; PetscInt blocks, first;
      ierr = PCMGGetCoarseSolve(pc, &smooth_ksp); CHKERRQ(ierr);
      ierr = KSPSetType(smooth_ksp, KSPPREONLY); CHKERRQ(ierr);
      ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
      ierr = PCSetType(smooth_pc, PCBJACOBI); CHKERRQ(ierr);
      ierr = PCSetUp(smooth_pc); CHKERRQ(ierr);
      ierr = KSPSetUp(smooth_ksp); CHKERRQ(ierr);
      ierr = PCBJacobiGetSubKSP(smooth_pc, &blocks, &first, &sub_ksp); CHKERRQ(ierr);
      if (blocks != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,
                                "blocks on this process, %D, is not one", blocks);
      ierr = KSPGetPC(sub_ksp[0], &sub_pc); CHKERRQ(ierr);
      ierr = PCSetType(sub_pc, PCLU); CHKERRQ(ierr);
      ierr = PCFactorSetShiftType(sub_pc, MAT_SHIFT_INBLOCKS); CHKERRQ(ierr);
      ierr = KSPSetTolerances(sub_ksp[0], PETSC_DEFAULT, PETSC_DEFAULT,
        PETSC_DEFAULT, 1); CHKERRQ(ierr);
      ierr = KSPSetType(sub_ksp[0], KSPPREONLY); CHKERRQ(ierr);
      ierr = PCSetUp(sub_pc); CHKERRQ(ierr);
    }
    else if (!strcmp(pctype,PCGAMG) && this->nFixDof == 0) {
      Mat A; PetscInt coarseSize;
      ierr = PCMGGetCoarseSolve(pc, &smooth_ksp); CHKERRQ(ierr);
      ierr = KSPSetType(smooth_ksp, KSPPREONLY); CHKERRQ(ierr);
      ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
      ierr = PCGetOperators(smooth_pc, &A, NULL); CHKERRQ(ierr);
      ierr = MatGetSize(A, &coarseSize, NULL); CHKERRQ(ierr);
      if (coarseSize < 100) { // This is an expensive solver but good if no Dirichlet BC
        ierr = PCSetType(smooth_pc, PCSHELL); CHKERRQ(ierr);
        ierr = CreateEigenShell(smooth_pc); CHKERRQ(ierr);
        ierr = PCShellSetSetUp(smooth_pc, EigenShellSetUp); CHKERRQ(ierr);
        ierr = PCShellSetApply(smooth_pc, EigenShellApply); CHKERRQ(ierr);
        ierr = PCShellSetDestroy(smooth_pc, EigenShellDestroy); CHKERRQ(ierr);
        ierr = PCShellSetName(smooth_pc, "Eigendecomposition Inverse"); CHKERRQ(ierr);
        ierr = PCSetUp(smooth_pc); CHKERRQ(ierr);
      }
    }

    // Verify that the requested smoothers are being used
    ierr = PCMGGetSmoother(pc, levels-1, &smooth_ksp); CHKERRQ(ierr);
    ierr = KSPGetType(smooth_ksp, &smooth_ksp_type); CHKERRQ(ierr);
    ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
    ierr = PCGetType(smooth_pc, &smooth_pc_type); CHKERRQ(ierr);
    if (strcmp(this->smoother.c_str(), smooth_ksp_type)) { // Specific smoother selected
      if (!strcmp(this->smoother.c_str(), KSPRICHARDSON)) { // Weighted jacobi
        for (int i = 1; i < levels; i++) {
          ierr = PCMGGetSmoother(pc, i, &smooth_ksp); CHKERRQ(ierr);
          ierr = KSPSetType(smooth_ksp, KSPRICHARDSON); CHKERRQ(ierr);
          ierr = KSPRichardsonSetScale(smooth_ksp, 5.0/10.0); CHKERRQ(ierr);
          ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
          ierr = PCSetType(smooth_pc, PCPBJACOBI); CHKERRQ(ierr);
        }
      }
      else if (!strcmp(this->smoother.c_str(), KSPCHEBYSHEV)) { // Chebyshev with SOR
        for (int i = 1; i < levels; i++) {
          ierr = PCMGGetSmoother(pc, i, &smooth_ksp); CHKERRQ(ierr);
          ierr = KSPSetType(smooth_ksp, KSPCHEBYSHEV); CHKERRQ(ierr);
          ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
          ierr = PCSetType(smooth_pc, PCSOR); CHKERRQ(ierr);
        }
      }
      ierr = PCMGGetSmoother(pc, levels-1, &smooth_ksp); CHKERRQ(ierr);
      ierr = KSPGetType(smooth_ksp, &smooth_ksp_type); CHKERRQ(ierr);
      ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
      ierr = PCGetType(smooth_pc, &smooth_pc_type); CHKERRQ(ierr);
    }
    if (this->verbose >= 2) {
      PetscFPrintf(comm, output, "Multigrid preconditioning is using %s "
                 "smoothing with %s preconditioning\n",
                  smooth_ksp_type, smooth_pc_type); CHKERRQ(ierr);
    }
  }

  // Isolate disconnected rigid bodies and set nullspace if necessary
  ierr = IsolateRigid(); CHKERRQ(ierr);
  ierr = SetMatNullSpace(); CHKERRQ(ierr);

  // Solve for displacements
  ierr = KSPSolve(this->KUF, this->F, this->U); CHKERRQ(ierr);

  // Check if we converged properly
  ierr = KSPGetConvergedReason(this->KUF, &KUF_reason); CHKERRQ(ierr);
  if (this->verbose >= 1) {
    PetscInt its;
    ierr = KSPGetIterationNumber(this->KUF, &its); CHKERRQ(ierr);
    ierr = PetscFPrintf(comm, output, "Solve for displacements %s after %i iterations"
                        " with reason: %i\n", KUF_reason < 0 ? "failed" : "succeeded",
                        its, KUF_reason); CHKERRQ(ierr);
  }

  ierr = VecGhostUpdateBegin(this->U, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(this->U, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  
  return 0;
}

/********************************************************************
 * Isolate disconnected features
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode TopOpt::IsolateRigid()
{
  PetscErrorCode ierr = 0;

  PC pc;
  KSP smoother;
  Mat A; Vec coarse, fine;
  PetscInt levels;
  PetscScalar *p_v, *p_U;
  // Get the coarsest level and create a vector of ones
  ierr = KSPGetPC(this->KUF, &pc); CHKERRQ(ierr);
  ierr = PCMGGetLevels(pc, &levels); CHKERRQ(ierr);
  ierr = PCMGGetSmoother(pc, 0, &smoother); CHKERRQ(ierr);
  ierr = KSPGetOperators(smoother, &A, NULL); CHKERRQ(ierr);
  ierr = MatCreateVecs(A, &coarse, NULL); CHKERRQ(ierr);
  ierr = VecSet(coarse, 1.0); CHKERRQ(ierr);
  // Project that vector of ones to the finest level
  for (PetscInt ii = 1; ii < levels; ii++) {
    ierr = PCMGGetInterpolation(pc, ii, &A); CHKERRQ(ierr);
    ierr = MatCreateVecs(A, NULL, &fine); CHKERRQ(ierr);
    ierr = MatMult(A, coarse, fine); CHKERRQ(ierr);
    ierr = VecDestroy(&coarse); CHKERRQ(ierr);
    coarse = fine;
  }
  // Where the projected vector is zero, zero out the displacements
  ierr = VecGetArray(coarse, &p_v); CHKERRQ(ierr);
  ierr = VecGetArray(this->U, &p_U); CHKERRQ(ierr);
  for (PetscInt i = 0; i < this->numDims*this->nLocNode; i++) {
    if (p_v[i] == 0)
      p_U[i] = 0;
    else if (this->KUF_reason == KSP_CONVERGED_ITERATING)
      p_U[i] = 1; // Haven't solved for u yet, set to 1 and unset in SetMatNullSpace
    else if (p_U[i] == 0)
      p_U[i] = 1e-12; // Previously detached, but not anymore
  }
  ierr = VecRestoreArray(coarse, &p_v); CHKERRQ(ierr);
  ierr = VecRestoreArray(this->U, &p_U); CHKERRQ(ierr);

  return ierr;
}

/********************************************************************
 * Set the nullspace when no Dirichlet BC are applied
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode TopOpt::SetMatNullSpace()
{
  PetscErrorCode ierr = 0;
  if (this->nFixDof > 0) // Dirichlet BC are applied, shouldn't be a nullspace
    return ierr;

  // Set the matrix nullspace (particularly important if no Dirichlet BC)
  Vec *NullVecs; PetscInt nRBM = (this->numDims)*(this->numDims+1)/2;
  ierr = VecDuplicateVecs(this->U, nRBM, &NullVecs); CHKERRQ(ierr);
  Eigen::InnerStride<-1> skip(numDims);
  Bmap RBMMap(NULL, 0, skip);
  PetscScalar *p_Vec;
  PetscInt mode = 0;
  // Translation modes
  for (PetscInt i = 0; i < this->numDims; i++) {
      ierr = VecSet(NullVecs[mode], 0); CHKERRQ(ierr);
      ierr = VecGetArray(NullVecs[mode], &p_Vec); CHKERRQ(ierr);
      new (&RBMMap) Bmap(p_Vec + i, this->node.rows(), skip);
      RBMMap.setOnes();
      ierr = VecRestoreArray(NullVecs[mode], &p_Vec); CHKERRQ(ierr);
      mode++;
  }
  // Rotation modes
  for (PetscInt i = 0; i < this->numDims; i++) {
    for (PetscInt j = i+1; j < this->numDims; j++) {
      ierr = VecSet(NullVecs[mode], 0); CHKERRQ(ierr);
      ierr = VecGetArray(NullVecs[mode], &p_Vec); CHKERRQ(ierr);
      new (&RBMMap) Bmap(p_Vec + i, this->node.rows(), skip);
      RBMMap = this->node.col(j);
      new (&RBMMap) Bmap(p_Vec + j, this->node.rows(), skip);
      RBMMap = -this->node.col(i);
      ierr = VecRestoreArray(NullVecs[mode], &p_Vec); CHKERRQ(ierr);
      mode++;
    }
  }

  // Adjust the diagonal and zero out detached parts of the rigid body modes
  PetscScalar *p_U, **p_Vecs;
  ierr = VecGetArray(this->U, &p_U); CHKERRQ(ierr);
  ierr = VecGetArrays(NullVecs, nRBM, &p_Vecs); CHKERRQ(ierr);
  for (PetscInt i = 0; i < this->numDims*this->nLocNode; i++) {
    if (p_U[i] == 0) {
      for (mode = 0; mode < nRBM; mode++)
        p_Vecs[mode][i] = 0;
    }
    else if (this->KUF_reason == KSP_CONVERGED_ITERATING)
      p_U[i] = 0; // See note in IsolateRigid
  }
  ierr = VecRestoreArray(this->U, &p_U); CHKERRQ(ierr);
  ierr = VecRestoreArrays(NullVecs, nRBM, &p_Vecs); CHKERRQ(ierr);

  // Normalize the nullspace vectors
  PetscScalar dots[nRBM-1];
  for (mode = 0; mode < nRBM; mode++) {
    ierr = VecMDot(NullVecs[mode], mode, NullVecs, dots); CHKERRQ(ierr);
    for (PetscInt i = 0; i < mode; i++) {dots[i] *= -1;}
    ierr = VecMAXPY(NullVecs[mode], mode, dots, NullVecs); CHKERRQ(ierr);
    ierr = VecNormalize(NullVecs[mode], NULL); CHKERRQ(ierr);
  }

  // Set the nullspace for the stiffness matrix
  MatNullSpace Rigid;
  ierr = MatNullSpaceCreate(this->comm, PETSC_FALSE, nRBM, NullVecs, &Rigid); CHKERRQ(ierr);
  ierr = MatSetNullSpace(this->K, Rigid); CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&Rigid); CHKERRQ(ierr);
  ierr = VecDestroyVecs(nRBM, &NullVecs); CHKERRQ(ierr);

  return ierr;
}

/********************************************************************
 * Create the element stiffness matrix
 * 
 * @param el: Element number
 * 
 * @return Ke: Element stiffness matrix
 * 
 *******************************************************************/
MatrixXPS TopOpt::LocalK(PetscInt el)
{
  // Nodes per element - this currently only works for rectangular elements
  int NE = pow(2, numDims);
  MatrixXPS Ke = MatrixXPS::Zero(numDims *NE, numDims *NE);
  MatrixXPS dNdxi;
  MatrixXPS coords(NE, numDims);
  ArrayXXPS GP = GaussPoints();
  for (int q = 0; q < GP.cols(); q++) {
      W[q] = 1;
      dNdxi = dN(GP.data() + q*numDims);
      for (int i = 0; i < NE; i++)
          coords.block(i, 0, 1, numDims) = node.block(element(el, i), 0, 1, numDims);
      MatrixXPS J = dNdxi * coords;
      MatrixXPS InvJ = J.inverse();
      detJ = J.determinant();
      MatrixXPS dNdx = InvJ*dNdxi;
      AssignB(dNdx, B[q]);
      AssignG(dNdx, G[q], GT[q]);
      Ke += W[q] * B[q].transpose() * d * B[q] * detJ;
  }

  return Ke;
}

/********************************************************************
 * Gaussian quadrature points for line/quad/hex elements
 * 
 * @return GP: Gauss points
 * 
 *******************************************************************/
ArrayXXPS TopOpt::GaussPoints()
{
  ArrayXXPS GP(numDims, (int)pow(2, numDims));
  switch (numDims) {
    case 1:
      GP(0,0) = -1;   GP(0,1) = 1;
      break;
    case 2:
      GP.col(0) << -1, -1;
      GP.col(1) <<  1, -1;
      GP.col(2) <<  1,  1;
      GP.col(3) << -1,  1;
      break;
    case 3:
      GP.col(0) << -1, -1, -1;
      GP.col(1) <<  1, -1, -1;
      GP.col(2) <<  1,  1, -1;
      GP.col(3) << -1,  1, -1;
      GP.col(4) << -1, -1,  1;
      GP.col(5) <<  1, -1,  1;
      GP.col(6) <<  1,  1,  1;
      GP.col(7) << -1,  1,  1;
      break;
  }
  GP *= 1/sqrt(3);
  return GP;
}

/********************************************************************
 * Shape function derivatives for line/quad/hex elements
 * 
 * @param gaussPoint: Coordinates of Gauss point
 * 
 * @return dNdxi: Shape function derivates at the Gauss point
 * 
 *******************************************************************/
MatrixXPS TopOpt::dN(PetscScalar *gaussPoint)
{
  // Shape function derivatives in parent coordinates
  MatrixXPS dNdxi(numDims, (PetscInt)pow(2, numDims));
  PetscScalar xi, eta, zeta;
  switch (numDims) {
    case 1:
      dNdxi(0,0) = -1.0/2;
      dNdxi(0,1) =  1.0/2;
      break;
    case 2:
      xi = gaussPoint[0]; eta = gaussPoint[1];
      dNdxi(0,0) = -1.0/4 * (1-eta); dNdxi(1,0) = -1.0/4 * (1-xi);
      dNdxi(0,1) =  1.0/4 * (1-eta); dNdxi(1,1) = -1.0/4 * (1+xi);
      dNdxi(0,2) =  1.0/4 * (1+eta); dNdxi(1,2) =  1.0/4 * (1+xi);
      dNdxi(0,3) = -1.0/4 * (1+eta); dNdxi(1,3) =  1.0/4 * (1-xi);
      break;
    case 3:
      xi = gaussPoint[0]; eta = gaussPoint[1]; zeta = gaussPoint[2];
      // N1 = 1/8*(1-xi)*(1-eta)*(1-zeta)
      dNdxi(0,0) = -1.0/8*(1-eta)*(1-zeta);
      dNdxi(1,0) = -1.0/8*(1-xi)*(1-zeta);
      dNdxi(2,0) = -1.0/8*(1-xi)*(1-eta);
      // N2 = 1/8*(1+xi)*(1-eta)*(1-zeta)
      dNdxi(0,1) =  1.0/8*(1-eta)*(1-zeta);
      dNdxi(1,1) = -1.0/8*(1+xi)*(1-zeta);
      dNdxi(2,1) = -1.0/8*(1+xi)*(1-eta);
      // N3 = 1/8*(1+xi)*(1+eta)*(1-zeta)
      dNdxi(0,2) =  1.0/8*(1+eta)*(1-zeta);
      dNdxi(1,2) =  1.0/8*(1+xi)*(1-zeta);
      dNdxi(2,2) = -1.0/8*(1+xi)*(1+eta);
      // N4 = 1/8*(1-xi)*(1+eta)*(1-zeta)
      dNdxi(0,3) = -1.0/8*(1+eta)*(1-zeta);
      dNdxi(1,3) =  1.0/8*(1-xi)*(1-zeta);
      dNdxi(2,3) = -1.0/8*(1-xi)*(1+eta);
      // N5 = 1/8*(1-xi)*(1-eta)*(1+zeta)
      dNdxi(0,4) = -1.0/8*(1-eta)*(1+zeta);
      dNdxi(1,4) = -1.0/8*(1-xi)*(1+zeta);
      dNdxi(2,4) =  1.0/8*(1-xi)*(1-eta);
      // N6 = 1/8*(1+xi)*(1-eta)*(1+zeta)
      dNdxi(0,5) =  1.0/8*(1-eta)*(1+zeta);
      dNdxi(1,5) = -1.0/8*(1+xi)*(1+zeta);
      dNdxi(2,5) =  1.0/8*(1+xi)*(1-eta);
      // N7 = 1/8*(1+xi)*(1+eta)*(1+zeta)
      dNdxi(0,6) =  1.0/8*(1+eta)*(1+zeta);
      dNdxi(1,6) =  1.0/8*(1+xi)*(1+zeta);
      dNdxi(2,6) =  1.0/8*(1+xi)*(1+eta);
      // N8 = 1/8*(1-xi)*(1+eta)*(1+zeta)
      dNdxi(0,7) = -1.0/8*(1+eta)*(1+zeta);
      dNdxi(1,7) =  1.0/8*(1-xi)*(1+zeta);
      dNdxi(2,7) =  1.0/8*(1-xi)*(1+eta);
      break;
  }
  return dNdxi;
}

/********************************************************************
 * Construct the B matrix at a specified Gauss point
 * 
 * @param dNdx: Shape function derivatives in mapped space
 * @param B: The B matrix
 * 
 * @return void
 * 
 *******************************************************************/
void TopOpt::AssignB(MatrixXPS &dNdx, MatrixXPS &B)
{
  switch (numDims)
  {
    case 1: {
      B.setZero(1, numDims*dNdx.cols());
      B = dNdx;
      break;
    }
    case 2: {
      B.setZero(3, numDims*dNdx.cols());
      Eigen::InnerStride<-1> skip(numDims*B.rows());
      Bmap rowInsert(NULL, 0, skip);
      new (&rowInsert)Bmap(B.data(), dNdx.cols(), skip);
      rowInsert = dNdx.row(0);
      new (&rowInsert)Bmap(B.data()+B.rows()+1, dNdx.cols(), skip);
      rowInsert = dNdx.row(1);
      new (&rowInsert)Bmap(B.data()+2, dNdx.cols(), skip);
      rowInsert = dNdx.row(1);
      new (&rowInsert)Bmap(B.data()+B.rows()+2, dNdx.cols(), skip);
      rowInsert = dNdx.row(0);
      break;
    }
    case 3: {
      B.setZero(6, numDims*dNdx.cols());
      Eigen::InnerStride<-1> skip(numDims*B.rows());
      Bmap rowInsert(NULL, 0, skip);
      new (&rowInsert)Bmap(B.data(), dNdx.cols(), skip);
      rowInsert = dNdx.row(0);
      new (&rowInsert)Bmap(B.data()+B.rows()+1, dNdx.cols(), skip);
      rowInsert = dNdx.row(1);
      new (&rowInsert)Bmap(B.data()+2*B.rows()+2, dNdx.cols(), skip);
      rowInsert = dNdx.row(2);
      new (&rowInsert)Bmap(B.data()+3, dNdx.cols(), skip);
      rowInsert = dNdx.row(1);
      new (&rowInsert)Bmap(B.data()+B.rows()+3, dNdx.cols(), skip);
      rowInsert = dNdx.row(0);
      new (&rowInsert)Bmap(B.data()+B.rows()+4, dNdx.cols(), skip);
      rowInsert = dNdx.row(2);
      new (&rowInsert)Bmap(B.data()+2*B.rows()+4, dNdx.cols(), skip);
      rowInsert = dNdx.row(1);
      new (&rowInsert)Bmap(B.data()+5, dNdx.cols(), skip);
      rowInsert = dNdx.row(2);
      new (&rowInsert)Bmap(B.data()+2*B.rows()+5, dNdx.cols(), skip);
      rowInsert = dNdx.row(0);
      break;
    }
  }
  return;
}

/********************************************************************
 * Construct the G matrix for a given Gauss point
 * 
 * @param dNdx: Shape function derivatives in mapped space
 * @param G: The G matrix
 * @param GT: The G matrix transposed
 * 
 * @return void
 * 
 *******************************************************************/
void TopOpt::AssignG(MatrixXPS &dNdx, MatrixXPS &G,
                     MatrixXPS &GT)
{
  int numDimsSquare = numDims*numDims;
  G.setZero(numDimsSquare, numDims*dNdx.cols());
  Eigen::InnerStride<-1> skip(numDims*G.rows());
  Bmap rowInsert(NULL, 0, skip);
  /// Outer iteration
  for (short i = 0; i < numDims; i++) {
    /// Inner iteration
    for (short j = 0; j < numDims; j++) {
      new (&rowInsert)Bmap(G.data()+(numDimsSquare+numDims)*i + j,
                           dNdx.cols(),skip);
      rowInsert = dNdx.row(j);
    }
  }
  GT = G.transpose();

  return;
}
