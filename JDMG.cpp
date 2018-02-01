#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include "JDMG.h"
#include "EigLab.h"

using namespace std;

PetscLogEvent EIG_Setup_Coarse, EIG_Comp_Coarse, EIG_MGSetup;
PetscLogEvent *EIG_ApplyOP1, *EIG_ApplyOP2, *EIG_ApplyOP3, *EIG_ApplyOP4;

// BLAS routines
extern "C"{
double ddot_(const int *N, const double *x, const int *incx, const double *y, const int *incy);
void daxpy_(const int *N, const double *a, const double *x, const int *incx,
    const double *y, const int *incy);
}

//TODO: Investigate the whether eigensolver in Eigen is sufficient or if solvers
// in BLAS/LAPACK are necessary for performance (for subspace problem)

/******************************************************************************/
/**                             Main constructor                             **/
/******************************************************************************/
JDMG::JDMG(MPI_Comm comm)
{
  this->comm = comm; Set_ID();
  epstr = 1e-3;
  PetscOptionsGetInt(NULL, NULL, "-JDMG_Verbose", &verbose, NULL);
  PetscFOpen(this->comm, "stdout", "w", &output);
  file_opened = 1;
}

/******************************************************************************/
/**                              Main destructor                             **/
/******************************************************************************/
JDMG::~JDMG()
{
}

/******************************************************************************/
/**                       How much information to print                      **/
/******************************************************************************/
PetscErrorCode JDMG::Set_Verbose(PetscInt verbose)
{
  this->verbose = verbose;
  Close_File();
  PetscErrorCode ierr = PetscOptionsGetInt(NULL, NULL, "-JDMG_Verbose", &this->verbose, NULL);
  CHKERRQ(ierr);
  return 0;
}

/******************************************************************************/
/**                      Creates multilevel hierarchy                        **/
/******************************************************************************/
PetscErrorCode JDMG::Create_Hierarchy()
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
  {
    ierr = PetscFPrintf(comm, output, "Creating operators at each level\n"); CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(EIG_Hierarchy, 0, 0, 0, 0); CHKERRQ(ierr);
  for (unsigned int i = 0; i < P.size(); i++)
  {
    if (this->A[i+1] == NULL)
    {
      ierr = MatPtAP(A[i], P[i], MAT_INITIAL_MATRIX, 1.0, A.data()+i+1); CHKERRQ(ierr);
    }
    if (this->B[i+1] == NULL)
    {
      ierr = MatPtAP(B[i], P[i], MAT_INITIAL_MATRIX, 1.0, B.data()+i+1); CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(EIG_Hierarchy, 0, 0, 0, 0); CHKERRQ(ierr);

  return 0;
}

/******************************************************************************/
/**        Preps for computing the eigenmodes of the specified system        **/
/******************************************************************************/
PetscErrorCode JDMG::Compute_Init()
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Initializing compute structures\n"); CHKERRQ(ierr);

  ierr = PetscLogEventBegin(EIG_Comp_Init, 0, 0, 0, 0); CHKERRQ(ierr);
  // Clear any old eigenvectors
  if (nev_conv > 0)
    ierr = VecDestroyVecs(nev_conv, &phi); CHKERRQ(ierr);

  // Check if MG cycle type was set at command line
  const PetscInt ct_length = 5;
  char cycle_type[ct_length];
  PetscBool cycle_type_set = PETSC_FALSE;
  ierr = PetscOptionsGetString(NULL, NULL, "-JDMG_Cycle_Type", cycle_type,
                               ct_length, &cycle_type_set); CHKERRQ(ierr);
  if (cycle_type_set)
  {
    for (int i = 0; i < ct_length; i++)
    {
      if (cycle_type[i] == '\0')
        break;
      cycle_type[i] = toupper(cycle_type[i]);
    }
    if (!strcmp(cycle_type, "FULL"))
      cycle = FMGCycle;
    else if (!strcmp(cycle_type, "V"))
      cycle = VCycle;
    else
      PetscPrintf(comm, "Bad JDMG_Cycle_Type given %s, should be \"FULL\" or \"V\"", cycle_type);
  }

  // Preallocate eigenvalues and eigenvectors at each level and get problem size
  Vec temp;
  Q.resize(levels); AQ.resize(levels); BQ.resize(levels);
  for (int ii = 0; ii < levels; ii++)
  {
    ierr = MatCreateVecs(A[ii], &temp, NULL); CHKERRQ(ierr);
    ierr = VecDuplicateVecs(temp, Qsize, Q.data()+ii); CHKERRQ(ierr);
    ierr = VecDuplicateVecs(temp, Qsize, AQ.data()+ii); CHKERRQ(ierr);
    ierr = VecDuplicateVecs(temp, Qsize, BQ.data()+ii); CHKERRQ(ierr);
    ierr = VecDestroy(&temp); CHKERRQ(ierr);
  }
  lambda.setOnes(Qsize);
  ierr = VecGetSize(Q[0][0], &n); CHKERRQ(ierr);
  ierr = VecGetLocalSize(Q[0][0], &nlocal); CHKERRQ(ierr);

  // Determine size of search space
  ierr = PetscOptionsGetInt(NULL, NULL, "-JDMG_jmin", &jmin, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-JDMG_jmax", &jmax, NULL); CHKERRQ(ierr);
  if (jmin < 0)
    jmin = std::min( std::max(2*nev_req, 10), std::min(n/2,10));
  if (jmax < 0)
    jmax = std::min( std::max(4*nev_req, 25), std::min(n , 50));

  // Preallocate search space, work space, and eigenvectors
  ierr = VecDuplicateVecs(Q[0][0], jmax, &V); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(Q[0][0], jmax, &TempVecs); CHKERRQ(ierr);
  TempScal.setZero(std::max(jmax,Qsize));
  
// Check for options in MG preconditioner
  ierr = PetscOptionsGetInt(NULL, NULL, "-JDMG_Jacobi_nSweep", &nsweep, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-JDMG_Jacobi_Weight", &w, NULL); CHKERRQ(ierr);
  // Preallocate for operators
  K.resize(levels); Acopy.resize(levels-1); Bcopy.resize(levels-1);
  Dlist.resize(levels-1);
  xlist.resize(levels);
  flist.resize(levels);
  QMatP.resize(levels-1);
  OPx.resize(levels-1);
  ierr = Setup_Coarse(); CHKERRQ(ierr);
  for (int ii = 0; ii < levels-1; ii++)
  {
    // Combined matrices at each level
    ierr = MatDuplicate(B[ii], MAT_DO_NOT_COPY_VALUES, K.data()+ii); CHKERRQ(ierr);
    ierr = MatZeroEntries(K[ii]); CHKERRQ(ierr);
    ierr = MatAXPY(K[ii], 1.0, A[ii], DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    /*ierr = MatDuplicate(K[ii], MAT_SHARE_NONZERO_PATTERN, Acopy.data()+ii); CHKERRQ(ierr);
    ierr = MatZeroEntries(Acopy[ii]); CHKERRQ(ierr);
    ierr = MatCopy(A[ii], Acopy[ii], DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);*/
    ierr = MatDuplicate(K[ii], MAT_SHARE_NONZERO_PATTERN, Bcopy.data()+ii); CHKERRQ(ierr);
    ierr = MatZeroEntries(Bcopy[ii]); CHKERRQ(ierr);
    ierr = MatCopy(B[ii], Bcopy[ii], DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    // Vectors
    ierr = MatCreateVecs(A[ii+1], xlist.data()+ii+1, flist.data()+ii+1); CHKERRQ(ierr);
    ierr = MatCreateVecs(A[ii], Dlist.data()+ii, OPx.data()+ii); CHKERRQ(ierr);
    ierr = VecDuplicateVecs(Q[ii][0], Qsize, QMatP.data()+ii); CHKERRQ(ierr);
  }

  // Prep coarse problem
  ierr = KSPCreate(comm, &ksp_coarse); CHKERRQ(ierr);
  ierr = KSPSetType(ksp_coarse, KSPPREONLY); CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp_coarse, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1); CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp_coarse, "JDMG_coarse_"); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp_coarse); CHKERRQ(ierr);
  PC pc;
  ierr = KSPGetPC(ksp_coarse, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, PCBJACOBI); CHKERRQ(ierr);
  ierr = PCSetOptionsPrefix(pc, "JDMG_coarse_"); CHKERRQ(ierr);
  ierr = PCSetFromOptions(pc); CHKERRQ(ierr);

  ierr = KSPSetOperators(ksp_coarse, K.back(), K.back()); CHKERRQ(ierr);
  ierr = KSPSetUp(ksp_coarse); CHKERRQ(ierr);
  PC sub_pc; KSP *sub_ksp;
  PetscInt blocks, first;
  ierr = PCBJacobiGetSubKSP(pc, &blocks, &first, &sub_ksp); CHKERRQ(ierr);
  if (blocks != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"blocks on this process, %D, is not one",blocks);
  ierr = KSPGetPC(sub_ksp[0], &sub_pc); CHKERRQ(ierr);
  ierr = PCSetType(sub_pc, PCLU); CHKERRQ(ierr);
  ierr = PCFactorSetShiftType(sub_pc, MAT_SHIFT_INBLOCKS); CHKERRQ(ierr);
  ierr = KSPSetTolerances(sub_ksp[0], PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1); CHKERRQ(ierr);
  ierr = KSPSetType(sub_ksp[0], KSPPREONLY); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EIG_Comp_Init, 0, 0, 0, 0); CHKERRQ(ierr);

  return 0;
}

/******************************************************************************/
/**             Setup the Mat and KSP objects for coarse solve               **/
/******************************************************************************/
PetscErrorCode JDMG::Setup_Coarse()
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Setting up coarse operator\n"); CHKERRQ(ierr);

  ierr = PetscLogEventBegin(EIG_Setup_Coarse, 0, 0, 0, 0); CHKERRQ(ierr);
  // Figure out where to put the new rows of the marix (should be on the last
  // process that has any values)
  PetscInt endrank = 0, rows, cols, lrows, lcols;
  ierr = MatGetSize(A.back(), &rows, &cols); CHKERRQ(ierr);
  ierr = MatGetLocalSize(A.back(), &lrows, &lcols); CHKERRQ(ierr);
  ncoarse = rows;
  nlcoarse = lrows; // Coarse problem size
  if (lrows > 0)
    endrank = myid;
  MPI_Allreduce(MPI_IN_PLACE, &endrank, 1, MPI_PETSCINT, MPI_MAX, comm);

  if (myid == endrank)
  {
    lrows += Qsize;
    lcols += Qsize;
  }
  // Initialize the matrix
  ierr = MatCreate(comm, &K.back()); CHKERRQ(ierr);
  ierr = MatSetSizes(K.back(), lrows, lcols, rows+Qsize, cols+Qsize); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(K.back(), "JDMG_K_"); CHKERRQ(ierr);
  ierr = MatSetFromOptions(K.back()); CHKERRQ(ierr);

  // Preallocation
  PetscInt rstart = 0, rend = 0, nz;
  std::vector<PetscInt> columns;
  std::vector<PetscInt>::iterator it;
  const PetscInt *cwork;
  const PetscScalar *vwork;
  ierr = MatGetOwnershipRange(A.back(), &rstart, &rend); CHKERRQ(ierr);
  PetscInt *onDiag  = new PetscInt[lrows];
  PetscInt *offDiag = new PetscInt[lrows];
  for (int ii = rstart; ii < rend; ii++)
  {
    // Get all columns in this row
    columns.resize(0);
    ierr = MatGetRow(A.back(), ii, &nz, &cwork, NULL); CHKERRQ(ierr);
    columns.insert(columns.end(), cwork, cwork+nz);
    ierr = MatRestoreRow(A.back(), ii, &nz, &cwork, NULL); CHKERRQ(ierr);
    ierr = MatGetRow(B.back(), ii, &nz, &cwork, NULL); CHKERRQ(ierr);
    columns.insert(columns.end(), cwork, cwork+nz);
    ierr = MatRestoreRow(B.back(), ii, &nz, &cwork, NULL); CHKERRQ(ierr);
    // Get unique columns in this row
    sort(columns.begin(), columns.end());
    it = unique(columns.begin(), columns.end());
    columns.resize(distance(columns.begin(), it));
    onDiag[ii]  = (myid==endrank)*Qsize;
    offDiag[ii] = (myid!=endrank)*Qsize;
    for (it = columns.begin(); it != columns.end(); it++)
    {
      if (*it >= rstart && *it < rend)
        onDiag[ii]++;
      else
        offDiag[ii]++;
    }
  }
  if (myid == endrank)
  {
    fill(onDiag+nlcoarse, onDiag+lrows, nlcoarse+1);
    fill(offDiag+nlcoarse, offDiag+lrows, nlcoarse);
  }
  MatXAIJSetPreallocation(K.back(), 1, onDiag, offDiag, NULL, NULL); CHKERRQ(ierr);
  delete[] onDiag; delete[] offDiag;

  // Add "Q" rows/columns
  if (myid == endrank)
  {
    ArrayXPI index1 = ArrayXPI::LinSpaced(rows, 0, rows-1);
    ArrayXPI index2 = ArrayXPI::LinSpaced(Qsize, rows, rows+Qsize-1);
    MatrixPS values = MatrixPS::Zero(rows, Qsize);
    ierr = MatSetValues(K.back(), rows, index1.data(), Qsize, index2.data(),
          values.data(), INSERT_VALUES);
    ierr = MatSetValues(K.back(), Qsize, index2.data(), rows, index1.data(),
          values.data(), INSERT_VALUES);
    for (int ii = 0; ii < Qsize; ii++){
      ierr = MatSetValue(K.back(), rows+ii, rows+ii, 1.0, INSERT_VALUES); CHKERRQ(ierr);}
  }
  // Add "A-sigma*B" chunk of matrix
  for (int ii = rstart; ii < rend; ii++)
  {
    ierr = MatGetRow(A.back(), ii, &nz, &cwork, &vwork); CHKERRQ(ierr);
    ierr = MatSetValues(K.back(), 1, &ii, nz, cwork, vwork, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(A.back(), ii, &nz, &cwork, &vwork); CHKERRQ(ierr);
    // This is added because the submatrix method didn't work
    ierr = MatGetRow(B.back(), ii, &nz, &cwork, &vwork); CHKERRQ(ierr);
    ierr = MatSetValues(K.back(), 1, &ii, nz, cwork, vwork, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(B.back(), ii, &nz, &cwork, &vwork); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(K.back(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(K.back(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  // Solution and rhs vectors at coarse level
  ierr = MatCreateVecs(K.back(), &x_end, &f_end); CHKERRQ(ierr);
  ierr = VecSet(f_end, 0.0); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EIG_Setup_Coarse, 0, 0, 0, 0); CHKERRQ(ierr);

  return 0;
}

/******************************************************************************/
/**              Computes the eigenmodes of the coarse system                **/
/******************************************************************************/
PetscErrorCode JDMG::Compute_Coarse()
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Solving for eigenvalues at coarse level\n"); CHKERRQ(ierr);

  ierr = PetscLogEventBegin(EIG_Comp_Coarse, 0, 0, 0, 0); CHKERRQ(ierr);
  // Initialize
  ierr = EPSCreate(comm, &eps_coarse); CHKERRQ(ierr);
  if (ncoarse < 500){
    ierr = EPSSetType(eps_coarse, EPSLAPACK); CHKERRQ(ierr);}
  else {
    ierr = EPSSetType(eps_coarse, EPSKRYLOVSCHUR); CHKERRQ(ierr);}
  ierr = EPSSetProblemType(eps_coarse, EPS_GHEP); CHKERRQ(ierr);
  if (tau == NUMERIC)
    ierr = EPSSetWhichEigenpairs(eps_coarse, EPS_TARGET_REAL);
  else if (tau == SM || tau == LM)
    ierr = EPSSetWhichEigenpairs(eps_coarse, EPS_LARGEST_MAGNITUDE);
  else if (tau == SR || tau == SA || tau == LR || tau == LA)
    ierr = EPSSetWhichEigenpairs(eps_coarse, EPS_LARGEST_REAL);
  CHKERRQ(ierr);

  // Prep
  if (tau == SM || tau == SR || tau == SA)
    ierr = EPSSetOperators(eps_coarse, B.back(), A.back());
  else
    ierr = EPSSetOperators(eps_coarse, A.back(), B.back());
  CHKERRQ(ierr);
  ierr = EPSSetDimensions(eps_coarse, nev_req, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = EPSSetTolerances(eps_coarse, 1e-6, PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = EPSSetOptionsPrefix(eps_coarse, "coarse_"); CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps_coarse); CHKERRQ(ierr);

  // Solve
  ierr = EPSSolve(eps_coarse); CHKERRQ(ierr); 
  nev_conv = 0;
  ierr = EPSGetConverged(eps_coarse, &nev_conv);
  nev_conv = std::min(nev_conv, nev_req);
  if (nev_conv == 0)
    SETERRQ(comm, PETSC_ERR_NOT_CONVERGED, "Coarse solver found no eigenvalues\n");

  EPSType ceps_type;
  EPSGetType(eps_coarse, &ceps_type);

  /// Pull out the converged eigenvectors
  for (short ii = 0; ii < nev_conv; ii++)
  {
    ierr = EPSGetEigenvector(eps_coarse, ii, Q.back()[ii], 0); CHKERRQ(ierr);
  }

  for (short ii = levels-2; ii >=0; ii--)
  {
    for (short jj = 0; jj < nev_conv; jj++)
      ierr = MatMult(P[ii], Q[ii+1][jj], Q[ii][jj]); CHKERRQ(ierr);
  }
  ierr = EPSDestroy(&eps_coarse); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EIG_Comp_Coarse, 0, 0, 0, 0); CHKERRQ(ierr);

  return 0;
}
/******************************************************************************/
/**                   Fill out the initial search space                      **/
/******************************************************************************/
PetscErrorCode JDMG::Initialize_V(PetscInt &j)
{
  PetscErrorCode ierr = 0;

  // Use Krylov-Schur on smallest grid to get a good starting point.
  PetscBool start;
  ierr = PetscOptionsHasName(NULL, NULL, "-JDMG_Static_Start", &start); CHKERRQ(ierr);
  if (start){
    PetscInt first;
    PetscScalar *p_vec;
    ierr = MatGetOwnershipRange(A[0], &first, NULL); CHKERRQ(ierr);
    for (int ii = 0; ii < jmin; ii++)
    {
      ierr = VecGetArray(V[ii], &p_vec); CHKERRQ(ierr);
      for (int jj = 0; jj < nlocal; jj++)
        p_vec[jj] = pow(jj+first+1,ii);
      ierr = VecRestoreArray(V[ii], &p_vec); CHKERRQ(ierr);
    }
    j = jmin;
  }
  else{
    ierr = Compute_Coarse(); CHKERRQ(ierr);
    j = std::min(nev_conv, jmax);
    for (int ii = 0; ii < j; ii++){
      ierr = VecCopy(Q[0][ii], V[ii]); CHKERRQ(ierr);
    }
  }

  sigma = 0;
  return 0;
}

/******************************************************************************/
/**                   Update parts of the preconditioner                     **/
/******************************************************************************/
PetscErrorCode JDMG::Update_Preconditioner(Vec residual, 
                     PetscScalar &rnorm, PetscScalar &Au_norm)
{
  PetscErrorCode ierr = 0;

  // Construct parts of the new operator
  for (int ii = 1; ii < levels; ii++)
  {
    ierr = MatMultTranspose(P[ii-1], Q[ii-1][nev_conv], Q[ii][nev_conv]); CHKERRQ(ierr);
    ierr = MatMultTranspose(P[ii-1], AQ[ii-1][nev_conv], AQ[ii][nev_conv]); CHKERRQ(ierr);
    ierr = MatMultTranspose(P[ii-1], BQ[ii-1][nev_conv], BQ[ii][nev_conv]); CHKERRQ(ierr);
  }

  // Add parts to coarse operators
  PetscScalar *p_BQ;
  PetscInt col = ncoarse + nev_conv;
  ArrayXPI rows = ArrayXPI::LinSpaced(nlcoarse, 0, nlcoarse-1);
  ierr = VecGetArray(BQ.back()[nev_conv], &p_BQ); CHKERRQ(ierr);
  ierr = MatSetValues(K.back(), 1, &col, nlcoarse, rows.data(),
              p_BQ, INSERT_VALUES); CHKERRQ(ierr);
  ierr = MatSetValues(K.back(), nlcoarse, rows.data(), 1, &col,
              p_BQ, INSERT_VALUES); CHKERRQ(ierr);
  ierr = MatSetValue(K.back(), col, col, 0.0, INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecRestoreArray(BQ.back()[nev_conv], &p_BQ); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(K.back(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = VecNorm(residual, NORM_2, &rnorm); CHKERRQ(ierr);
  ierr = VecNorm(AQ[0][nev_conv], NORM_2, &Au_norm); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(K.back(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  return 0;
}

/******************************************************************************/
/**                      Clean up after compute phase                        **/
/******************************************************************************/
PetscErrorCode JDMG::Compute_Clean()
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Cleaning up\n"); CHKERRQ(ierr);

  // Extract and sort converged eigenpairs from Q and lambda
  lambda.conservativeResize(nev_conv);
  MatrixPS empty(0,0);
  Eigen::ArrayXi order = Sorteig(empty, lambda);
  if (nev_conv > 0) {
    ierr = VecDuplicateVecs(Q[0][0], nev_conv, &phi); CHKERRQ(ierr); }
  for (int ii = 0; ii < nev_conv; ii++)
  {
    ierr = VecCopy(Q[0][order(ii)], phi[ii]); CHKERRQ(ierr);
  }

  // Destroy eigenvectors at each level
  for (int ii = 0; ii < levels; ii++)
  {
    ierr = VecDestroyVecs(Qsize, Q.data()+ii);  CHKERRQ(ierr);
    ierr = VecDestroyVecs(Qsize, AQ.data()+ii); CHKERRQ(ierr);
    ierr = VecDestroyVecs(Qsize, BQ.data()+ii); CHKERRQ(ierr);
    ierr = MatDestroy(K.data() + ii); CHKERRQ(ierr);
  }
  AQ.resize(0); BQ.resize(0);

  // Destroy workspace and search space
  ierr = VecDestroyVecs(jmax, &V); CHKERRQ(ierr);
  ierr = VecDestroyVecs(jmax, &TempVecs); CHKERRQ(ierr);

  // Destroy coarse problem
  ierr = KSPDestroy(&ksp_coarse); CHKERRQ(ierr);

  // Destroy Operators
  for (int ii = 0; ii < levels-1; ii++)
  {
    //ierr = MatDestroy(Acopy.data()+ii); CHKERRQ(ierr);
    ierr = MatDestroy(Bcopy.data()+ii); CHKERRQ(ierr);
    ierr = VecDestroy(xlist.data()+ii+1); CHKERRQ(ierr);
    ierr = VecDestroy(flist.data()+ii+1); CHKERRQ(ierr);
    ierr = VecDestroy(Dlist.data()+ii); CHKERRQ(ierr);
    ierr = VecDestroy(OPx.data()+ii); CHKERRQ(ierr);
    ierr = VecDestroyVecs(Qsize, QMatP.data()+ii); CHKERRQ(ierr);
  }
  ierr = VecDestroy(&x_end); CHKERRQ(ierr); ierr = VecDestroy(&f_end); CHKERRQ(ierr);
  Q.resize(0);

  return 0;
}

/******************************************************************************/
/**                Set up multigrid for correction equation                  **/
/******************************************************************************/
PetscErrorCode JDMG::MGSetup(Vec f, PetscReal fnorm)
{
  //TODO: Eliminate uncessary operations on Q, AQ, and BQ that aren't changing
  PetscErrorCode ierr = 0;
  ierr = PetscLogEventBegin(EIG_MGSetup, 0, 0, 0, 0); CHKERRQ(ierr);
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Setting up multigrid\n"); CHKERRQ(ierr);

  // Set Shift
  vicinity = fnorm/lambda(nev_conv) < epstr; sigma_old = sigma;
  if (vicinity)
    sigma = lambda(nev_conv);
  else if (tau == NUMERIC)
    sigma = tau_num;
  else if (nev_conv == 0)
    sigma = lambda(nev_conv);
  else
    sigma = lambda(nev_conv-1);

  // Coefficients for summing terms of diagonal
  ArrayPS sumcoeffs(6);
  //sumcoeffs << 1, -2, 1, -sigma, 2*sigma, -sigma;
  sumcoeffs << 1, -2, 1, -sigma, sigma, 0;
  PetscReal oldnorm = std::numeric_limits<PetscReal>::max();
  PetscInt fix = 0, maxfix = 5;

  // Create D
  for (int ii = 0; ii < levels-1; ii++)
  {
    Vec PAPD, PAQPD, PQAQPD, PBPD, PBQPD, PQBQPD, *ALLD, WORK;
    ierr = VecDuplicateVecs(Dlist[ii], 7, &ALLD); CHKERRQ(ierr);
    PAPD = ALLD[0]; PAQPD = ALLD[1]; PQAQPD = ALLD[2];
    PBPD = ALLD[3]; PBQPD = ALLD[4]; PQBQPD = ALLD[5]; WORK = ALLD[6];

    // Subtract full Matrices
    /*ierr = MatZeroEntries(K[ii]); CHKERRQ(ierr);
    ierr = MatCopy(Acopy[ii], K[ii], SAME_NONZERO_PATTERN); CHKERRQ(ierr);
    ierr = MatAXPY(K[ii], -sigma, Bcopy[ii], SAME_NONZERO_PATTERN); CHKERRQ(ierr);*/
    ierr = MatAXPY(K[ii], sigma_old-sigma, Bcopy[ii], SAME_NONZERO_PATTERN); CHKERRQ(ierr);

    // A terms
    ierr = MatGetDiagonal(A[ii], PAPD); CHKERRQ(ierr);
    ierr = VecPointwiseMult(PAQPD, AQ[ii][0], BQ[ii][0]); CHKERRQ(ierr);
    ierr = VecPointwiseMult(PQAQPD, BQ[ii][0], BQ[ii][0]); CHKERRQ(ierr);
    ierr = VecScale(PQAQPD, lambda(0)); CHKERRQ(ierr);
    for (int jj = 1; jj < nev_conv+1; jj++)
    {
      ierr = VecPointwiseMult(WORK, AQ[ii][jj], BQ[ii][jj]); CHKERRQ(ierr);
      ierr = VecAXPY(PAQPD, 1.0, WORK); CHKERRQ(ierr);
      ierr = VecPointwiseMult(WORK, BQ[ii][jj], BQ[ii][jj]); CHKERRQ(ierr);
      ierr = VecAXPY(PQAQPD, lambda(jj), WORK); CHKERRQ(ierr);
    }

    // B terms
    ierr = MatGetDiagonal(B[ii], PBPD); CHKERRQ(ierr);
    ierr = VecPointwiseMult(PBQPD, BQ[ii][0], BQ[ii][0]); CHKERRQ(ierr);
    // PBQPD = PQBQPD
    //ierr = VecPointwiseMult(PQBQPD, BQ[ii][0], BQ[ii][0]); CHKERRQ(ierr);
    for (int jj = 1; jj < nev_conv+1; jj++)
    {
      ierr = VecPointwiseMult(WORK, BQ[ii][jj], BQ[ii][jj]); CHKERRQ(ierr);
      ierr = VecAXPY(PBQPD, 1.0, WORK); CHKERRQ(ierr);
      //ierr = VecPointwiseMult(WORK, BQ[ii][jj], BQ[ii][jj]); CHKERRQ(ierr);
      //ierr = VecAXPY(PQBQPD, 1.0, WORK); CHKERRQ(ierr);
    }

    //Total diagonal
    ierr = VecSet(Dlist[ii], 0.0); CHKERRQ(ierr);
    ierr = VecMAXPY(Dlist[ii], 5, sumcoeffs.data(), ALLD); CHKERRQ(ierr);
    ierr = VecDestroyVecs(7, &ALLD); CHKERRQ(ierr);

    //Smoothers
    for (int jj = 0; jj < nev_conv+1; jj++)
    {
      ierr = VecCopy(AQ[ii][jj], QMatP[ii][jj]); CHKERRQ(ierr);
      ierr = VecAXPY(QMatP[ii][jj], -sigma, BQ[ii][jj]); CHKERRQ(ierr);
    }

    if ((ii + nev_conv == 0) && !vicinity)
    {
      Vec x;
      ierr = VecDuplicate(f, &x); CHKERRQ(ierr); 
      ArrayPS QMatQ = lambda.segment(0,nev_conv+1) - sigma;
      ierr = VecSet(x, 0.0); CHKERRQ(ierr);
      ierr = WJac(f, x, ii); CHKERRQ(ierr);
      ierr = ApplyOP(x, OPx[ii], ii); CHKERRQ(ierr);
      ierr = VecAYPX(OPx[ii], -1.0, f); CHKERRQ(ierr);
      PetscReal OPx_norm;
      ierr = VecNorm(OPx[ii], NORM_2, &OPx_norm); CHKERRQ(ierr);
      if (fix < maxfix && OPx_norm > 10*fnorm && OPx_norm < oldnorm)
      {
        fix++;
        PetscFPrintf(comm, output, "F norm: %8.8g\t, OP norm: %8.8g\t, old norm %8.8g\n", fnorm, OPx_norm, oldnorm);
        oldnorm = OPx_norm;
        ierr = PetscFPrintf(comm, output, "Bad shift parameter, increasing shift from %1.6g to %1.6g\n", sigma, sigma*10); CHKERRQ(ierr); 
        ierr = MatAXPY(K[ii], sigma-sigma_old, Bcopy[ii], SAME_NONZERO_PATTERN); CHKERRQ(ierr);
        sigma *= 10;
        sumcoeffs << 1, -2, 1, -sigma, 2*sigma, -sigma;
	// Reset the shifted operator
        ii--;
      }
      ierr = VecDestroy(&x); CHKERRQ(ierr);
    }
  }

  // Coarse scale solver 
  ierr = MatShift(); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EIG_MGSetup, 0, 0, 0, 0); CHKERRQ(ierr);

  return ierr;
}

/******************************************************************************/
/**                        Coarse solve for multigrid                        **/
/******************************************************************************/
PetscErrorCode JDMG::Coarse_Solve()
{
  PetscErrorCode ierr = 0;

  PetscScalar *p_Vec1, *p_Vec2;
  ierr = VecGetArray(flist.back(), &p_Vec1); CHKERRQ(ierr);
  ierr = VecGetArray(f_end, &p_Vec2); CHKERRQ(ierr);
  copy(p_Vec1, p_Vec1+nlcoarse, p_Vec2);
  ierr = VecRestoreArray(flist.back(), &p_Vec1); CHKERRQ(ierr);
  ierr = VecRestoreArray(f_end, &p_Vec2); CHKERRQ(ierr);
  ierr = KSPSolve(ksp_coarse, f_end, x_end); CHKERRQ(ierr);
  ierr = VecGetArray(xlist.back(), &p_Vec1); CHKERRQ(ierr);
  ierr = VecGetArray(x_end, &p_Vec2); CHKERRQ(ierr);
  copy(p_Vec2, p_Vec2+nlcoarse, p_Vec1);
  ierr = VecRestoreArray(xlist.back(), &p_Vec1); CHKERRQ(ierr);
  ierr = VecRestoreArray(x_end, &p_Vec2); CHKERRQ(ierr);

  return 0;
}

/******************************************************************************/
/**                   Subtracting matrices at coarse level                   **/
/******************************************************************************/
PetscErrorCode JDMG::MatShift()
{
  PetscErrorCode ierr = 0;

  PetscInt rstart = 0, rend = 0, nzA, nzB;
  const PetscInt *cworkA, *cworkB;
  const PetscScalar *vworkA, *vworkB;
  vector<PetscInt> allCols;
  vector<PetscScalar> allVals;
  vector<PetscInt>::iterator aCit;
  ierr = MatGetOwnershipRange(A.back(), &rstart, &rend); CHKERRQ(ierr);
  for (int ii = rstart; ii < rend; ii++)
  {
    // Get rows of each matrix
    ierr = MatGetRow(A.back(), ii, &nzA, &cworkA, &vworkA); CHKERRQ(ierr);
    ierr = MatGetRow(B.back(), ii, &nzB, &cworkB, &vworkB); CHKERRQ(ierr);
    // Combine indices
    allCols.resize(nzA+nzB);
    copy(cworkA, cworkA+nzA, allCols.data());
    copy(cworkB, cworkB+nzB, allCols.data()+nzA);
    sort(allCols.begin(), allCols.end());
    aCit = unique(allCols.begin(), allCols.end());
    // Combine Values
    allVals.resize(aCit-allCols.begin());
    fill(allVals.begin(), allVals.end(), 0.0);
    // Add A
    unsigned int col = 0;
    for (int jj = 0; jj < nzA; jj++)
    {
      while (col < allVals.size() && cworkA[jj] != allCols[col])
        col++;
      if (col == allVals.size())
        break;

      allVals[col] = vworkA[jj];
    }
    // Subtract sigma*B
    col = 0;
    for (int jj = 0; jj < nzB; jj++)
    {
      while (col < allVals.size() && cworkB[jj] != allCols[col])
        col++;
      if (col == allVals.size())
        break;

      allVals[col] -= sigma*vworkB[jj];
    }
    ierr = PetscLogFlops(nzB); CHKERRQ(ierr);

    // Insert into matrix
    ierr = MatSetValues(K.back(), 1, &ii, allVals.size(), allCols.data(),
        allVals.data(), INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(A.back(), ii, &nzA, &cworkA, &vworkA); CHKERRQ(ierr);
    ierr = MatRestoreRow(B.back(), ii, &nzB, &cworkB, &vworkB); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(K.back(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(K.back(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  return 0;
}

/******************************************************************************/
/**                         Apply combined operator                          **/
/******************************************************************************/
PetscErrorCode JDMG::ApplyOP(Vec x, Vec y, PetscInt level)
{
  PetscErrorCode ierr = 0;

  ArrayPS QMatQ = lambda.segment(0,nev_conv+1) - sigma;
  ierr = PetscLogEventBegin(EIG_ApplyOP[level], 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = PetscLogEventBegin(EIG_ApplyOP1[level], 0, 0, 0, 0); CHKERRQ(ierr);
  int one = 1, bn;
  ierr = VecGetLocalSize(x, &bn); CHKERRQ(ierr);
  ArrayPS PQBx(nev_conv+1), QMatPx(nev_conv+1);
  PetscScalar *p_x, **p_BQ, **p_QMatP;
  MPI_Request request1 = MPI_REQUEST_NULL, request2 = MPI_REQUEST_NULL;

  // Local dot products
  ierr = VecGetArray(x, &p_x); CHKERRQ(ierr);
  ierr = VecGetArrays(BQ[level], nev_conv+1, &p_BQ); CHKERRQ(ierr);
  ierr = VecGetArrays(QMatP[level], nev_conv+1, &p_QMatP); CHKERRQ(ierr);
  for (int ii = 0; ii < nev_conv+1; ii++)
  {
    PQBx(ii) =  -ddot_(&bn, p_x, &one, p_BQ[ii], &one);
    QMatPx(ii) = ddot_(&bn, p_x, &one, p_QMatP[ii], &one);
  }
  if (bn > 0)
    ierr = PetscLogFlops(2*(nev_conv+1)*(2*bn-1)); CHKERRQ(ierr);
  ierr = VecRestoreArray(x, &p_x); CHKERRQ(ierr);
  ierr = VecRestoreArrays(BQ[level], nev_conv+1, &p_BQ); CHKERRQ(ierr);
  ierr = VecRestoreArrays(QMatP[level], nev_conv+1, &p_QMatP); CHKERRQ(ierr);

  // Reduction for dot products
  MPI_Iallreduce(MPI_IN_PLACE, PQBx.data(), nev_conv+1, MPI_DOUBLE,
                MPI_SUM, MG_comms[level], &request1);
  MPI_Iallreduce(MPI_IN_PLACE, QMatPx.data(), nev_conv+1, MPI_DOUBLE,
                MPI_SUM, MG_comms[level], &request2);
  // This is so VecMAXPY doesn't throw an error when debugging
  #if defined(PETSC_USE_DEBUG)
  MPI_Wait(&request1, MPI_STATUS_IGNORE);
  MPI_Wait(&request2, MPI_STATUS_IGNORE);
  MPI_Bcast(PQBx.data(), nev_conv+1, MPI_DOUBLE, 0, comm);
  MPI_Bcast(QMatPx.data(), nev_conv+1, MPI_DOUBLE, 0, comm);
  #endif

  // Term 1
  ierr = PetscLogEventEnd(EIG_ApplyOP1[level], 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = PetscLogEventBegin(EIG_ApplyOP2[level], 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = MatMult(K[level], x, y); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EIG_ApplyOP2[level], 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = PetscLogEventBegin(EIG_ApplyOP3[level], 0, 0, 0, 0); CHKERRQ(ierr);

  // Term 2
  MPI_Wait(&request1, MPI_STATUS_IGNORE);
  ierr = PetscLogEventEnd(EIG_ApplyOP3[level], 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = PetscLogEventBegin(EIG_ApplyOP4[level], 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = VecMAXPY(y, nev_conv+1, PQBx.data(), QMatP[level]); CHKERRQ(ierr);

  // Terms 3 and 4
  MPI_Wait(&request2, MPI_STATUS_IGNORE);

  PQBx = PQBx.cwiseProduct(QMatQ);
  ierr = PetscLogFlops(QMatQ.size()); CHKERRQ(ierr);
  PQBx += QMatPx;
  PQBx *= -1;
  ierr = VecMAXPY(y, nev_conv+1, PQBx.data(), BQ[level]); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EIG_ApplyOP4[level], 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EIG_ApplyOP[level], 0, 0, 0, 0); CHKERRQ(ierr);

  return 0;
}
