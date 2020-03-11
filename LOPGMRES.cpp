#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include "LOPGMRES.h"
#include "EigLab.h"

using namespace std;

// BLAS routines
extern "C"{
double ddot_(const int *N, const double *x, const int *incx, const double *y, const int *incy);
void daxpy_(const int *N, const double *a, const double *x, const int *incx,
    const double *y, const int *incy);
}

//TODO: Investigate the whether eigensolver in Eigen is sufficient or if solvers
// in BLAS/LAPACK are necessary for performance (for subspace problem)

/********************************************************************
 * Main constructor
 * 
 * @param comm: MPI communicator for the object
 * 
 *******************************************************************/
LOPGMRES::LOPGMRES(MPI_Comm comm)
{
  this->comm = comm; Set_ID();
  PetscOptionsGetInt(NULL, NULL, "-LOPGMRES_Verbose", &verbose, NULL);
  PetscFOpen(this->comm, "stdout", "w", &output);
  file_opened = 1;
}

/********************************************************************
 * Main destructor
 * 
 *******************************************************************/
LOPGMRES::~LOPGMRES()
{
  PetscErrorCode ierr = PCDestroy(&this->pc); CHKERRV(ierr);
}

/********************************************************************
 * How much information to print
 * 
 * @param verbose: The higher the number, the more to print
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode LOPGMRES::Set_Verbose(PetscInt verbose)
{
  this->verbose = verbose;
  PetscErrorCode ierr = PetscOptionsGetInt(NULL, NULL, "-LOPGMRES_Verbose",
                                           &this->verbose, NULL);
  CHKERRQ(ierr);
  return 0;
}

/********************************************************************
 * Preps for computing the eigenmodes of the specified system
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode LOPGMRES::Compute_Init()
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Initializing compute structures\n"); CHKERRQ(ierr);

  ierr = PetscLogEventBegin(EIG_Comp_Init, 0, 0, 0, 0); CHKERRQ(ierr);
  // Clear any old eigenvectors
  if (nev_conv > 0)
    ierr = VecDestroyVecs(nev_conv, &phi); CHKERRQ(ierr);

  // Preallocate eigenvalues and eigenvectors at each level and get problem size
  Vec temp;
  Q.resize(1); AQ.resize(1); BQ.resize(1);
  ierr = MatCreateVecs(A[0], &temp, NULL); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(temp, Qsize, Q.data()); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(temp, Qsize, AQ.data()); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(temp, Qsize, BQ.data()); CHKERRQ(ierr);
  ierr = VecDestroy(&temp); CHKERRQ(ierr);
  lambda.setOnes(Qsize);
  ierr = VecGetSize(Q[0][0], &n); CHKERRQ(ierr);
  ierr = VecGetLocalSize(Q[0][0], &nlocal); CHKERRQ(ierr);

  // Determine size of search space
  ierr = PetscOptionsGetInt(NULL, NULL, "-LOPGMRES_jmin", &jmin, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-LOPGMRES_jmax", &jmax, NULL); CHKERRQ(ierr);
  if (jmin < 0)
    jmin = std::min(std::max(2*nev_req, 10), std::min(n/2,10));
  if (jmax < 0)
    jmax = std::min(std::max(4*nev_req, 25), std::min(n , 50));

  // Preallocate search space, work space, and eigenvectors
  ierr = VecDuplicateVecs(Q[0][0], jmax, &V); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(Q[0][0], jmax, &TempVecs); CHKERRQ(ierr);
  TempScal.setZero(std::max(jmax,Qsize));

  return 0;
}

/********************************************************************
 * Initialize the search space
 * 
 * @param j: Number of vectors to initialize
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode LOPGMRES::Initialize_V(PetscInt &j)
{
  PetscErrorCode ierr = 0;

  Vec Temp;
  ierr = VecDuplicate(V[0], &Temp); CHKERRQ(ierr);

  PetscBool start;
  ierr = PetscOptionsHasName(NULL, NULL, "-LOPGMRES_Static_Start", &start); CHKERRQ(ierr);
  if (start) {
    PetscInt first;
    PetscScalar *p_vec;
    ierr = MatGetOwnershipRange(A[0], &first, NULL); CHKERRQ(ierr);
    for (int ii = 0; ii < jmin; ii++) {
      ierr = VecGetArray(Temp, &p_vec); CHKERRQ(ierr);
      for (int jj = 0; jj < nlocal; jj++)
        p_vec[jj] = pow(jj+first+1,ii);
      ierr = VecRestoreArray(Temp, &p_vec); CHKERRQ(ierr);
      ierr = MatMult(this->A[0], Temp, V[ii]); CHKERRQ(ierr);
      ierr = Remove_NullSpace(this->A[0], V[ii]); CHKERRQ(ierr);
    }
    j = jmin;
  }
  else {  
    PetscRandom random;
    ierr = PetscRandomCreate(comm, &random); CHKERRQ(ierr);
    for (int ii = 0; ii < jmin; ii++) {
      ierr = VecSetRandom(Temp, random); CHKERRQ(ierr);
      ierr = MatMult(this->A[0], Temp, V[ii]); CHKERRQ(ierr);
      ierr = Remove_NullSpace(this->A[0], V[ii]); CHKERRQ(ierr);
    }
    ierr = PetscRandomDestroy(&random);
    j = jmin;
  }
  ierr = VecDestroy(&Temp); CHKERRQ(ierr);

  return 0;
}

/********************************************************************
 * Update all parts of the preconditioner after eigenvalue update
 * 
 * @param residual: The current residual vector
 * @param rnorm: Residual norm
 * @param Au_norm: Norm of A*Q
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode LOPGMRES::Update_Preconditioner(Vec residual,
                         PetscScalar &rnorm, PetscScalar &Au_norm)
{
  PetscErrorCode ierr = 0;

  ierr = VecNorm(residual, NORM_2, &rnorm); CHKERRQ(ierr);
  ierr = VecNorm(AQ[0][nev_conv], NORM_2, &Au_norm); CHKERRQ(ierr);

  return 0;
}

/********************************************************************
 * Update the search space
 * 
 * @param x: Vector to add to the search space
 * @param residual: The current residual vector
 * @param rnorm: Residual norm
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode LOPGMRES::Update_Search(Vec x, Vec residual, PetscReal rnorm)
{
  PetscErrorCode ierr = 0;
  
  Mat A;
  ierr = PCGetOperators(this->pc, &A, NULL); CHKERRQ(ierr);
  ierr = PCApply(this->pc, residual, x); CHKERRQ(ierr);
  ierr = Remove_NullSpace(A, x); CHKERRQ(ierr);

  return ierr;
}

/********************************************************************
 * Clean up after the compute phase
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode LOPGMRES::Compute_Clean()
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Cleaning up\n"); CHKERRQ(ierr);

  // Extract and sort converged eigenpairs from Q and lambda
  lambda.conservativeResize(nev_conv);
  MatrixXPS empty(0,0);
  Eigen::ArrayXi order = Sorteig(empty, lambda);
  if (nev_conv > 0) {
    ierr = VecDuplicateVecs(Q[0][0], nev_conv, &phi); CHKERRQ(ierr);
  }
  for (int ii = 0; ii < nev_conv; ii++)
  {
    ierr = VecCopy(Q[0][order(ii)], phi[ii]); CHKERRQ(ierr);
  }

  // Destroy eigenvectors at each level
  ierr = VecDestroyVecs(Qsize, Q.data());  CHKERRQ(ierr);
  ierr = VecDestroyVecs(Qsize, AQ.data()); CHKERRQ(ierr);
  ierr = VecDestroyVecs(Qsize, BQ.data()); CHKERRQ(ierr);
  AQ.resize(0); BQ.resize(0);

  // Destroy workspace and search space
  ierr = VecDestroyVecs(jmax, &V); CHKERRQ(ierr);
  ierr = VecDestroyVecs(jmax, &TempVecs); CHKERRQ(ierr);

  Q.resize(0);

  return 0;
}
