#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include "PRINVIT.h"
#include "EigLab.h"

using namespace std;

PetscLogEvent EIG_Compute;
PetscLogEvent EIG_Initialize, EIG_Prep, EIG_Convergence, EIG_Expand, EIG_Update;
PetscLogEvent EIG_Comp_Init, EIG_Hierarchy, EIG_Precondition, EIG_Jacobi, *EIG_ApplyOP;

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
 *******************************************************************/
PRINVIT::PRINVIT()
{
  PetscErrorCode ierr = 0;
  // Determine size of search space
  jmin = -1, jmax = -1;
  Qsize = nev_req;
  ierr = PetscOptionsGetInt(NULL, NULL, "-PRINVIT_Verbose", &verbose, NULL); CHKERRV(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-PRINVIT_jmin", &jmin, &jmin_set); CHKERRV(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-PRINVIT_jmax", &jmax, &jmax_set); CHKERRV(ierr);
  this->V = NULL;
  TempVecs = NULL;
}

/********************************************************************
 * Main constructor
 * 
 *******************************************************************/
PRINVIT::~PRINVIT()
{
  PetscErrorCode ierr = 0;
  // Destroy workspace and search space
  if (V != NULL) {
    ierr = VecDestroyVecs(jmax, &V); CHKERRV(ierr);
    ierr = VecDestroyVecs(jmax, &TempVecs); CHKERRV(ierr);
  }
}

/********************************************************************
 * Set how much information the subroutines print
 * 
 * @param verbose: The higher the number, the more is printed
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode PRINVIT::Set_Verbose(PetscInt verbose)
{
  this->verbose = verbose;
  PetscErrorCode ierr = PetscOptionsGetInt(NULL, NULL, "-PRINVIT_Verbose", &this->verbose, NULL);
  CHKERRQ(ierr);
  return 0;
}

/********************************************************************
 * Set the operators that define the eigen system
 * 
 * @param A: The first matrix
 * @param B: The (optional) second matrix
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode PRINVIT::Set_Operators(Mat A, Mat B)
{
  PetscErrorCode ierr = 0;

  ierr = EigenPeetz::Set_Operators(A, B); CHKERRQ(ierr);
  ierr = Update_jmin(); CHKERRQ(ierr);
  ierr = Update_jmax(); CHKERRQ(ierr);
  ierr = Setup_Q(); CHKERRQ(ierr);

  return 0;
}

/********************************************************************
 * Set up the space to store eigenvector approximations
 * 
 * @param A: The first matrix
 * @param B: The (optional) second matrix
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode PRINVIT::Setup_Q()
{
  PetscErrorCode ierr = 0;

  // Preallocate eigenvector storage space
  Vec temp;
  Q.resize(this->A.size()); AQ.resize(this->A.size()); BQ.resize(this->A.size());
  for (int ii = 0; ii < this->A.size(); ii++) {
    ierr = MatCreateVecs(A[ii], &temp, NULL); CHKERRQ(ierr);
    ierr = VecDuplicateVecs(temp, Qsize, Q.data()+ii); CHKERRQ(ierr);
    ierr = VecDuplicateVecs(temp, Qsize, AQ.data()+ii); CHKERRQ(ierr);
    ierr = VecDuplicateVecs(temp, Qsize, BQ.data()+ii); CHKERRQ(ierr);
    ierr = VecDestroy(&temp); CHKERRQ(ierr);
  }
  lambda.setOnes(Qsize);
  ierr = VecGetSize(Q[0][0], &n); CHKERRQ(ierr);
  ierr = VecGetLocalSize(Q[0][0], &nlocal); CHKERRQ(ierr);

  return 0;
}

/********************************************************************
 * Destroy the space to store eigenvector approximations
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode PRINVIT::Destroy_Q()
{
  PetscErrorCode ierr = 0;

  // Destroy existing space
  for (PetscInt i = 0; i < this->Q.size(); i++) {
    ierr = VecDestroyVecs(Qsize, Q.data()+i); CHKERRQ(ierr);
    ierr = VecDestroyVecs(Qsize, AQ.data()+i); CHKERRQ(ierr);
    ierr = VecDestroyVecs(Qsize, BQ.data()+i); CHKERRQ(ierr);
  }
  Q.resize(0); AQ.resize(0); BQ.resize(0);

  return 0;
}

/********************************************************************
 * Sets target eigenvalues and number to find
 * 
 * @param tau: The target type (e.g. 'LM') or a target value
 * @param nev: The number of eigenmodes to calculate
 * @param ntype: The type of eigenvalues (e.g. 6 total, or 6 unique)
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode PRINVIT::Set_Target(Tau tau, PetscInt nev, Nev_Type ntype)
{
  PetscErrorCode ierr = 0;
  
  ierr = EigenPeetz::Set_Target(tau, nev, ntype); CHKERRQ(ierr);
  ierr = Update_jmin(); CHKERRQ(ierr);
  ierr = Update_jmax(); CHKERRQ(ierr);

  return ierr;
}
PetscErrorCode PRINVIT::Set_Target(PetscScalar tau, PetscInt nev, Nev_Type ntype)
{ 
  PetscErrorCode ierr = 0;

  ierr = EigenPeetz::Set_Target(tau, nev, ntype); CHKERRQ(ierr);
  ierr = Update_jmin(); CHKERRQ(ierr);
  ierr = Update_jmax(); CHKERRQ(ierr);

  return ierr;
}

/********************************************************************
 * Update the search space size
 * 
 * @param jmin: Minimum search space size
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode PRINVIT::Update_jmin(PetscInt jmin)
{
  if (jmin_set == PETSC_FALSE) {
    if (jmin > 0) {
      this->jmin = jmin;
      jmin_set = PETSC_TRUE;
    }
    else
      this->jmin = std::min(std::max(2*nev_req, 10), std::min(n/2,10));
  }

  return 0;
}

/********************************************************************
 * Update the search space size
 * 
 * @param jmin: Maximum search space size
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode PRINVIT::Update_jmax(PetscInt jmax)
{
  PetscErrorCode ierr = 0;

  PetscInt new_jmax;
  if (jmax_set == PETSC_FALSE) {
    if (jmax > 0) {
      new_jmax = jmax;
      jmax_set = PETSC_TRUE;
    }
    else
      new_jmax = std::min(std::max(4*nev_req, 25), std::min(n, 50));
    if (this->jmax == new_jmax)
      return 0;

    // Preallocate search space, work space, and eigenvectors
    if (V != NULL) {
      ierr = VecDestroyVecs(this->jmax, &V); CHKERRQ(ierr);
      ierr = VecDestroyVecs(this->jmax, &TempVecs); CHKERRQ(ierr);
    }
    this->jmax = new_jmax;
    j = 0;
    if (this->A.size() > 0) {
      Vec temp;
      ierr = MatCreateVecs(A[0], &temp, NULL); CHKERRQ(ierr);
      ierr = VecDuplicateVecs(temp, this->jmax, &V); CHKERRQ(ierr);
      ierr = VecDuplicateVecs(temp, this->jmax, &TempVecs); CHKERRQ(ierr);
      TempScal.setZero(std::max(this->jmax, Qsize));
      ierr = VecDestroy(&temp); CHKERRQ(ierr);
    }
  }

  return 0;
}

/********************************************************************
 * Initialize the search space
 * 
 * @return ierr: PetscErrorCode
 * 
 * @options: -PRINVIT_Static_Start: Use nonrandom initial search space
 * 
 *******************************************************************/
PetscErrorCode PRINVIT::Initialize_V()
{
  PetscErrorCode ierr = 0;

  // Use Krylov-Schur on smallest grid to get a good starting point.
  PetscBool start;
  ierr = PetscOptionsHasName(NULL, NULL, "-PRINVIT_Static_Start", &start); CHKERRQ(ierr);
  if (start) {
    PetscInt first;
    PetscScalar *p_vec;
    ierr = MatGetOwnershipRange(A[0], &first, NULL); CHKERRQ(ierr);
    for (int ii = j; ii < jmin; ii++) {
      ierr = VecGetArray(V[ii], &p_vec); CHKERRQ(ierr);
      for (int jj = 0; jj < nlocal; jj++)
        p_vec[jj] = pow(jj+first+1,ii);
      ierr = VecRestoreArray(V[ii], &p_vec); CHKERRQ(ierr);
    }
    j = std::max(jmin, j);
  }
  else {  
    PetscRandom random;
    ierr = PetscRandomCreate(comm, &random); CHKERRQ(ierr);
    Vec Temp;
    ierr = VecDuplicate(V[0], &Temp); CHKERRQ(ierr);
    for (int ii = j; ii < jmin; ii++) {
      ierr = VecSetRandom(Temp, random); CHKERRQ(ierr);
      ierr = MatMult(this->A[0], Temp, V[ii]); CHKERRQ(ierr);
      ierr = Remove_NullSpace(this->A[0], V[ii]); CHKERRQ(ierr);
    }
    ierr = PetscRandomDestroy(&random);
    ierr = VecDestroy(&Temp); CHKERRQ(ierr);
    j = std::max(jmin, j);
  }

  return 0;
}

/********************************************************************
 * Computes the eigenmodes of the specified system
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode PRINVIT::Compute()
{
  PetscErrorCode ierr = 0;
  ierr = PetscLogEventBegin(EIG_Compute, 0, 0, 0, 0); CHKERRQ(ierr);
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Computing\n"); CHKERRQ(ierr);

  // Prep work
  ierr = PetscLogEventBegin(EIG_Initialize, 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = Compute_Init(); CHKERRQ(ierr);

  // Initialize search subspace with interpolated coarse eigenvectors
  ierr = Initialize_V(); CHKERRQ(ierr);

  // Orthonormalize search space vectors
  ierr = MatMult(B[0], V[0], TempVecs[0]); CHKERRQ(ierr);
  ierr = VecDot(V[0], TempVecs[0], TempScal.data()); CHKERRQ(ierr);
  ierr = VecScale(V[0], 1.0/sqrt(TempScal(0))); CHKERRQ(ierr);
  for (int ii = 1; ii < j; ii++) {
    ierr = Icgsm(V, B[0], V[ii], TempScal(0), ii); 
    if (ierr != 0)
      j = ii;
    ierr = VecScale(V[ii], 1.0/TempScal(0)); CHKERRQ(ierr);
  }

  // Construct initial search subspace
  MatrixXPS G = MatrixXPS::Zero(jmax,jmax);
  for (int ii = 0; ii < j; ii++) {
    ierr = MatMult(A[0], V[ii], TempVecs[ii]); CHKERRQ(ierr);
    ierr = VecMDot(TempVecs[ii], j, V, G.data() + jmax*ii); CHKERRQ(ierr);
  }

  // Construct eigensolver context for that subspace
  Eigen::SelfAdjointEigenSolver<MatrixXPS> eps_sub(jmax);
  PetscScalar theta = 0; // Approximation of lambda

  // Things needed in the computation loop
  Vec residual;
  VecDuplicate(Q[0][0], &residual);
  PetscReal rnorm = 0, rnorm_old = 0, Au_norm = 0, orth_norm = 0;
  MatrixXPS W; ArrayXPS S;
  PetscInt base_it = maxit; PetscScalar base_eps = eps;
  ierr = PetscLogEventEnd(EIG_Initialize, 0, 0, 0, 0); CHKERRQ(ierr);

  // The actual computation loop
  it = 0;
  while ((it++/(nev_conv+1)) < maxit) {
    eps_sub.compute(G.block(0,0,j,j));
    W = eps_sub.eigenvectors();
    S = eps_sub.eigenvalues();
    Sorteig(W, S);

    while (true) {
      ierr = PetscLogEventBegin(EIG_Prep, 0, 0, 0, 0); CHKERRQ(ierr);
      // Get new eigenvector approximation
      ierr = VecSet(Q[0][nev_conv], 0.0); CHKERRQ(ierr);
      ierr = VecMAXPY(Q[0][nev_conv], j, W.data(), V); CHKERRQ(ierr);
      theta = S(0);
      if (isnan(theta)) {
        SETERRQ(comm, PETSC_ERR_FP, "Approximate eigenvalue is not a number");
      }

      ierr = MatMult(A[0], Q[0][nev_conv], AQ[0][nev_conv]); CHKERRQ(ierr);
      ierr = MatMult(B[0], Q[0][nev_conv], BQ[0][nev_conv]); CHKERRQ(ierr);
      ierr = VecWAXPY(residual, -theta, BQ[0][nev_conv], AQ[0][nev_conv]); CHKERRQ(ierr);
      ierr = VecScale(residual, -1.0); CHKERRQ(ierr);

      ierr = Update_Preconditioner(residual, rnorm, Au_norm); CHKERRQ(ierr);

      ierr = PetscLogEventEnd(EIG_Prep, 0, 0, 0, 0); CHKERRQ(ierr);
      if (((rnorm/abs(theta) >= eps) && // Converged on residual norm
          ((std::abs(theta - lambda(nev_conv))/theta > 1e-14) || (it/(nev_conv+1) < 10))) // stagnation
           || (j <= 1)) {
        lambda(nev_conv) = theta;
        rnorm_old = rnorm;
        break;
      }

      ierr = PetscLogEventBegin(EIG_Convergence, 0, 0, 0, 0); CHKERRQ(ierr);
      // Convergence routine
      if (this->verbose >= 2)
        PetscFPrintf(comm, output, "Eigenvalue #%i converged with residual %1.4g after %i iterations\n", nev_conv+1, rnorm, it);
      tau_num = theta;
      lambda(nev_conv) = theta;
      for (int ii = 0; ii < j; ii++) {
        ierr = VecCopy(V[ii], TempVecs[ii]); CHKERRQ(ierr);
        ierr = VecSet(V[ii], 0.0); CHKERRQ(ierr);
      }

      // This line is to prevent an error when debugging but should be
      // unnecessary (and costly) if this is a production build
      #if defined(PETSC_USE_DEBUG)
      MPI_Allreduce(MPI_IN_PLACE, W.data(), j*j, MPI_DOUBLE, MPI_MAX, comm);
      #endif
      for (int ii = 1; ii < j; ii++) {
        ierr = VecMAXPY(V[ii-1], j, W.data()+ii*j, TempVecs); CHKERRQ(ierr);
      }

      for (int ii = 0; ii < j-1; ii++)
        S(ii) = S(ii+1);
      G.block(0, 0, j-1, j-1) = S.segment(0,j-1).matrix().asDiagonal();
      W = MatrixXPS::Identity(j-1, j-1);
      j--; nev_conv++;

      ierr = PetscLogEventEnd(EIG_Convergence, 0, 0, 0, 0); CHKERRQ(ierr);
      if (Done()) {
        // Cleanup
        ierr = VecDestroy(&residual); CHKERRQ(ierr);
        ierr = Compute_Clean(); CHKERRQ(ierr);
        if (this->verbose >= 1) {
          ierr = Print_Result(); CHKERRQ(ierr);
        }
        ierr = PetscLogEventEnd(EIG_Compute, 0, 0, 0, 0); CHKERRQ(ierr);
        return 0;
      }
    }

    if (this->verbose >= 2) {
      ierr = Print_Status(rnorm); CHKERRQ(ierr);
    }

    // Check for restart
    if (j == jmax) {
      j = jmin;
      for (int ii = 0; ii < jmax; ii++) {
        ierr = VecCopy(V[ii], TempVecs[ii]); CHKERRQ(ierr);
        ierr = VecSet(V[ii], 0.0); CHKERRQ(ierr);
      }
      for (int ii = 0; ii < j; ii++) {
        ierr = VecMAXPY(V[ii], jmax, W.data()+ii*jmax, TempVecs); CHKERRQ(ierr);
      }
      G.block(0, 0, j, j) = S.segment(0,j).matrix().asDiagonal();
    }

    // Get search space expansion vector
    ierr = PetscLogEventBegin(EIG_Update, 0, 0, 0, 0); CHKERRQ(ierr);
    ierr = Update_Search(V[j], residual, rnorm); CHKERRQ(ierr);
    ierr = PetscLogEventEnd(EIG_Update, 0, 0, 0, 0); CHKERRQ(ierr);

    ierr = PetscLogEventBegin(EIG_Expand, 0, 0, 0, 0); CHKERRQ(ierr);

    // Ensure orthogonality
    ierr = Mgsm(Q[0], BQ[0], V[j], nev_conv+1); CHKERRQ(ierr);
    ierr = Icgsm(V, B[0], V[j], orth_norm, j); CHKERRQ(ierr);
    ierr = Remove_NullSpace(this->B[0], V[j]); CHKERRQ(ierr);

    // Re-normalize
    ierr = MatMult(this->B[0], V[j], TempVecs[0]); CHKERRQ(ierr);
    ierr = VecDot(V[j], TempVecs[0], &orth_norm); CHKERRQ(ierr);
    ierr = VecScale(V[j], 1/sqrt(orth_norm)); CHKERRQ(ierr);

    // Update search space
    ierr = MatMult(A[0], V[j], TempVecs[0]); CHKERRQ(ierr);
    ierr = VecMDot(TempVecs[0], j+1, V, G.data()+j*jmax); CHKERRQ(ierr);
    G.block(j, 0, 1, j) = G.block(0, j, j, 1).transpose();

    // This is to prevent NaN breakdown if update was bad      
    if (isnan(G(j,j))) {
      ierr = VecSetRandom(V[j], NULL); CHKERRQ(ierr);

      // Ensure orthogonality
      ierr = Mgsm(Q[0], BQ[0], V[j], nev_conv+1); CHKERRQ(ierr);
      ierr = Icgsm(V, B[0], V[j], orth_norm, j); CHKERRQ(ierr);
      ierr = Remove_NullSpace(this->B[0], V[j]); CHKERRQ(ierr);

      // Re-normalize
      ierr = MatMult(this->B[0], V[j], TempVecs[0]); CHKERRQ(ierr);
      ierr = VecDot(V[j], TempVecs[0], &orth_norm); CHKERRQ(ierr);
      ierr = VecScale(V[j], 1/sqrt(orth_norm)); CHKERRQ(ierr);

      // Update search space
      ierr = MatMult(A[0], V[j], TempVecs[0]); CHKERRQ(ierr);
      ierr = VecMDot(TempVecs[0], j+1, V, G.data()+j*jmax); CHKERRQ(ierr);
      G.block(j, 0, 1, j) = G.block(0, j, j, 1).transpose();
    }
     
    ierr = PetscLogEventEnd(EIG_Expand, 0, 0, 0, 0); CHKERRQ(ierr);

    j++;
    if ((it/(nev_conv+1)) == maxit && eps/base_eps < 1000) {
      if (this->verbose >= 1)
        PetscFPrintf(comm, output, "Only %i converged eigenvalues in %i iterations, "
                     "increasing tolerance to %1.2g\n", nev_conv, maxit, eps*=10);
      maxit += base_it;
    }
  }
  // Cleanup
  if (this->verbose >= 1) {
    ierr = Print_Result(); CHKERRQ(ierr);
  }
  ierr = VecDestroy(&residual); CHKERRQ(ierr);
  this->nev_conv++;
  ierr = Compute_Clean(); CHKERRQ(ierr);
  this->nev_conv--;
  it--;
  eps = base_eps;

  ierr = PetscLogEventEnd(EIG_Compute, 0, 0, 0, 0); CHKERRQ(ierr);
  return 0;
}

/********************************************************************
 * Clean up after the compute phase
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode PRINVIT::Compute_Clean()
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
  // Fill in eigenvectors and add Q to the search space as memory size
  // allows in case we want to restart
  PetscInt start = std::min(j, jmax-nev_conv);
  PetscInt end = std::min(jmax, nev_conv);
  for (PetscInt ii = 0; ii < nev_conv; ii++) {
    ierr = VecCopy(Q[0][order(ii)], phi[ii]); CHKERRQ(ierr);
    if (ii < end) {
      ierr = VecCopy(Q[0][order(ii)], V[ii+start]); CHKERRQ(ierr);
    }
    j = std::min(jmax, j+nev_conv-1);
  }

  // Destroy eigenvectors storage space at each level
  ierr = Destroy_Q(); CHKERRQ(ierr);

  return 0;
}

/********************************************************************
 * Remove a matrix nullspace from a vector
 * 
 * @param A: Matrix that has a nullspace
 * @param x: Vector to remove nullspace from
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode PRINVIT::Remove_NullSpace(Mat A, Vec x)
{
  PetscErrorCode ierr = 0;

  MatNullSpace nullsp;
  ierr = MatGetNullSpace(A, &nullsp); CHKERRQ(ierr);
  if (nullsp) {
    ierr = MatNullSpaceRemove(nullsp, x); CHKERRQ(ierr);
  }

  return ierr;
}