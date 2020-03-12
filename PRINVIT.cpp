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
  Qsize = nev_req;
  jmin = -1; jmax = -1;
  PetscOptionsGetInt(NULL, NULL, "-PRINVIT_Verbose", &verbose, NULL);
}

/********************************************************************
 * Main constructor
 * 
 *******************************************************************/
PRINVIT::~PRINVIT()
{
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
  PetscInt j = 0;
  ierr = Initialize_V(j); CHKERRQ(ierr);
  nev_conv = 0;

  // Orthonormalize search space vectors
  ierr = MatMult(B[0], V[0], TempVecs[0]); CHKERRQ(ierr);
  ierr = VecDot(V[0], TempVecs[0], TempScal.data()); CHKERRQ(ierr);
  ierr = VecScale(V[0], 1.0/sqrt(TempScal(0))); CHKERRQ(ierr);
  for (int ii = 1; ii < j; ii++) {
    ierr = Icgsm(V, B[0], V[ii], TempScal(0), ii); CHKERRQ(ierr);
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
      ierr = MatMult(A[0], Q[0][nev_conv], AQ[0][nev_conv]); CHKERRQ(ierr);
      ierr = MatMult(B[0], Q[0][nev_conv], BQ[0][nev_conv]); CHKERRQ(ierr);
      ierr = VecWAXPY(residual, -theta, BQ[0][nev_conv], AQ[0][nev_conv]); CHKERRQ(ierr);
      ierr = VecScale(residual, -1.0); CHKERRQ(ierr);

      ierr = Update_Preconditioner(residual, rnorm, Au_norm); CHKERRQ(ierr);

      if (this->verbose >= 2) {
        ierr = Print_Status(rnorm); CHKERRQ(ierr);
      }
      if (isnan(theta)) {
        SETERRQ(comm, PETSC_ERR_FP, "Approximate eigenvalue is not a number");
      }

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

  ierr = PetscLogEventEnd(EIG_Compute, 0, 0, 0, 0); CHKERRQ(ierr);
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