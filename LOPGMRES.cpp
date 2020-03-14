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
  if (this->pc != NULL) {
    PetscErrorCode ierr = PCDestroy(&this->pc); CHKERRV(ierr);
  }
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
  ierr = Setup_Q(); CHKERRQ(ierr);
  if (nev_conv > 0) {
    ierr = VecDestroyVecs(nev_conv, &phi); CHKERRQ(ierr);
  }
  nev_conv = 0;

  ierr = PetscLogEventEnd(EIG_Comp_Init, 0, 0, 0, 0); CHKERRQ(ierr);

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
