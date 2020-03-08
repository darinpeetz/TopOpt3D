#include <petscksp.h>
#include "EigenInverse.h"
#include <fstream>

extern "C"{
  void dsytrd_(char *UPLO, int *N, double *A, int *LDA, double *D, double *E,
              double *TAU, double *WORK, int *LWORK, int *INFO);
  void dorgtr_(char *UPLO, int *N, double *A, int *LDA,
              double *TAU, double *WORK, int *LWORK, int *INFO);
  void dsteqr_(char *COMPZ, int *N, double *D, double *E, double *Z, int *LDZ,
               double *WORK, int *INFO);
  void dgemv_(char *TRANS, int *M, int *N, double *ALPHA, double *A, int *LDA,
              const double *X, int *INCX, double *BETA, double *Y, int *INCY);
}

/*****************************************************/
/**        Set a shell PC to use eigenvalue         **/
/**        decomposition on coarse operator         **/
/*****************************************************/
PetscErrorCode CreateEigenShell(PC pc)
{
  PetscErrorCode ierr = 0;

  EigenShellPC *eigenPC = new EigenShellPC;
  eigenPC->n = 0;
  eigenPC->nLoc = 0;
  eigenPC->lam = NULL;
  eigenPC->Q = NULL;
  eigenPC->SetUp = false;

  ierr = PCShellSetContext(pc, eigenPC); CHKERRQ(ierr);

  return ierr;
}

/*****************************************************/
/** Set up the shell pc (perform eigendecomposition)**/
/*****************************************************/
PetscErrorCode EigenShellSetUp(PC pc)
{
  PetscErrorCode ierr = 0;

  EigenShellPC *eigenPC;
  ierr = PCShellGetContext(pc, (void**)&eigenPC); CHKERRQ(ierr);

  Mat A;
  ierr = PCGetOperators(pc, &A, NULL); CHKERRQ(ierr);
  ierr = MatGetSize(A, &eigenPC->n, NULL); CHKERRQ(ierr);
  ierr = MatGetLocalSize(A, &eigenPC->nLoc, NULL); CHKERRQ(ierr);

  if (eigenPC->nLoc == 0)
    return 0;
  else if (eigenPC->nLoc != eigenPC->n)
    SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP,
            "Coarsest matrix is split across multiple processes");
  
  // Allocate EigenShellPC data structures
  PetscInt n = eigenPC->n;
  if (!eigenPC->SetUp) {
    eigenPC->Q = new PetscScalar[n*n];
    eigenPC->lam = new PetscScalar[n];
  }

  // Get A in a dense format
  PetscInt ncols;
  const PetscInt *cols;
  const PetscScalar *vals;
  std::fill(eigenPC->Q, eigenPC->Q + n*n, 0);
  for (PetscInt i = 0; i < n; i++) {
    ierr = MatGetRow(A, i, &ncols, &cols, &vals);
    for (PetscInt j = 0; j < ncols; j++) {
      eigenPC->Q[n*cols[j] + i] = vals[j];
    }
    ierr = MatRestoreRow(A, i, &ncols, &cols, &vals); CHKERRQ(ierr);
  }

  // Call lapack routines to calculate eigenvalues and eigenvectors of system
  PetscScalar *e = new PetscScalar[n-1];
  PetscScalar *tau = new PetscScalar[n-1], *work = new PetscScalar[n*n];
  PetscBLASInt lwork = n*n, info = 0;
  char c = 'L';
  dsytrd_(&c, &n, eigenPC->Q, &n, eigenPC->lam, e, tau, work, &lwork, &info);
  dorgtr_(&c, &n, eigenPC->Q, &n, tau, work, &lwork, &info);
  c = 'V';
  dsteqr_(&c, &n, eigenPC->lam, e, eigenPC->Q, &n, work, &info);

  // Invert eigenvalues. Small eigenvalues (rigid modes) are set to zero
  PetscInt ind = 0;
  while (true) {
    if (std::abs(eigenPC->lam[ind] / eigenPC->lam[ind+1]) < 1e-3) {
      eigenPC->lam[ind++] = 0;
      break;
    }
    else {
      eigenPC->lam[ind++] = 0;
    }
    if (ind == n-1) {
      eigenPC->lam[ind] = 0;
      ierr = PetscPrintf(PetscObjectComm((PetscObject)pc), "Warning, all eigenvalues are "
                        "of similar magnitude in coarse operator.\nRoutines will assume "
                        "that the coarse operator has no non-rigid modes\n"); CHKERRQ(ierr);
      ind++;
      break;
    }
  }
  while (ind < n) {
    eigenPC->lam[ind] = 1/eigenPC->lam[ind];
    ind++;
  }

  eigenPC->SetUp = true;
  delete[] e; delete[] tau; delete[] work;

  return ierr;
}

/*****************************************************/
/**   Apply the inverse of the eigendecomposition   **/
/*****************************************************/
PetscErrorCode EigenShellApply(PC pc, Vec x, Vec y)
{
  PetscErrorCode ierr = 0;

  EigenShellPC *eigenPC;
  ierr = PCShellGetContext(pc, (void**)&eigenPC); CHKERRQ(ierr);
  if (eigenPC->nLoc == 0)
    return 0;

  const PetscScalar *p_x;
  PetscScalar *p_y, *p_z;
  ierr = VecGetArrayRead(x, &p_x); CHKERRQ(ierr);
  ierr = VecGetArray(y, &p_y); CHKERRQ(ierr);
  p_z = new PetscScalar[eigenPC->n];

  PetscBLASInt inc = 1;
  PetscScalar one = 1, zero = 0;
  char c = 'T';
  dgemv_(&c, &eigenPC->n, &eigenPC->n, &one, eigenPC->Q,
         &eigenPC->n, p_x, &inc, &zero, p_z, &inc);
  for (PetscInt i = 0; i < eigenPC->n; i++)
    p_z[i] = p_z[i] * eigenPC->lam[i];
  c = 'N';
  dgemv_(&c, &eigenPC->n, &eigenPC->n, &one, eigenPC->Q,
         &eigenPC->n, p_z, &inc, &zero, p_y, &inc);
  
  delete[] p_z;
  ierr = VecRestoreArray(y, &p_y); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &p_x); CHKERRQ(ierr);
  
  return ierr;
}

/*****************************************************/
/**   Apply the inverse of the eigendecomposition   **/
/*****************************************************/
PetscErrorCode EigenShellDestroy(PC pc)
{
  PetscErrorCode ierr = 0;

  EigenShellPC *eigenPC;
  ierr = PCShellGetContext(pc, (void**)&eigenPC); CHKERRQ(ierr);
  if (eigenPC->SetUp)
  {
    delete[] eigenPC->Q;
    delete[] eigenPC->lam;
  }
  delete eigenPC;

  return ierr;
}