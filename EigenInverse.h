#include <petscksp.h>

typedef struct {
  PetscInt n, nLoc;
  PetscScalar *lam, *Q;
  bool SetUp;
} EigenShellPC;

// Routines to create, use, and destroy the shell preconditioner
PetscErrorCode CreateEigenShell ( PC pc );
PetscErrorCode EigenShellSetUp ( PC pc );
PetscErrorCode EigenShellApply ( PC pc, Vec x, Vec y );
PetscErrorCode EigenShellDestroy ( PC pc );