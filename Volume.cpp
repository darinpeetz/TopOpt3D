#include <iostream>
#include "Functions.h"
#include "TopOpt.h"

using namespace std;
typedef unsigned int uint;

namespace Functions
{
  int Volume( TopOpt *topOpt, double &obj, double *grad )
  {
    PetscErrorCode ierr = 0;
    /// Objective
    ierr = VecSum( topOpt->V, &obj ); CHKERRQ(ierr);
    obj /= topOpt->nElem;

    // Return if sensitivities aren't needed
    if (grad == NULL)
      return 0;

    /// Sensitivities
    // dVdrhof
    Vec dVdy;
    ierr = VecDuplicate( topOpt->dVdy, &dVdy ); CHKERRQ(ierr);
    ierr = VecCopy( topOpt->dVdy, dVdy ); CHKERRQ(ierr);
    ierr = VecScale( dVdy, 1.0/topOpt->nElem ); CHKERRQ(ierr);
    // dVdrhof*drhofdrho
    Vec PETSc_grad;
    ierr = VecCreate( topOpt->comm, &PETSc_grad ); CHKERRQ(ierr);
    ierr = VecSetType( PETSc_grad, VECMPI ); CHKERRQ(ierr);
    ierr = VecSetSizes( PETSc_grad, topOpt->nLocElem, topOpt->nElem ); CHKERRQ(ierr);
    ierr = VecPlaceArray( PETSc_grad, grad ); CHKERRQ(ierr);
    ierr = MatMultTranspose( topOpt->P, dVdy, PETSc_grad ); CHKERRQ(ierr);
    ierr = VecResetArray( PETSc_grad ); CHKERRQ(ierr);
    ierr = VecDestroy( &dVdy ); CHKERRQ(ierr);
    ierr = VecDestroy( &PETSc_grad ); CHKERRQ(ierr);

    return 0;
  }
}
