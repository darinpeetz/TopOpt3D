#include <iostream>
#include "Functions.h"
#include "TopOpt.h"
#include "fstream"

using namespace std;

namespace Functions
{
  int Compliance( TopOpt *topOpt, double &obj, double *grad )
  {
    PetscErrorCode ierr;
    // Objective
    ierr = VecTDot(topOpt->U, topOpt->F, &obj); CHKERRQ(ierr);

    // Return if sensitivities aren't needed
    if (grad == NULL)
      return 0;

    // Sensitivities
    Vec dCdy; PetscScalar *p_dC;
    ierr = VecCreate( topOpt->comm, &dCdy ); CHKERRQ(ierr);
    ierr = VecSetType( dCdy, VECMPI ); CHKERRQ(ierr);
    ierr = VecSetSizes( dCdy, topOpt->nLocElem, topOpt->nElem ); CHKERRQ(ierr);
    ierr = VecGetArray( dCdy, &p_dC ); CHKERRQ(ierr);

    const PetscScalar *p_U, *p_dEdy;
    VecGetArrayRead( topOpt->U, &p_U ); CHKERRQ(ierr);
    VecGetArrayRead( topOpt->dEdy, &p_dEdy ); CHKERRQ(ierr);
    // dCdrhof
    short DN = topOpt->numDims;
    short NE = pow(2, topOpt->numDims);
    ArrayXPI eDof( DN*NE );
    for (long el = 0; el < topOpt->nLocElem; el++)
    {
      p_dC[el] = 0;
      for (int j = 0; j < NE; j++){
        for (int i = 0; i < DN; i++){
          eDof(DN*j+i) = DN * topOpt->element(el, j) + i; } }

      for (int ii = 0; ii < DN*NE; ii++){
        for (int jj = 0; jj < DN*NE; jj++){
          if (topOpt->regular)
            p_dC[el] += p_U[ eDof(ii) ] * topOpt->ke[0](ii,jj)
              * p_U[ eDof(jj) ];
          else
            p_dC[el] += p_U[ eDof(ii) ] * topOpt->ke[el](ii,jj)
              * p_U[ eDof(jj) ];
        }
      }
      p_dC[el] *= -p_dEdy[el];
    }
    ierr = VecRestoreArray( dCdy, &p_dC ); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead( topOpt->U, &p_U ); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead( topOpt->dEdy, &p_dEdy ); CHKERRQ(ierr);

    // dCdrhof*drhofdrho
    Vec PETSc_grad;
    ierr = VecCreate( topOpt->comm, &PETSc_grad ); CHKERRQ(ierr);
    ierr = VecSetType( PETSc_grad, VECMPI ); CHKERRQ(ierr);
    ierr = VecSetSizes( PETSc_grad, topOpt->nLocElem, topOpt->nElem ); CHKERRQ(ierr);
    ierr = VecPlaceArray( PETSc_grad, grad ); CHKERRQ(ierr);
    ierr = MatMultTranspose( topOpt->P, dCdy, PETSc_grad ); CHKERRQ(ierr);
    ierr = VecResetArray( PETSc_grad ); CHKERRQ(ierr);
    ierr = VecDestroy( &dCdy ); CHKERRQ(ierr);
    ierr = VecDestroy( &PETSc_grad ); CHKERRQ(ierr);

    return 0;
  }
}
