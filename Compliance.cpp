#include "Functions.h"
#include "TopOpt.h"

using namespace std;

PetscErrorCode Compliance::Function( TopOpt *topOpt )
{
  PetscErrorCode ierr = 0;
  // Objective
  ierr = VecTDot(topOpt->U, topOpt->F, values.data()); CHKERRQ(ierr);

  // Return if sensitivities aren't needed
  if (calc_gradient == PETSC_FALSE)
    return 0;

  // Sensitivities
  Vec dCdy;
  ierr = VecDuplicate( topOpt->dEdy, &dCdy ); CHKERRQ(ierr);
  ierr = VecPlaceArray( dCdy, gradients.data() ); CHKERRQ(ierr);

  const PetscScalar *p_U, *p_dEdy;
  VecGetArrayRead( topOpt->U, &p_U ); CHKERRQ(ierr);
  VecGetArrayRead( topOpt->dEdy, &p_dEdy ); CHKERRQ(ierr);
  // dCdrhof
  short DN = topOpt->numDims;
  short NE = pow(2, topOpt->numDims);
  ArrayXPI eDof( DN*NE );
  for (long el = 0; el < topOpt->nLocElem; el++)
  {
    gradients(el,0) = 0;
    for (int j = 0; j < NE; j++){
    for (int i = 0; i < DN; i++){
      eDof(DN*j+i) = DN * topOpt->element(el, j) + i; } }

    for (int ii = 0; ii < DN*NE; ii++){
    for (int jj = 0; jj < DN*NE; jj++){
      if (topOpt->regular)
        gradients(el,0) += p_U[ eDof(ii) ] * topOpt->ke[0](ii,jj)
        * p_U[ eDof(jj) ];
      else
        gradients(el,0) += p_U[ eDof(ii) ] * topOpt->ke[el](ii,jj)
        * p_U[ eDof(jj) ];
    }
    }
    gradients(el,0) *= -p_dEdy[el];
  }
  ierr = VecRestoreArrayRead( topOpt->U, &p_U ); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead( topOpt->dEdy, &p_dEdy ); CHKERRQ(ierr);

  // dCdrhof*drhofdrho
  ierr = Chain_Filter( topOpt->P, dCdy); CHKERRQ(ierr);

  ierr = VecResetArray( dCdy ); CHKERRQ(ierr);
  ierr = VecDestroy( &dCdy ); CHKERRQ(ierr);

  return 0;
}
