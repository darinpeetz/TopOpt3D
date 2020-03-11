#include "Functions.h"
#include "TopOpt.h"

using namespace std;


/********************************************************************
 * Compute compliance and its sensitivity
 * 
 * @param topOpt: The topology optimization object
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode Compliance::Function(TopOpt *topOpt)
{
  PetscErrorCode ierr = 0;
  // Objective
  ierr = VecTDot(topOpt->U, topOpt->F, values.data()); CHKERRQ(ierr);

  // Return if sensitivities aren't needed
  if (calc_gradient == PETSC_FALSE)
    return 0;

  // Sensitivities
  Vec dCdy;
  ierr = VecDuplicate(topOpt->dEdz, &dCdy); CHKERRQ(ierr);
  ierr = VecPlaceArray(dCdy, gradients.data()); CHKERRQ(ierr);

  const PetscScalar *p_U, *p_dEdz;
  ierr = VecGetArrayRead(topOpt->U, &p_U); CHKERRQ(ierr);
  ierr = VecGetArrayRead(topOpt->dEdz, &p_dEdz); CHKERRQ(ierr);
  // dCdrhof
  short DN = topOpt->numDims;
  short NE = pow(2, topOpt->numDims);
  VectorXPS U_loc(DN*NE);
  for (long el = 0; el < topOpt->nLocElem; el++) {
    for (int j = 0; j < NE; j++) {
      for (int i = 0; i < DN; i++) {
        U_loc(DN*j+i) = p_U[DN * topOpt->element(el, j) + i]; } }

    if (topOpt->regular)
      gradients(el,0) = -p_dEdz[el] * U_loc.dot(topOpt->ke[0]  * U_loc);
    else
      gradients(el,0) = -p_dEdz[el] * U_loc.dot(topOpt->ke[el] * U_loc);
  }
  ierr = VecRestoreArrayRead(topOpt->U, &p_U); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(topOpt->dEdz, &p_dEdz); CHKERRQ(ierr);

  // dCdrhof*drhofdrho
  ierr = topOpt->Chain_Filter(NULL, dCdy); CHKERRQ(ierr);

  ierr = VecResetArray(dCdy); CHKERRQ(ierr);
  ierr = VecDestroy(&dCdy); CHKERRQ(ierr);

  return 0;
}