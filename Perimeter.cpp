#include <cmath>
#include "Functions.h"
#include "TopOpt.h"
#include "EigLab.h"

using namespace std;

PetscErrorCode Perimeter::Function( TopOpt *topOpt )
{
  PetscErrorCode ierr = 0;
  values.setZero();
  double Weight;
  const double *p_V;
  ierr = VecGetArrayRead( topOpt->V, &p_V ); CHKERRQ(ierr);

  // Sensitivities
  Vec dPdy; PetscScalar *p_dP;
  ierr = VecDuplicate( topOpt->V, &dPdy ); CHKERRQ(ierr);
  ierr = VecZeroEntries( dPdy ); CHKERRQ(ierr);
  ierr = VecGetArray( dPdy, &p_dP ); CHKERRQ(ierr);

  // The element number of an exterior edge
  PetscInt maxVind = topOpt->element.rows();
  for (long i = 0; i < topOpt->edgeElem.rows(); i++)
  {
    // Get the difference in weight across the edge
    if ( topOpt->edgeElem(i, 1) == maxVind )
      Weight = p_V[ topOpt->edgeElem(i , 0) ];
    else
      Weight = p_V[ topOpt->edgeElem(i, 0) ] - p_V[ topOpt->edgeElem(i, 1) ];

    // Tabulate the information into objective and sensitivities
    if (abs(Weight) > 1e-14)
    {
      values(0) += topOpt->edgeSize(i) * abs(Weight);
      p_dP[topOpt->edgeElem(i, 0)] += topOpt->edgeSize(i) * copysign(1,Weight);
      if ( topOpt->edgeElem(i, 1) < maxVind )
      p_dP[topOpt->edgeElem(i, 1)] -= topOpt->edgeSize(i) * copysign(1,Weight);
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, values.data(), 1, MPI_DOUBLE, MPI_SUM, topOpt->comm);

  values(0) /= -topOpt->nElem*topOpt->PerimNormFactor;

  // Return if sensitivities aren't needed
  if (calc_gradient == PETSC_FALSE)
    return 0;

  ierr = VecRestoreArrayRead( topOpt->V, &p_V ); CHKERRQ(ierr);
  ierr = VecRestoreArray( dPdy, &p_dP ); CHKERRQ(ierr);

  ierr = VecGhostUpdateBegin(dPdy, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(dPdy, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScale(dPdy, -1/(topOpt->nElem*topOpt->PerimNormFactor)); CHKERRQ(ierr);

  // dPdrhof*drhofdrho
  ierr = Chain_Filter( topOpt->P, dPdy ); CHKERRQ(ierr);

  ierr = VecGetArray( dPdy, &p_dP ); CHKERRQ(ierr);
  copy( p_dP, p_dP+topOpt->nLocElem, gradients.data() );
  ierr = VecRestoreArray( dPdy, &p_dP ); CHKERRQ(ierr);
  ierr = VecDestroy( &dPdy ); CHKERRQ(ierr);

  return 0;
}
