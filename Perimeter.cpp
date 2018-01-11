#include <iostream>
#include <cmath>
#include "Functions.h"
#include "TopOpt.h"
#include "EigLab.h"
#include <fstream>

using namespace std;

namespace Functions
{
  int Perimeter( TopOpt *topOpt, double &obj, double *grad )
  {
    PetscErrorCode ierr = 0;
    obj = 0;
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
          obj += topOpt->edgeSize(i) * abs(Weight);
          p_dP[topOpt->edgeElem(i, 0)] += topOpt->edgeSize(i) * copysign(1,Weight);
          if ( topOpt->edgeElem(i, 1) < maxVind )
            p_dP[topOpt->edgeElem(i, 1)] -= topOpt->edgeSize(i) * copysign(1,Weight);
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &obj, 1, MPI_DOUBLE, MPI_SUM, topOpt->comm);
    obj /= -topOpt->nElem*topOpt->PerimNormFactor;

    // Return if sensitivities aren't needed
    if (grad == NULL)
      return 0;

    ierr = VecRestoreArrayRead( topOpt->V, &p_V ); CHKERRQ(ierr);
    ierr = VecRestoreArray( dPdy, &p_dP ); CHKERRQ(ierr);

    ierr = VecGhostUpdateBegin(dPdy, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(dPdy, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
    ierr = VecScale(dPdy, -1/(topOpt->nElem*topOpt->PerimNormFactor)); CHKERRQ(ierr);

    // dPdrhof*drhofdrho
    Vec PETSc_grad;
    ierr = VecDuplicate( topOpt->x, &PETSc_grad ); CHKERRQ(ierr);
    ierr = VecPlaceArray( PETSc_grad, grad ); CHKERRQ(ierr);
    ierr = MatMultTranspose( topOpt->P, dPdy, PETSc_grad ); CHKERRQ(ierr);
    ierr = VecResetArray( PETSc_grad ); CHKERRQ(ierr);
    ierr = VecDestroy( &dPdy ); CHKERRQ(ierr);
    ierr = VecDestroy( &PETSc_grad ); CHKERRQ(ierr);

    return 0;
  }
}
