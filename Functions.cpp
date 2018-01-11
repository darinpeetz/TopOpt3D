#include "TopOpt.h"
#include "Functions.h"

using namespace std;

namespace Functions
{
int FunctionCall (TopOpt *topOpt, double &f, Eigen::VectorXd &dfdx,
                   Eigen::VectorXd &g, Eigen::MatrixXd &dgdx)
{
  PetscErrorCode ierr = 0;

  /// Evaluates objective function and constraint functions
  int num_constraints = (int)(topOpt->Comp==2) + (int)(topOpt->Perim==2) + (int)(topOpt->Vol==2);
  num_constraints += topOpt->Stab_nev - topOpt->Stab_optnev;
  num_constraints += topOpt->Dyn_nev - topOpt->Dyn_optnev;
  int active_constraint = 0;

  f = 0;
  dfdx.setZero(topOpt->nLocElem);
  g.setZero(max(num_constraints,1));
  dgdx.setZero(topOpt->nLocElem, max(num_constraints,1));
  Eigen::VectorXd temp(max(topOpt->Stab_nev,topOpt->Dyn_nev)+6);
  Eigen::MatrixXd dtemp = Eigen::MatrixXd::Zero(topOpt->nLocElem,
      max(max(topOpt->Stab_nev,topOpt->Dyn_nev),(short)1));

  // Compliance
  if (topOpt->Comp == 1)
  {
    ierr = Functions::Compliance( topOpt, temp(0), dtemp.data() ); CHKERRQ(ierr);
    f += topOpt->Comp_val[0]*(temp(0)-topOpt->Comp_min)/(topOpt->Comp_max-topOpt->Comp_min);
    dfdx += topOpt->Comp_val[0]*dtemp.col(0)/(topOpt->Comp_max-topOpt->Comp_min);
  }
  else if (topOpt->Comp == 2)
  {
    ierr = Functions::Compliance( topOpt, g(active_constraint),
      dgdx.data()+topOpt->nLocElem*active_constraint ); CHKERRQ(ierr);
    g(active_constraint) -= topOpt->Comp_val[0];
    g(active_constraint) /= topOpt->Comp_max-topOpt->Comp_min;
    dgdx.col(active_constraint) /= (topOpt->Comp_max-topOpt->Comp_min);
    active_constraint += 1;
  }

  // Perimeter
  temp *= 0; dtemp *= 0;
  if (topOpt->Perim == 1)
  {
    ierr = Functions::Perimeter( topOpt, temp(0), dtemp.data() ); CHKERRQ(ierr);
    f += topOpt->Perim_val[0]*(temp(0)-topOpt->Perim_min)/(topOpt->Perim_max-topOpt->Perim_min);
    dfdx += topOpt->Perim_val[0]*dtemp.col(0)/(topOpt->Perim_max-topOpt->Perim_min);
  }
  else if (topOpt->Perim == 2)
  {
    ierr = Functions::Perimeter( topOpt, g(active_constraint),
      dgdx.data()+topOpt->nLocElem*active_constraint ); CHKERRQ(ierr);
    g(active_constraint) -= topOpt->Perim_val[0];
    g(active_constraint) /= topOpt->Perim_max-topOpt->Perim_min;
    dgdx.col(active_constraint) /= (topOpt->Perim_max-topOpt->Perim_min);
    active_constraint += 1;
  }

  // Volume
  temp *= 0; dtemp *= 0;
  if (topOpt->Vol == 1)
  {
    ierr = Functions::Volume( topOpt, temp(0), dtemp.data() ); CHKERRQ(ierr);
    f += topOpt->Vol_val[0]*(temp(0)-topOpt->Vol_min)/(topOpt->Vol_max-topOpt->Vol_min);
    dfdx += topOpt->Vol_val[0]*dtemp.col(0)/(topOpt->Vol_max-topOpt->Vol_min);
  }
  else if (topOpt->Vol == 2)
  {
    ierr = Functions::Volume( topOpt, g(active_constraint),
      dgdx.data()+topOpt->nLocElem*active_constraint ); CHKERRQ(ierr);
    g(active_constraint) -= topOpt->Vol_val[0];
    g(active_constraint) /= topOpt->Vol_max-topOpt->Vol_min;
    dgdx.col(active_constraint) /= (topOpt->Vol_max-topOpt->Vol_min);
    active_constraint += 1;
  }

  // Stability
  temp *= 0; dtemp *= 0;
  if (topOpt->Stab == 1)
  {
    PetscInt nevals = topOpt->Stab_nev;
    ierr = Functions::Buckling( topOpt, temp.data(), dtemp.data(), nevals ); CHKERRQ(ierr);
    for (short i = 0; i < min(nevals,(PetscInt)topOpt->Stab_optnev); i++)
    {
      f += topOpt->Stab_val[i] * (temp(i) - topOpt->Stab_min) /
        (topOpt->Stab_max - topOpt->Stab_min);
      dfdx += topOpt->Stab_val[i] * dtemp.col(i) /
        (topOpt->Stab_max - topOpt->Stab_min);
    }
    for (short i = topOpt->Stab_optnev, j = 0; i < nevals; i++, j++)
    {
      g(active_constraint+j) = (temp(i) - 0.99*temp(i-1))/(topOpt->Stab_max-topOpt->Stab_min);
      dgdx.col(active_constraint+j) = dtemp.col(i)/(topOpt->Stab_max-topOpt->Stab_min);
    }
    active_constraint += topOpt->Stab_nev - topOpt->Stab_optnev;
  }
  else if (topOpt->Stab == 2)
  {
    PetscInt nevals = topOpt->Stab_nev;
    ierr = Functions::Buckling( topOpt, temp.data(), dtemp.data(), nevals ); CHKERRQ(ierr);
    for (short i = 0; i < min(nevals,(PetscInt)topOpt->Stab_optnev); i++)
    {
      g(active_constraint+i) += (temp(i) - topOpt->Stab_val[i]) /
        (topOpt->Stab_max - topOpt->Stab_min);
      dgdx.col(active_constraint+i) += dtemp.col(0)/(topOpt->Stab_max-topOpt->Stab_min);
    }
    for (short i = topOpt->Stab_optnev; i < nevals; i++)
    {
      g(active_constraint+i) = (temp(i) - 0.99*temp(i-1))/(topOpt->Stab_max-topOpt->Stab_min);
      dgdx.col(active_constraint+i) = dtemp.col(i)/(topOpt->Stab_max-topOpt->Stab_min);
    }
    active_constraint += topOpt->Stab_nev;
  }

  // Frequency
  temp *= 0; dtemp *= 0;
  if (topOpt->Dyn == 1)
  {
    PetscInt nevals = topOpt->Dyn_nev;
    ierr = Functions::Dynamic( topOpt, temp.data(), dtemp.data(), nevals ); CHKERRQ(ierr);
    for (short i = 0; i < min(nevals,(PetscInt)topOpt->Dyn_optnev); i++)
    {
      f += topOpt->Dyn_val[i] * (temp(i) - topOpt->Dyn_min) /
        (topOpt->Dyn_max - topOpt->Dyn_min);
      dfdx += topOpt->Dyn_val[i] * dtemp.col(i) /
        (topOpt->Dyn_max - topOpt->Dyn_min);
    }
    for (short i = topOpt->Dyn_optnev, j = 0; i < nevals; i++, j++)
    {
      g(active_constraint+j) = (temp(i) - 0.99*temp(i-1))/(topOpt->Dyn_max-topOpt->Dyn_min);
      dgdx.col(active_constraint+j) = dtemp.col(i)/(topOpt->Dyn_max-topOpt->Dyn_min);
    }
    active_constraint += topOpt->Dyn_nev - topOpt->Dyn_optnev;
  }
  else if (topOpt->Dyn == 2)
  {
    PetscInt nevals = topOpt->Dyn_nev;
    ierr = Functions::Dynamic( topOpt, temp.data(), dtemp.data(), nevals ); CHKERRQ(ierr);
    for (short i = 0; i < min(nevals,(PetscInt)topOpt->Dyn_optnev); i++)
    {
      g(active_constraint+i) += (temp(i) - topOpt->Dyn_val[i]) /
        (topOpt->Dyn_max - topOpt->Dyn_min);
      dgdx.col(active_constraint+i) += dtemp.col(0)/(topOpt->Dyn_max-topOpt->Dyn_min);
    }
    for (short i = topOpt->Dyn_optnev; i < nevals; i++)
    {
      g(active_constraint+i) = (temp(i) - 0.99*temp(i-1))/(topOpt->Dyn_max-topOpt->Dyn_min);
      dgdx.col(active_constraint+i) = dtemp.col(i)/(topOpt->Dyn_max-topOpt->Dyn_min);
    }
  }

  return 0;
}
}
