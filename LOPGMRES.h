#ifndef LOPGMRES_H_INCLUDED
#define LOPGMRES_H_INCLUDED

#include <petscksp.h>
#include <vector>
#include <Eigen/Eigen>
#include <algorithm>
#include "PRINVIT.h"

/// The master structure containing all information to be carried between iterations
class LOPGMRES : public PRINVIT
{
  /// Class variables
public:
  /// Class methods
  // Constructors
  LOPGMRES() {LOPGMRES(MPI_COMM_WORLD);}
  LOPGMRES(MPI_Comm comm);
  // Destructor
  ~LOPGMRES();
  // How much information to print
  PetscErrorCode Set_Verbose(PetscInt verbose);

private:
  /// Private methods
  // Prepare solver for compute step
  PetscErrorCode Compute_Init();
  PetscErrorCode Create_Hierarchy();
  PetscErrorCode Initialize_V(PetscInt &j);
  // Cleanup after compute step
  PetscErrorCode Compute_Clean();
  // Multigrid solver
  PetscErrorCode Update_Preconditioner(Vec residual, PetscScalar &rnorm, PetscScalar &Au_norm);
  PetscErrorCode MGSetup(Vec f, PetscReal fnorm);
  // Apply Operator at a given level
  PetscErrorCode ApplyOP(Vec x, Vec y, PetscInt level);
  // Coarse solver in multigrid
  PetscErrorCode Coarse_Solve();

};

#endif // LOPGMRES_H_INCLUDED
