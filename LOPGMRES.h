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
  LOPGMRES(MPI_Comm comm=MPI_COMM_WORLD);
  // Destructor
  ~LOPGMRES();
  // How much information to print
  PetscErrorCode Set_Verbose(PetscInt verbose);
  // Set the preconditioner instance
  PetscErrorCode Set_PC(PC pc) {this->pc = pc; 
        PetscErrorCode ierr = PetscObjectReference((PetscObject)pc);
        CHKERRQ(ierr); return ierr;}

private:
  /// Class variables
  // Preconditioner instance
  PC pc;
  /// Private methods
  // Prepare solver for compute step
  PetscErrorCode Compute_Init();
  // Multigrid solver
  PetscErrorCode Update_Preconditioner(Vec residual, PetscScalar &rnorm,
                                       PetscScalar &Au_norm);
  // Update the search space with the correction equation
  PetscErrorCode Update_Search(Vec x, Vec residual, PetscReal rnorm);
  // Output information
  PetscErrorCode Print_Result() {
    return PetscFPrintf(comm, output, "LOPGMRES found %i of a requested %i "
                        "eigenvalues after %i iterations \n\n", nev_conv, nev_req, it);
  }

};

#endif // LOPGMRES_H_INCLUDED
