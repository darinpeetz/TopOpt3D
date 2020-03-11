#ifndef PRINVIT_H_INCLUDED
#define PRINVIT_H_INCLUDED

#include <slepceps.h>
#include <vector>
#include <Eigen/Eigen>
#include <algorithm>
#include "EigenPeetz.h"

/// The master structure containing all information to be carried between iterations
class PRINVIT : public EigenPeetz
{
  /// Class variables
public:

  /// Class methods
  // Constructors
  PRINVIT();
  // Destructor
  ~PRINVIT();
  // How much information to print
  virtual PetscErrorCode Set_Verbose(PetscInt verbose);
  // Solver
  PetscErrorCode Compute();
  // Search space size
  void Set_jmin(PetscInt jmin) {this->jmin = jmin;}
  void Set_jmax(PetscInt jmax) {this->jmax = jmax;}

protected:
  // Subspace and work array
  Vec *V;
  // Subspace min and max size
  PetscInt jmin, jmax;

  /// Variables only needed in compute step
  // phi, A*phi, and B*phi at each level
  std::vector<Vec*> Q, AQ, BQ;

  /// Protected methods
  // Remove the nullspace of a matrix from a vector
  PetscErrorCode Remove_NullSpace(Mat A, Vec x);
  // Prepare solver for compute step
  virtual PetscErrorCode Compute_Init() = 0;
  virtual PetscErrorCode Initialize_V(PetscInt &j) = 0;
  // Update the search space with the correction equation
  virtual PetscErrorCode Update_Search(Vec x, Vec residual, PetscReal rnorm) = 0;
  // Cleanup after compute step
  virtual PetscErrorCode Compute_Clean() = 0;
  // Update parts of preconditioner at each step
  virtual PetscErrorCode Update_Preconditioner(Vec residual, PetscScalar &rnorm,
                                               PetscScalar &Au_norm) = 0;
  // Output information
  virtual PetscErrorCode Print_Result() {
    return PetscFPrintf(comm, output, "PRINVIT found %i of a requested %i eigenvalues "
                        "after %i iterations\n\n", nev_conv, nev_req, it);
  }

};

#endif // PRINVIT_H_INCLUDED
