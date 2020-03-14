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
  // Set operators of eigensystem
  PetscErrorCode Set_Operators(Mat A, Mat B);
  // Setting target eigenvalues and number to find
  PetscErrorCode Set_Target(Tau tau, PetscInt nev, Nev_Type ntype);
  PetscErrorCode Set_Target(PetscScalar tau, PetscInt nev, Nev_Type ntype);
  // Solver
  PetscErrorCode Compute();
  // Search space size
  PetscErrorCode Set_jmin(PetscInt jmin) {return Update_jmin(jmin);}
  PetscErrorCode Set_jmax(PetscInt jmax) {return Update_jmax(jmax);}

protected:
  // Subspace and work array
  Vec *V;
  // Subspace current, min, and max size
  PetscInt j, jmin, jmax;
  // Subspace size set by option
  PetscBool jmin_set, jmax_set;

  /// Variables only needed in compute step
  // phi, A*phi, and B*phi at each level
  std::vector<Vec*> Q, AQ, BQ;

  /// Protected methods
  // Update the search space dimensions
  PetscErrorCode Update_jmin(PetscInt jmin=0);
  PetscErrorCode Update_jmax(PetscInt jmax=0);
  // Setup or destroy eigenvector storage space during compute phase
  PetscErrorCode Setup_Q();
  PetscErrorCode Destroy_Q();
  // Remove the nullspace of a matrix from a vector
  PetscErrorCode Remove_NullSpace(Mat A, Vec x);
  // Prepare solver for compute step
  virtual PetscErrorCode Compute_Init() = 0;
  virtual PetscErrorCode Initialize_V();
  // Update the search space with the correction equation
  virtual PetscErrorCode Update_Search(Vec x, Vec residual, PetscReal rnorm) = 0;
  // Cleanup after compute step
  virtual PetscErrorCode Compute_Clean();
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
