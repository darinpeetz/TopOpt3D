#ifndef JDMG_H_INCLUDED
#define JDMG_H_INCLUDED

#include <slepceps.h>
#include <vector>
#include <Eigen/Eigen>
#include <algorithm>
#include "PRINVIT.h"

PETSC_EXTERN PetscLogEvent EIG_Setup_Coarse, EIG_Comp_Coarse, EIG_MGSetup;
PETSC_EXTERN PetscLogEvent *EIG_ApplyOP1, *EIG_ApplyOP2, *EIG_ApplyOP3, *EIG_ApplyOP4;

/// The master structure containing all information to be carried between iterations
class JDMG : public PRINVIT
{
  /// Class variables
public:
  /// Class methods
  // Constructors
  JDMG() {JDMG(MPI_COMM_WORLD);}
  JDMG(MPI_Comm comm);
  // Destructor
  ~JDMG();
  // How much information to print
  PetscErrorCode Set_Verbose(PetscInt verbose);

private:
  // Solve the coarse scale eigenvalue problem
  PetscErrorCode Compute_Coarse();
  // Process to store PQ part of coarse operator
  PetscInt endrank;
  // Problem size
  PetscInt ncoarse, nlcoarse;
  // Convergence tolerance and tracking tolerance
  double epstr;
  // Flag if sigma is getting close to lambda_max
  bool vicinity;
  // Flag indicating if Compute_Init needs to be run
  bool prepped;
  // EPS object for coarse scale eigenvalue problem and KSP for solver
  EPS eps_coarse;

  /// Variables only needed in compute step
  std::vector<Mat> Acopy, Bcopy;
  Vec f_end, x_end;
  std::vector<Vec*> QMatP;
  PetscScalar sigma, sigma_old;

  /// Private methods
  // Prepare solver for compute step
  PetscErrorCode Compute_Init();
  PetscErrorCode Setup_Coarse();
  PetscErrorCode Create_Hierarchy();
  PetscErrorCode Initialize_V(PetscInt &j);
  // Cleanup after compute step
  PetscErrorCode Compute_Clean();
  // Update parts of preconditioner at each step
  PetscErrorCode Update_Preconditioner(Vec residual, PetscScalar &rnorm, PetscScalar &Au_norm);
  PetscErrorCode MGSetup(Vec f, PetscReal fnorm);
  // Set up coarse grid matrix
  PetscErrorCode MatShift();
  // Apply Operator at a given level
  PetscErrorCode ApplyOP(Vec x, Vec y, PetscInt level);
  // Coarse solver in multigrid
  PetscErrorCode Coarse_Solve();

};

#endif // JDMG_H_INCLUDED
