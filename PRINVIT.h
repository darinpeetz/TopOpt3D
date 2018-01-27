#ifndef PRINVIT_H_INCLUDED
#define PRINVIT_H_INCLUDED

#include <slepceps.h>
#include <vector>
#include <Eigen/Eigen>
#include <algorithm>
#include "EigenPeetz.h"

PETSC_EXTERN PetscLogEvent EIG_Initialize, EIG_Prep, EIG_Convergence, EIG_Expand, EIG_Update;
PETSC_EXTERN PetscLogEvent EIG_Comp_Init, EIG_Hierarchy, EIG_Precondition, EIG_Jacobi, *EIG_ApplyOP;

enum MG_Cycle_Type {VCycle, FMGCycle};

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
  // Get hierarchy from existing PCMG object
  PetscErrorCode PCMG_Extract(PC pcmg, bool isB=true, bool isA=false);
  PetscErrorCode Set_Hierarchy(std::vector<Mat> P, const std::vector<MPI_Comm> MG_comms = std::vector<MPI_Comm>());
  // Set which multigrid cycle to use
  void Set_Cycle(MG_Cycle_Type cycle) {this->cycle = cycle;}
  // Solver
  PetscErrorCode Compute();
  // Settings for the preconditioner
  void Set_Jacobi_Weight(PetscScalar w) {this->w = w;}
  void Set_Jacobi_nSweep(PetscInt nsweep) {this->nsweep = nsweep;}
  void Set_jmin(PetscInt jmin) {this->jmin = jmin;}
  void Set_jmax(PetscInt jmax) {this->jmax = jmax;}

protected:
  // Multigrid cyle to use
  MG_Cycle_Type cycle;
  // Multigrid prolongation operators
  std::vector<Mat> P;
  // Operator Matrices in multigrid
  std::vector<Mat> K;
  // Number of levels in hierarchy
  PetscInt levels;
  // Subspace and work array
  Vec *V;
  // Subspace min and max size
  PetscInt jmin, jmax;
  // KSP for coarse solver
  KSP ksp_coarse;

  /// Variables only needed in compute step
  // phi, A*phi, and B*phi at each level
  std::vector<Vec*> Q, AQ, BQ;
  // Number of sweeps and weight for weighted Jacobi smoother
  PetscInt nsweep;
  PetscScalar w;
  // Parts of the operators
  std::vector<Vec> Dlist;
  std::vector<Vec> xlist;
  std::vector<Vec> flist;
  Vec f_end, x_end;
  std::vector<Vec> OPx;

  /// Protected methods
  // Prepare solver for compute step
  virtual PetscErrorCode Compute_Init() = 0;
  virtual PetscErrorCode Create_Hierarchy() = 0;
  virtual PetscErrorCode Initialize_V(PetscInt &j) = 0;
  // Cleanup after compute step
  virtual PetscErrorCode Compute_Clean() = 0;
  // Update parts of preconditioner at each step
  virtual PetscErrorCode Update_Preconditioner(Vec residual, PetscScalar &rnorm, PetscScalar &Au_norm) = 0;
  virtual PetscErrorCode MGSetup(Vec f, PetscReal fnorm) = 0;
  // Multigrid cycles
  PetscErrorCode FullMGSolve(Vec x, Vec f);
  PetscErrorCode MGSolve(Vec x, Vec f);
  // Weighted Jacobi smoothing
  PetscErrorCode WJac(Vec y, Vec x, PetscInt level);
  // Apply Operator at a given level
  virtual PetscErrorCode ApplyOP(Vec x, Vec y, PetscInt level) = 0;
  // Coarse solver in multigrid
  virtual PetscErrorCode Coarse_Solve() = 0;

};

#endif // PRINVIT_H_INCLUDED
