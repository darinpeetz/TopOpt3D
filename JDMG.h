#ifndef JDMG_H_INCLUDED
#define JDMG_H_INCLUDED

#include <slepceps.h>
#include <vector>
#include <Eigen/Eigen>
#include <algorithm>
#include "PRINVIT.h"

enum MG_Cycle_Type {VCycle, FMGCycle};

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
  // Get hierarchy from existing PCMG object
  PetscErrorCode PCMG_Extract(PC pcmg, bool isB=true, bool isA=false);
  PetscErrorCode Set_Hierarchy(std::vector<Mat> P,
    const std::vector<MPI_Comm> MG_comms = std::vector<MPI_Comm>());
  // Set which multigrid cycle to use
  void Set_Cycle(MG_Cycle_Type cycle) {this->cycle = cycle;}
  // Settings for the preconditioner
  void Set_Jacobi_Weight(PetscScalar w) {this->w = w;}
  void Set_Jacobi_nSweep(PetscInt nsweep) {this->nsweep = nsweep;}

private:
  // Multigrid cyle to use
  MG_Cycle_Type cycle;
  // Multigrid prolongation operators
  std::vector<Mat> P;
  // Operator Matrices in multigrid
  std::vector<Mat> K;
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
  std::vector<Vec*> QMatP;
  PetscScalar sigma, sigma_old;
  // Number of sweeps and weight for weighted Jacobi smoother
  PetscInt nsweep;
  PetscScalar w;
  // Number of levels in hierarchy
  PetscInt levels;
  // KSP for coarse solver
  KSP ksp_coarse;
  // Parts of the operators
  std::vector<Vec> Dlist;
  std::vector<Vec> xlist;
  std::vector<Vec> flist;
  Vec f_end, x_end;
  std::vector<Vec> OPx;

  /// Private methods
  // Solve the coarse scale eigenvalue problem
  PetscErrorCode Compute_Coarse();
  // Prepare solver for compute step
  PetscErrorCode Compute_Init();
  PetscErrorCode Setup_Coarse();
  PetscErrorCode Create_Hierarchy();
  PetscErrorCode Initialize_V(PetscInt &j);
  // Update the search space with the correction equation
  PetscErrorCode Update_Search(Vec x, Vec residual, PetscReal rnorm);
  // Cleanup after compute step
  PetscErrorCode Compute_Clean();
  // Update parts of preconditioner at each step
  PetscErrorCode Update_Preconditioner(Vec residual, PetscScalar &rnorm, PetscScalar &Au_norm);
  PetscErrorCode MGSetup(Vec f, PetscReal fnorm);
  // Set up coarse grid matrix
  PetscErrorCode MatShift();
  // Multigrid cycles
  PetscErrorCode FullMGSolve(Vec x, Vec f);
  PetscErrorCode MGSolve(Vec x, Vec f);
  // Weighted Jacobi smoothing
  PetscErrorCode WJac(Vec y, Vec x, PetscInt level);
  // Apply Operator at a given level
  PetscErrorCode ApplyOP(Vec x, Vec y, PetscInt level);
  // Coarse solver in multigrid
  PetscErrorCode Coarse_Solve();
  // Output information
  PetscErrorCode Print_Result() {
    return PetscFPrintf(comm, output, "JDMG found %i of a requested %i eigenvalues "
                        "after %i iterations\n\n", nev_conv, nev_req, it);
  }
};

#endif // JDMG_H_INCLUDED
