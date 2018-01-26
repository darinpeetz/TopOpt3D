#include <iostream>
#include <fstream>
#include <numeric>
#include <math.h>
#include "Functions.h"
#include "TopOpt.h"
#include "EigLab.h"
#include "EigenPeetz.h"
#include "LOPGMRES.h"
#include <unsupported/Eigen/KroneckerProduct>
#include <sstream>

using namespace std;
typedef Eigen::Matrix<PetscScalar, -1, -1> MatrixPS;
typedef Eigen::Matrix<PetscScalar, -1, 1>  VectorPS;

extern PetscLogStage stage_JD, stage_JDMG;

namespace Functions
{
  int DiagMassFnc( TopOpt *topOpt, Mat &M, Eigen::VectorXd &dMdy );

  int Dynamic( TopOpt *topOpt, double *lambda, double *grad, PetscInt &nevals )
  {
    PetscErrorCode ierr = 0;

    if (topOpt->verbose >= 3)
    {
      ierr = PetscFPrintf(topOpt->comm, topOpt->output, "Performing dynamic analysis\n"); CHKERRQ(ierr);
    }

    /// Assemble Mass matrix and get sensitivity information
    Eigen::VectorXd dMdy;
    Mat M;
    ierr = DiagMassFnc( topOpt, M, dMdy ); CHKERRQ(ierr);
    /// Remove fixed and spring dof from M (and K if necessary)
    ierr = MatZeroRowsColumns(M, topOpt->fixedDof.size(),
                       topOpt->fixedDof.data(), 1e-8, NULL, NULL); CHKERRQ(ierr);
    if (topOpt->nSpringDof > 0)
    {
      ierr = MatZeroRowsColumns(M, topOpt->springDof.size(),
                         topOpt->springDof.data(), 1e-8, NULL, NULL); CHKERRQ(ierr);
      ierr = MatZeroRowsColumns(topOpt->K, topOpt->springDof.size(),
                         topOpt->springDof.data(), 1.0, NULL, NULL); CHKERRQ(ierr);
      ierr = KSPSetOperators(topOpt->KUF, topOpt->K, topOpt->K); CHKERRQ(ierr);
    }

PetscInt total_conv = 0, iter = 10, nev = 5, its = 0;
FILE *timings;
ierr = PetscFOpen(topOpt->comm, "Timings", "a", &timings); CHKERRQ(ierr);
ierr = PetscOptionsGetInt(NULL, NULL, "-nepssolves", &iter, NULL); CHKERRQ(ierr);
ierr = PetscOptionsGetInt(NULL, NULL, "-nevals", &nev, NULL); CHKERRQ(ierr);
PetscBool run = PETSC_FALSE;
PetscOptionsHasName(NULL, NULL, "-run_KS", &run);
double t0 = MPI_Wtime();
for (int i = 0; i < run*iter; i++)
{
  EPS eps; ST st; KSP ksp; PC pc;
  ierr = EPSCreate(topOpt->comm, &eps); CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps, EPS_GHEP); CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL); CHKERRQ(ierr);
  //ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE); CHKERRQ(ierr);
  ierr = EPSSetTolerances(eps, pow(10,log10(2*topOpt->nNode)/2-8), PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = EPSGetST(eps, &st); CHKERRQ(ierr);
  ierr = STGetKSP(st, &ksp); CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPPREONLY); CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, PCLU); CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pc, MATSOLVERSUPERLU_DIST); CHKERRQ(ierr);
  ierr = EPSSetDimensions(eps, nev, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = EPSSetType(eps, EPSKRYLOVSCHUR); CHKERRQ(ierr);
  ierr = EPSSetOperators(eps, M, topOpt->K); CHKERRQ(ierr);
  //ierr = EPSSetOperators(eps, topOpt->K, M); CHKERRQ(ierr);
  ierr = EPSSetOptionsPrefix(eps, "KS_"); CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
  ierr = EPSSolve(eps); CHKERRQ(ierr);
  PetscInt nconv = 0;
  ierr = EPSGetConverged(eps, &nconv); CHKERRQ(ierr);
  total_conv += nconv;
  PetscInt it = 0;
  ierr = EPSGetIterationNumber(eps, &it); CHKERRQ(ierr);
  its += it;
  ierr = EPSDestroy(&eps); CHKERRQ(ierr);
}
ierr = PetscFPrintf(topOpt->comm, timings, "Using KRYLOVSCHUR, frequency found %i eigenvalues in %1.8g seconds, taking %i iterations for a problem of size %i on %i processor\n", total_conv, MPI_Wtime()-t0, its, 2*topOpt->nNode, topOpt->nprocs); CHKERRQ(ierr);

total_conv = 0;
its = 0;
PetscOptionsHasName(NULL, NULL, "-run_LOBPCG", &run);
t0 = MPI_Wtime();
for (int i = 0; i < run*iter; i++)
{
  EPS eps; ST st; KSP ksp; PC pc, JUNK;
  ierr = EPSCreate(topOpt->comm, &eps); CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps, EPS_GHEP); CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL); CHKERRQ(ierr);
  ierr = EPSSetTolerances(eps, pow(10,log10(2*topOpt->nNode)/2-8), PETSC_DEFAULT); CHKERRQ(ierr);
  // Setting up krylov ST structure
  ierr = EPSGetST(eps, &st); CHKERRQ(ierr);
  ierr = STGetKSP(st, &ksp); CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, PCMG); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, topOpt->K, topOpt->K); CHKERRQ(ierr);
  PetscInt nlevels = topOpt->PR.size()+1;
  ierr = PCMGSetLevels(pc, nlevels, NULL); CHKERRQ(ierr);
  ierr = PCMGSetType(pc, PC_MG_FULL); CHKERRQ(ierr);
  ierr = PCMGSetGalerkin(pc, PETSC_TRUE); CHKERRQ(ierr);
  for (int i = 1; i < nlevels; i++) {
    ierr = PCMGSetInterpolation(pc, i, topOpt->PR[nlevels-i-1]); CHKERRQ(ierr); }
  // Use direct solve on coarse level
  ierr = PCSetUp(pc); CHKERRQ(ierr);
  KSP smooth_ksp, *sub_ksp; PC smooth_pc, sub_pc; PetscInt blocks, first;
  ierr = PCMGGetCoarseSolve(pc, &smooth_ksp); CHKERRQ(ierr);
  ierr = KSPSetType(smooth_ksp, KSPPREONLY); CHKERRQ(ierr);
  ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
  ierr = PCSetType(smooth_pc, PCBJACOBI); CHKERRQ(ierr);
  ierr = PCSetUp(smooth_pc); CHKERRQ(ierr);
  ierr = KSPSetUp(smooth_ksp); CHKERRQ(ierr);
  ierr = PCBJacobiGetSubKSP(smooth_pc, &blocks, &first, &sub_ksp); CHKERRQ(ierr);
  ierr = KSPGetPC(sub_ksp[0], &sub_pc); CHKERRQ(ierr);
  ierr = PCSetType(sub_pc, PCLU); CHKERRQ(ierr);
  ierr = PCFactorSetShiftType(sub_pc, MAT_SHIFT_INBLOCKS); CHKERRQ(ierr);
  ierr = KSPSetTolerances(sub_ksp[0], PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1); CHKERRQ(ierr);
  ierr = KSPSetType(sub_ksp[0], KSPPREONLY); CHKERRQ(ierr);
  ierr = PCSetUp(pc); CHKERRQ(ierr);
  // Use Jacobi smoothing
  for (int i = 1; i < nlevels; i++)
  {
    ierr = PCMGGetSmoother(pc, i, &smooth_ksp); CHKERRQ(ierr);
    ierr = KSPSetType(smooth_ksp, KSPRICHARDSON); CHKERRQ(ierr);
    ierr = KSPRichardsonSetScale(smooth_ksp, 5.0/10.0); CHKERRQ(ierr);
    ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
    ierr = PCSetType(smooth_pc, PCJACOBI); CHKERRQ(ierr);
  }

  ierr = EPSSetDimensions(eps, nev, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = EPSSetType(eps, EPSLOBPCG); CHKERRQ(ierr);
  ierr = EPSSetOperators(eps, topOpt->K, M); CHKERRQ(ierr);
  ierr = EPSSetOptionsPrefix(eps, "LOBPCG_"); CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
  ierr = EPSSolve(eps); CHKERRQ(ierr);
  PetscInt nconv = 0;
  ierr = EPSGetConverged(eps, &nconv); CHKERRQ(ierr);
  total_conv += nconv;
  PetscInt it = 0;
  ierr = EPSGetIterationNumber(eps, &it); CHKERRQ(ierr);
  its += it;
  ierr = KSPSetPC(ksp, JUNK); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(topOpt->K, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(topOpt->K, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = EPSDestroy(&eps); CHKERRQ(ierr);
}
ierr = PetscFPrintf(topOpt->comm, timings, "Using LOBPCG, frequency found %i eigenvalues in %1.8g seconds, taking %i iterations for a problem of size %i on %i processors with %i levels in the multigrid\n", total_conv, MPI_Wtime()-t0, its, 2*topOpt->nNode, topOpt->nprocs, topOpt->PR.size()+1); CHKERRQ(ierr);

total_conv = 0;
its = 0;
PetscOptionsHasName(NULL, NULL, "-run_JD", &run);
t0 = MPI_Wtime();
for (int i = 0; i < run*iter; i++)
{
  EPS eps; ST st; KSP ksp; PC pc;
  // Setting up eps solver
  ierr = EPSCreate(topOpt->comm, &eps); CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps, EPS_GHEP); CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL); CHKERRQ(ierr);
  //ierr = EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL); CHKERRQ(ierr);
  ierr = EPSSetTolerances(eps, pow(10,log10(2*topOpt->nNode)/2-8), PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = EPSSetDimensions(eps, nev, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = EPSSetType(eps, EPSJD); CHKERRQ(ierr);
  ierr = EPSSetOperators(eps, topOpt->K, M); CHKERRQ(ierr);
  //ierr = EPSSetOperators(eps, M, topOpt->K); CHKERRQ(ierr);
  // Setting up krylov ST structure
  ierr = EPSGetST(eps, &st); CHKERRQ(ierr);
  ierr = STGetKSP(st, &ksp); CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, PCMG); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, topOpt->K, topOpt->K); CHKERRQ(ierr);
  PetscInt nlevels = topOpt->PR.size()+1;
  ierr = PCMGSetLevels(pc, nlevels, NULL); CHKERRQ(ierr);
  ierr = PCMGSetType(pc, PC_MG_MULTIPLICATIVE); CHKERRQ(ierr);
  ierr = PCMGSetGalerkin(pc, PETSC_TRUE); CHKERRQ(ierr);
  for (int i = 1; i < nlevels; i++) {
    ierr = PCMGSetInterpolation(pc, i, topOpt->PR[nlevels-i-1]); CHKERRQ(ierr); }
  // Use direct solve on coarse level
  ierr = PCSetUp(pc); CHKERRQ(ierr);
  KSP smooth_ksp, *sub_ksp; PC smooth_pc, sub_pc; PetscInt blocks, first;
  ierr = PCMGGetCoarseSolve(pc, &smooth_ksp); CHKERRQ(ierr);
  ierr = KSPSetType(smooth_ksp, KSPPREONLY); CHKERRQ(ierr);
  ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
  ierr = PCSetType(smooth_pc, PCBJACOBI); CHKERRQ(ierr);
  ierr = PCSetUp(smooth_pc); CHKERRQ(ierr);
  ierr = KSPSetUp(smooth_ksp); CHKERRQ(ierr);
  ierr = PCBJacobiGetSubKSP(smooth_pc, &blocks, &first, &sub_ksp); CHKERRQ(ierr);
  ierr = KSPGetPC(sub_ksp[0], &sub_pc); CHKERRQ(ierr);
  ierr = PCSetType(sub_pc, PCLU); CHKERRQ(ierr);
  ierr = PCFactorSetShiftType(sub_pc, MAT_SHIFT_INBLOCKS); CHKERRQ(ierr);
  ierr = KSPSetTolerances(sub_ksp[0], PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1); CHKERRQ(ierr);
  ierr = KSPSetType(sub_ksp[0], KSPPREONLY); CHKERRQ(ierr);
  ierr = PCSetUp(pc); CHKERRQ(ierr);
  // Use Jacobi smoothing
  for (int i = 1; i < nlevels; i++)
  {
    ierr = PCMGGetSmoother(pc, i, &smooth_ksp); CHKERRQ(ierr);
    ierr = KSPSetType(smooth_ksp, KSPRICHARDSON); CHKERRQ(ierr);
    ierr = KSPRichardsonSetScale(smooth_ksp, 5.0/10.0); CHKERRQ(ierr);
    ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
    ierr = PCSetType(smooth_pc, PCJACOBI); CHKERRQ(ierr);
  }

  // Solve the eigenvalue problem
  ierr = EPSSetOptionsPrefix(eps, "JD_"); CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
  ierr = EPSSolve(eps); CHKERRQ(ierr);
  PetscInt nconv = 0;
  ierr = EPSGetConverged(eps, &nconv); CHKERRQ(ierr);
  total_conv += nconv;
  PetscInt it = 0;
  ierr = EPSGetIterationNumber(eps, &it); CHKERRQ(ierr);
  its += it;

  ierr = EPSDestroy(&eps); CHKERRQ(ierr);
}
ierr = PetscFPrintf(topOpt->comm, timings, "Using built-in JD, frequency found %i eigenvalues in %1.8g seconds, taking %i iterations for a problem of size %i on %i processors with %i levels in the multigrid\n", total_conv, MPI_Wtime()-t0, its, 2*topOpt->nNode, topOpt->nprocs, topOpt->PR.size()+1); CHKERRQ(ierr);

//ierr = PetscLogStagePop(); CHKERRQ(ierr);
//MPI_Barrier(topOpt->comm);
ierr = PetscLogStagePush(stage_JDMG); CHKERRQ(ierr);

total_conv = 0; its = 0;
PetscOptionsHasName(NULL, NULL, "-run_JDMG", &run);
t0 = MPI_Wtime();
for (int i = 0; i < run*iter; i++)
{
  // Create JDMG instance
  JDMG jdmg(topOpt->comm);
  jdmg.Set_Verbose(topOpt->verbose);
  jdmg.Set_File(topOpt->output);
  // Get restrictors from FEM problem
  ierr = jdmg.Set_Hierarchy(topOpt->PR, topOpt->MG_comms); CHKERRQ(ierr);
  // Set Operators
  jdmg.Set_Operators(topOpt->K, M);
  // Set target eigenvalues
  Nev_Type target_type = TOTAL_NEV;
  jdmg.Set_Target(SR, nev, target_type);
  jdmg.Set_Tol(pow(10,log10(2*topOpt->nNode)/2-8));
  jdmg.Set_Cycle(VCycle);
  ierr = jdmg.Compute(); CHKERRQ(ierr);
  total_conv += jdmg.Get_nev_conv();
  its += jdmg.Get_Iterations();
}
ierr = PetscFPrintf(topOpt->comm, timings, "Using JDMG, frequency found %i eigenvalues in %1.8g seconds, taking %i iterations for a problem of size %i on %i processors with %i levels in the multigrid\n", total_conv, MPI_Wtime()-t0, its, 2*topOpt->nNode, topOpt->nprocs, topOpt->PR.size()+1); CHKERRQ(ierr);
ierr = PetscLogStagePop(); CHKERRQ(ierr);

total_conv = 0; its = 0;
PetscOptionsHasName(NULL, NULL, "-run_LOPGMRES", &run);
t0 = MPI_Wtime();
for (int i = 0; i < run*iter; i++)
{
  // Create JDMG instance
  LOPGMRES jdmg(topOpt->comm);
  jdmg.Set_Verbose(topOpt->verbose);
  jdmg.Set_File(topOpt->output);
  // Get restrictors from FEM problem
  ierr = jdmg.Set_Hierarchy(topOpt->PR, topOpt->MG_comms); CHKERRQ(ierr);
  // Set Operators
  jdmg.Set_Operators(topOpt->K, M);
  // Set target eigenvalues
  Nev_Type target_type = TOTAL_NEV;
  jdmg.Set_Target(SR, nev, target_type);
  jdmg.Set_Tol(pow(10,log10(2*topOpt->nNode)/2-8));
  jdmg.Set_Cycle(VCycle);
  ierr = jdmg.Compute(); CHKERRQ(ierr);
  total_conv += jdmg.Get_nev_conv();
  its += jdmg.Get_Iterations();
}
ierr = PetscFPrintf(topOpt->comm, timings, "Using LOPGMRES, frequency found %i eigenvalues in %1.8g seconds, taking %i iterations for a problem of size %i on %i processors with %i levels in the multigrid\n", total_conv, MPI_Wtime()-t0, its, 2*topOpt->nNode, topOpt->nprocs, topOpt->PR.size()+1); CHKERRQ(ierr);
ierr = PetscFClose(topOpt->comm, timings); CHKERRQ(ierr);

    return 0;
  }

  /********************************************************************/
  /*                  Creates the diagonal mass matrix               **/
  /********************************************************************/
  int DiagMassFnc( TopOpt *topOpt, Mat &M, Eigen::VectorXd &dMdy )
  {
    PetscErrorCode ierr = 0;

    // Initialize M
    ierr = MatCreate(topOpt->comm, &M); CHKERRQ(ierr);
    ierr = MatSetSizes(M, topOpt->numDims*topOpt->nLocNode, topOpt->numDims*topOpt->nLocNode,
                topOpt->numDims*topOpt->nNode, topOpt->numDims*topOpt->nNode); CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(M,"M_"); CHKERRQ(ierr);
    ierr = MatSetFromOptions(M); CHKERRQ(ierr);
    ArrayXPI onDiag = ArrayXPI::Ones(topOpt->nLocNode);
    ArrayXPI offDiag = ArrayXPI::Zero(topOpt->nLocNode);
    ierr = MatXAIJSetPreallocation(M, topOpt->numDims, onDiag.data(), offDiag.data(), 0, 0); CHKERRQ(ierr);

    // Mesh characteristics
    const short NE = topOpt->element.cols(), DN = topOpt->numDims, DE = NE*DN;
    // Track construction of Ks, dKs
    long dMmarker = 0;
    dMdy.resize( topOpt->nLocElem*(long)pow(DE,2) );

    // Get pointers to Petsc vectors
    const PetscScalar *p_V, *p_dVdy;
    ierr = VecGetArrayRead(topOpt->V, &p_V); CHKERRQ(ierr);
    ierr = VecGetArrayRead(topOpt->dVdy, &p_dVdy); CHKERRQ(ierr);

    MatrixPS mMat = 1.0/pow(2,topOpt->numDims)/topOpt->numDims*
            topOpt->elemSize(0)*topOpt->density*Eigen::MatrixXd::Identity(DE, DE);
    Eigen::Map< VectorPS > mVec(mMat.data(), mMat.size());
    MatrixPS nodeMat(topOpt->numDims, topOpt->numDims);
    /// Loop over elements
    for (long el = 0; el < topOpt->element.rows(); el++)
    {
      if (!topOpt->regular)
      {
        mMat.setIdentity();
        mMat *= 1.0/pow(2,topOpt->numDims)/topOpt->numDims *
            topOpt->density * topOpt->elemSize(0);
      }

      /// Fill in the sensitivity dMdy
      if (el < topOpt->nLocElem)
      {
        dMdy.segment(dMmarker, mVec.size()) = p_dVdy[el] * mVec;
        dMmarker += mVec.size();
      }

      /// Loop over indices to fill in M
      for (int n = 0; n < NE; n++) // Looping over rows
      {
        PetscInt node = topOpt->element(el,n);
        if (node < topOpt->nLocNode) // If node is local to this process
        {
          nodeMat = p_V[el]*mMat.block(n*topOpt->numDims, n*topOpt->numDims,
              topOpt->numDims, topOpt->numDims);
          PetscInt row = topOpt->gNode(node);
          ierr = MatSetValuesBlocked(M, 1, &row, 1, &row, nodeMat.data(), ADD_VALUES); CHKERRQ(ierr);
        }
      }
    }

    ierr = MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(topOpt->V, &p_V); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(topOpt->dVdy, &p_dVdy); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatDiagonalSet(M, topOpt->MLump, ADD_VALUES); CHKERRQ(ierr);

    return 0;
  }
}
