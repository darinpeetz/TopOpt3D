#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include "LOPGMRES.h"
#include "EigLab.h"

using namespace std;

// BLAS routines
extern "C"{
double ddot_(const int *N, const double *x, const int *incx, const double *y, const int *incy);
void daxpy_(const int *N, const double *a, const double *x, const int *incx,
    const double *y, const int *incy);
}

//TODO: Investigate the whether eigensolver in Eigen is sufficient or if solvers
// in BLAS/LAPACK are necessary for performance (for subspace problem)

/******************************************************************************/
/**                             Main constructor                             **/
/******************************************************************************/
LOPGMRES::LOPGMRES(MPI_Comm comm)
{
  this->comm = comm; Set_ID();
  nsweep = 2;
  PetscOptionsGetInt(NULL, NULL, "-LOPGMRES_Verbose", &verbose, NULL);
  PetscFOpen(this->comm, "stdout", "w", &output);
  file_opened = 1;
}

/******************************************************************************/
/**                              Main destructor                             **/
/******************************************************************************/
LOPGMRES::~LOPGMRES()
{
}

/******************************************************************************/
/**                       How much information to print                      **/
/******************************************************************************/
PetscErrorCode LOPGMRES::Set_Verbose(PetscInt verbose)
{
  this->verbose = verbose;
  PetscErrorCode ierr = PetscOptionsGetInt(NULL, NULL, "-LOPGMRES_Verbose", &this->verbose, NULL);
  CHKERRQ(ierr);
  return 0;
}

/******************************************************************************/
/**                      Creates multilevel hierarchy                        **/
/******************************************************************************/
PetscErrorCode LOPGMRES::Create_Hierarchy()
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
  {
    ierr = PetscFPrintf(comm, output, "Creating operators at each level\n"); CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(EIG_Hierarchy, 0, 0, 0, 0); CHKERRQ(ierr);
  for (unsigned int i = 0; i < P.size(); i++)
  {
    if (this->tau == SR || this->tau == SM || this->tau == SA)
    {
      if (this->A[i+1] == NULL)
      {
        ierr = MatPtAP(A[i], P[i], MAT_INITIAL_MATRIX, 1.0, A.data()+i+1); CHKERRQ(ierr);
      }
      K = A;
      B.resize(1);
    }
    if (this->tau == LR || this->tau == LM || this->tau == LA)
    {
      if (this->B[i+1] == NULL)
      {
        ierr = MatPtAP(B[i], P[i], MAT_INITIAL_MATRIX, 1.0, B.data()+i+1); CHKERRQ(ierr);
      }
      K = B;
      A.resize(1);
    }
  }
  ierr = PetscLogEventEnd(EIG_Hierarchy, 0, 0, 0, 0); CHKERRQ(ierr);

  return 0;
}

/******************************************************************************/
/**        Preps for computing the eigenmodes of the specified system        **/
/******************************************************************************/
PetscErrorCode LOPGMRES::Compute_Init()
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Initializing compute structures\n"); CHKERRQ(ierr);

  ierr = PetscLogEventBegin(EIG_Comp_Init, 0, 0, 0, 0); CHKERRQ(ierr);
  // Clear any old eigenvectors
  if (nev_conv > 0)
    ierr = VecDestroyVecs(nev_conv, &phi); CHKERRQ(ierr);

  // Check if MG cycle type was set at command line
  const PetscInt ct_length = 5;
  char cycle_type[ct_length];
  PetscBool cycle_type_set = PETSC_FALSE;
  ierr = PetscOptionsGetString(NULL, NULL, "-LOPGMRES_Cycle_Type", cycle_type,
                               ct_length, &cycle_type_set); CHKERRQ(ierr);
  if (cycle_type_set)
  {
    for (int i = 0; i < ct_length; i++)
    {
      if (cycle_type[i] == '\0')
        break;
      cycle_type[i] = toupper(cycle_type[i]);
    }
    if (!strcmp(cycle_type, "FULL"))
      cycle = FMGCycle;
    else if (!strcmp(cycle_type, "V"))
      cycle = VCycle;
    else
      PetscPrintf(comm, "Bad LOPGMRES_Cycle_Type given %s, should be \"FULL\" or \"V\"", cycle_type);
  }

  // Preallocate eigenvalues and eigenvectors at each level and get problem size
  Vec temp;
  Q.resize(1); AQ.resize(1); BQ.resize(1);
  ierr = MatCreateVecs(A[0], &temp, NULL); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(temp, Qsize, Q.data()); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(temp, Qsize, AQ.data()); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(temp, Qsize, BQ.data()); CHKERRQ(ierr);
  ierr = VecDestroy(&temp); CHKERRQ(ierr);
  lambda.setOnes(Qsize);
  ierr = VecGetSize(Q[0][0], &n); CHKERRQ(ierr);
  ierr = VecGetLocalSize(Q[0][0], &nlocal); CHKERRQ(ierr);

  // Determine size of search space
  ierr = PetscOptionsGetInt(NULL, NULL, "-LOPGMRES_jmin", &jmin, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-LOPGMRES_jmax", &jmax, NULL); CHKERRQ(ierr);
  if (jmin < 0)
    jmin = std::min( std::max(2*nev_req, 10), std::min(n/2,10));
  if (jmax < 0)
    jmax = std::min( std::max(4*nev_req, 25), std::min(n , 50));

  // Preallocate search space, work space, and eigenvectors
  ierr = VecDuplicateVecs(Q[0][0], jmax, &V); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(Q[0][0], jmax, &TempVecs); CHKERRQ(ierr);
  TempScal.setZero(std::max(jmax,Qsize));

  // Check for options in MG preconditioner
  ierr = PetscOptionsGetInt(NULL, NULL, "-LOPGMRES_Jacobi_nSweep", &nsweep, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-LOPGMRES_Jacobi_Weight", &w, NULL); CHKERRQ(ierr);
  // Preallocate for operators
  Dlist.resize(levels-1);
  xlist.resize(levels);
  flist.resize(levels);
  OPx.resize(levels-1);
  for (int ii = 0; ii < levels-1; ii++)
  {
    ierr = MatCreateVecs(K[ii+1], xlist.data()+ii+1, flist.data()+ii+1); CHKERRQ(ierr);
    ierr = MatCreateVecs(K[ii], Dlist.data()+ii, OPx.data()+ii); CHKERRQ(ierr);
    ierr = MatGetDiagonal(K[ii], Dlist[ii]); CHKERRQ(ierr);
  }

  // Prep coarse problem
  ierr = KSPCreate(comm, &ksp_coarse); CHKERRQ(ierr);
  ierr = KSPSetType(ksp_coarse, KSPPREONLY); CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp_coarse, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1); CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp_coarse, "LOPGMRES_coarse_"); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp_coarse); CHKERRQ(ierr);
  PC pc;
  ierr = KSPGetPC(ksp_coarse, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, PCBJACOBI); CHKERRQ(ierr);
  ierr = PCSetOptionsPrefix(pc, "LOPGMRES_coarse_"); CHKERRQ(ierr);
  ierr = PCSetFromOptions(pc); CHKERRQ(ierr);

  ierr = KSPSetOperators(ksp_coarse, K.back(), K.back()); CHKERRQ(ierr);
  ierr = KSPSetUp(ksp_coarse); CHKERRQ(ierr);
  PC sub_pc; KSP *sub_ksp;
  PetscInt blocks, first;
  ierr = PCBJacobiGetSubKSP(pc, &blocks, &first, &sub_ksp); CHKERRQ(ierr);
  if (blocks != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"blocks on this process, %D, is not one",blocks);
  ierr = KSPGetPC(sub_ksp[0], &sub_pc); CHKERRQ(ierr);
  ierr = PCSetType(sub_pc, PCLU); CHKERRQ(ierr);
  ierr = PCFactorSetShiftType(sub_pc, MAT_SHIFT_INBLOCKS); CHKERRQ(ierr);
  ierr = KSPSetTolerances(sub_ksp[0], PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1); CHKERRQ(ierr);
  ierr = KSPSetType(sub_ksp[0], KSPPREONLY); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EIG_Comp_Init, 0, 0, 0, 0); CHKERRQ(ierr);

  return 0;
}

/******************************************************************************/
/**                    Fill out the initial search space                     **/
/******************************************************************************/
PetscErrorCode LOPGMRES::Initialize_V(PetscInt &j)
{
  PetscErrorCode ierr = 0;

  PetscBool start;
  ierr = PetscOptionsHasName(NULL, NULL, "-LOPGMRES_Static_Start", &start); CHKERRQ(ierr);
  if (start){
    PetscInt first;
    PetscScalar *p_vec;
    ierr = MatGetOwnershipRange(A[0], &first, NULL); CHKERRQ(ierr);
    for (int ii = 0; ii < jmin; ii++)
    {
      ierr = VecGetArray(V[ii], &p_vec); CHKERRQ(ierr);
      for (int jj = 0; jj < nlocal; jj++)
        p_vec[jj] = pow(jj+first+1,ii);
      ierr = VecRestoreArray(V[ii], &p_vec); CHKERRQ(ierr);
    }
    j = jmin;
  }
  else{  
    PetscRandom random;
    ierr = PetscRandomCreate(comm, &random); CHKERRQ(ierr);
    for (int ii = 0; ii < jmin; ii++)
    {
      ierr = VecSetRandom(V[ii], random); CHKERRQ(ierr);
    }
    ierr = PetscRandomDestroy(&random);
    j = jmin;
  }

  return 0;
}

/******************************************************************************/
/**                    Update parts of the preconditioner                    **/
/******************************************************************************/
PetscErrorCode LOPGMRES::Update_Preconditioner(Vec residual,
                         PetscScalar &rnorm, PetscScalar &Au_norm)
{
  PetscErrorCode ierr = 0;

  ierr = VecNorm(residual, NORM_2, &rnorm); CHKERRQ(ierr);
  ierr = VecNorm(AQ[0][nev_conv], NORM_2, &Au_norm); CHKERRQ(ierr);

  return 0;
}

/******************************************************************************/
/**                      Clean up after compute phase                        **/
/******************************************************************************/
PetscErrorCode LOPGMRES::Compute_Clean()
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Cleaning up\n"); CHKERRQ(ierr);

  // Extract and sort converged eigenpairs from Q and lambda
  lambda.conservativeResize(nev_conv);
  MatrixPS empty(0,0);
  Eigen::ArrayXi order = Sorteig(empty, lambda);
  if (nev_conv > 0) {
    ierr = VecDuplicateVecs(Q[0][0], nev_conv, &phi); CHKERRQ(ierr); }
  for (int ii = 0; ii < nev_conv; ii++)
  {
    ierr = VecCopy(Q[0][order(ii)], phi[ii]); CHKERRQ(ierr);
  }

  // Destroy eigenvectors at each level
  ierr = VecDestroyVecs(Qsize, Q.data());  CHKERRQ(ierr);
  ierr = VecDestroyVecs(Qsize, AQ.data()); CHKERRQ(ierr);
  ierr = VecDestroyVecs(Qsize, BQ.data()); CHKERRQ(ierr);
  AQ.resize(0); BQ.resize(0);

  // Destroy workspace and search space
  ierr = VecDestroyVecs(jmax, &V); CHKERRQ(ierr);
  ierr = VecDestroyVecs(jmax, &TempVecs); CHKERRQ(ierr);

  // Destroy coarse problem
  ierr = KSPDestroy(&ksp_coarse); CHKERRQ(ierr);

  // Destroy Operators
  for (int ii = 0; ii < levels-1; ii++)
  {
    ierr = VecDestroy(xlist.data()+ii+1); CHKERRQ(ierr);
    ierr = VecDestroy(flist.data()+ii+1); CHKERRQ(ierr);
    ierr = VecDestroy(Dlist.data()+ii); CHKERRQ(ierr);
    ierr = VecDestroy(OPx.data()+ii); CHKERRQ(ierr);
  }
  Q.resize(0);

  return 0;
}

/******************************************************************************/
/**                  Set up multigrid for correction equation                **/
/******************************************************************************/
PetscErrorCode LOPGMRES::MGSetup(Vec f, PetscReal fnorm)
{
  return 0; // No action needed for LOPGMRES
}

/******************************************************************************/
/**                        Coarse solve for multigrid                        **/
/******************************************************************************/
PetscErrorCode LOPGMRES::Coarse_Solve()
{
  PetscErrorCode ierr = KSPSolve(ksp_coarse, flist.back(), xlist.back()); CHKERRQ(ierr);
  return 0;
}

/******************************************************************************/
/**                         Apply combined operator                          **/
/******************************************************************************/
PetscErrorCode LOPGMRES::ApplyOP(Vec x, Vec y, PetscInt level)
{
  PetscErrorCode ierr = 0;
  ierr = PetscLogEventBegin(EIG_ApplyOP[level], 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = MatMult(K[level],x,y); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EIG_ApplyOP[level], 0, 0, 0, 0); CHKERRQ(ierr);

  return 0;
}
