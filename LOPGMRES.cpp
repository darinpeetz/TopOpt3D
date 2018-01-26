#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include "LOPGMRES.h"
#include "EigLab.h"

using namespace std;

typedef Eigen::Matrix<PetscScalar, -1, -1> MatrixPS;
typedef Eigen::Array<PetscScalar, -1, 1>  ArrayPS;

extern PetscLogEvent EIG_Initialize, EIG_Prep, EIG_Convergence, EIG_Expand, EIG_Update;
extern PetscLogEvent EIG_Comp_Init, EIG_Hierarchy, EIG_Setup_Coarse, EIG_Comp_Coarse;
extern PetscLogEvent EIG_MGSetUp, EIG_Precondition, EIG_Jacobi, *EIG_ApplyOP;
extern PetscLogEvent *EIG_ApplyOP1, *EIG_ApplyOP2, *EIG_ApplyOP3, *EIG_ApplyOP4;

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
  levels = 0;
  n = 0;
  nev_req = 6; nev_conv = 0;
  Qsize = nev_req;
  tau = LM;
  tau_num = 0;
  eps = 1e-6;
  epstr = 1e-3;
  jmin = -1; jmax = -1;
  maxit = 500;
  nsweep = 2;
  w = 4.0/7;
  verbose = 0;
  PetscOptionsGetInt(NULL, NULL, "-LOPGMRES_Verbose", &verbose, NULL);
  PetscFOpen(this->comm, "stdout", "w", &output);
  file_opened = 1;
  cycle = FMGCycle;
}

/******************************************************************************/
/**                              Main destructor                             **/
/******************************************************************************/
LOPGMRES::~LOPGMRES()
{
  for (unsigned int ii = 0; ii < P.size(); ii++)
    MatDestroy(P.data()+ii);
  for (unsigned int ii = 0; ii < A.size(); ii++)
    MatDestroy(A.data()+ii);
  for (unsigned int ii = 0; ii < B.size(); ii++)
    MatDestroy(B.data()+ii);
  VecDestroyVecs(nev_conv, &phi);
  if (file_opened)
    Close_File();
}

/******************************************************************************/
/**                       How much information to print                      **/
/******************************************************************************/
PetscErrorCode LOPGMRES::Set_Verbose(PetscInt verbose)
{
  this->verbose = verbose;
  Close_File();
  PetscErrorCode ierr = PetscOptionsGetInt(NULL, NULL, "-LOPGMRES_Verbose", &this->verbose, NULL);
  CHKERRQ(ierr);
  return 0;
}

/******************************************************************************/
/**              Designating an already opened file for output               **/
/******************************************************************************/
PetscErrorCode LOPGMRES::Set_File(FILE *output)
{
  PetscErrorCode ierr = 0;
  if (file_opened)
    ierr = Close_File(); CHKERRQ(ierr);
  this->output = output;
  file_opened = false;
  return ierr;
}

/******************************************************************************/
/**                      Where to print the information                      **/
/******************************************************************************/
PetscErrorCode LOPGMRES::Open_File(const char filename[])
{
  PetscErrorCode ierr = 0;
  if (this->file_opened)
    ierr = Close_File(); CHKERRQ(ierr);
  ierr = PetscFOpen(comm, filename, "w", &output); CHKERRQ(ierr);
  file_opened = true;
  return ierr;
}

/******************************************************************************/
/**                       Set operators of eigensystem                       **/
/******************************************************************************/
PetscErrorCode LOPGMRES::Set_Operators(Mat A, Mat B)
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Setting operators\n"); CHKERRQ(ierr);

  if (this->A.size() > 0)
  {
    ierr = MatDestroy(this->A.data()); CHKERRQ(ierr);
    this->A[0] = A;
    ierr = PetscObjectReference((PetscObject)A); CHKERRQ(ierr);
  }
  else
  {
    this->A.push_back(A);
    ierr = PetscObjectReference((PetscObject)A); CHKERRQ(ierr);
  }
  if (this->B.size() > 0)
  {
    ierr = MatDestroy(this->B.data()); CHKERRQ(ierr);
    this->B[0] = B;
    ierr = PetscObjectReference((PetscObject)B); CHKERRQ(ierr);
  }
  else
  {
    this->B.push_back(B);
    ierr = PetscObjectReference((PetscObject)B); CHKERRQ(ierr);
  }

  return 0;
}

/******************************************************************************/
/**                     Extract hierarchy from PCMG object                   **/
/******************************************************************************/
PetscErrorCode LOPGMRES::PCMG_Extract(PC pcmg, bool isB, bool isA)
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Extracting hierarchy from PC object\n"); CHKERRQ(ierr);

  KSP smoother;
  ierr = PCMGGetLevels(pcmg, &levels); CHKERRQ(ierr);
  P.resize(levels-1, NULL);
  for (unsigned int ii = 1; ii < A.size(); ii++) {
    ierr = MatDestroy(A.data()+ii); CHKERRQ(ierr);
    A[ii] = NULL;
  }
  for (unsigned int ii = 1; ii < B.size(); ii++) {
    ierr = MatDestroy(B.data()+ii); CHKERRQ(ierr);
    B[ii] = NULL;
  }
  A.resize(levels, NULL); B.resize(levels, NULL);
  for (PetscInt ii = levels-1, jj = 0; ii > 0; ii--, jj++)
  {
    ierr = PCMGGetInterpolation(pcmg, ii, P.data()+jj); CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)P[jj]); CHKERRQ(ierr);
    if (isA)
    {
      ierr = PCMGGetSmoother(pcmg, ii, &smoother); CHKERRQ(ierr);
      ierr = KSPGetOperators(smoother, A.data()+jj+1, NULL); CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)A[jj+1]); CHKERRQ(ierr);
    }
    if (isB)
    {
      ierr = PCMGGetSmoother(pcmg, ii-1, &smoother); CHKERRQ(ierr);
      ierr = KSPGetOperators(smoother, B.data()+jj+1, NULL); CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)B[jj+1]); CHKERRQ(ierr);
    }
  }

  return 0;
}

/******************************************************************************/
/**                            Use given hierarchy                           **/
/******************************************************************************/
PetscErrorCode LOPGMRES::Set_Hierarchy(const std::vector<Mat> P, const std::vector<MPI_Comm> MG_comms)
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Setting hierarchy from list of interpolators\n"); CHKERRQ(ierr);

  this->P = P;
  for (unsigned int ii = 0; ii < this->P.size(); ii++)
  {
    ierr = PetscObjectReference((PetscObject)this->P[ii]); CHKERRQ(ierr);
  }

  for (unsigned int ii = 1; ii < A.size(); ii++) {
    ierr = MatDestroy(A.data()+ii); CHKERRQ(ierr);
    A[ii] = NULL;
  }
  for (unsigned int ii = 1; ii < B.size(); ii++) {
    ierr = MatDestroy(B.data()+ii); CHKERRQ(ierr);
    B[ii] = NULL;
  }

  levels = P.size()+1; A.resize(levels, NULL); B.resize(levels, NULL);
  this->MG_comms.resize(levels);
  if (MG_comms.size() == 0)
    std::fill(this->MG_comms.begin(), this->MG_comms.end(), comm);
  else if (MG_comms.size() == 1)
    std::fill(this->MG_comms.begin(), this->MG_comms.end(), MG_comms[0]);
  else if ((PetscInt)MG_comms.size() == levels)
    this->MG_comms = MG_comms;
  else
    SETERRQ(comm, PETSC_ERR_ARG_SIZ, "List of communicators does not match size of hierarchy");

  return ierr;
}

/******************************************************************************/
/**              Sets target eigenvalues and number to find                  **/
/******************************************************************************/
void LOPGMRES::Set_Target(Tau tau, PetscInt nev, Nev_Type ntype)
{
  if (this->verbose >= 3)
    PetscFPrintf(comm, output, "Setting target eigenvalues\n");
  this->tau = tau;
  this->nev_req = nev;
  this->nev_type = ntype;
  this->Qsize = nev_req + (this->nev_type == TOTAL_NEV ? 0 : 6);

  return;
}
void LOPGMRES::Set_Target(PetscScalar tau, PetscInt nev, Nev_Type ntype)
{ 
  if (this->verbose >= 3)
    PetscFPrintf(comm, output, "Setting target eigenvalues\n");
  this->tau = NUMERIC;
  this->tau_num = tau;
  this->nev_req = nev;
  this->nev_type = ntype;
  this->Qsize = nev_req + (this->nev_type == TOTAL_NEV ? 0 : 6);
  return;
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
  }
  else{  
    PetscRandom random;
    ierr = PetscRandomCreate(comm, &random); CHKERRQ(ierr);
    for (int ii = 0; ii < jmin; ii++)
    {
      ierr = VecSetRandom(V[ii], random); CHKERRQ(ierr);
    }
    ierr = PetscRandomDestroy(&random);
  }

  ierr = VecDuplicateVecs(Q[0][0], jmax, &TempVecs); CHKERRQ(ierr);
  TempScal.setZero(std::max(jmax,Qsize));
  // Check for options in MG preconditioner
  ierr = PetscOptionsGetInt(NULL, NULL, "-LOPGMRES_Jacobi_nSweep", &nsweep, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-LOPGMRES_Jacobi_Weight", &w, NULL); CHKERRQ(ierr);
  // Preallocate for operators
  Dlist.resize(levels-1);
  xlist.resize(levels);
  flist.resize(levels);
  QMatP.resize(levels-1);
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
/**             Computes the eigenmodes of the specified system              **/
/******************************************************************************/
PetscErrorCode LOPGMRES::Compute()
{
  PetscErrorCode ierr = 0;
  double compute_time = 0;
  if (this->verbose >= 1)
    compute_time = MPI_Wtime();
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Computing\n"); CHKERRQ(ierr);

  // Prep work
  ierr = PetscLogEventBegin(EIG_Initialize, 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = Create_Hierarchy(); CHKERRQ(ierr);
  ierr = Compute_Init(); CHKERRQ(ierr);

  // Initialize search subspace with interpolated coarse eigenvectors
  //PetscInt j = jmin;
  PetscInt j = jmin;
  nev_conv = 0;

  // Orthonormalize search space vectors
  ierr = MatMult(B[0], V[0], TempVecs[0]); CHKERRQ(ierr);
  ierr = VecDot(V[0], TempVecs[0], TempScal.data()); CHKERRQ(ierr);
  ierr = VecScale(V[0], 1.0/sqrt(TempScal(0))); CHKERRQ(ierr);
  for (int ii = 1; ii < j; ii++)
  {
    ierr = Icgsm(V, B[0], V[ii], TempScal(0), ii); CHKERRQ(ierr);
    ierr = VecScale(V[ii], 1.0/TempScal(0)); CHKERRQ(ierr);
  }

  // Construct initial search subspace
  MatrixPS G = MatrixPS::Zero(jmax,jmax);
  for (int ii = 0; ii < j; ii++)
  {
    ierr = MatMult(A[0], V[ii], TempVecs[ii]); CHKERRQ(ierr);
    ierr = VecMDot(TempVecs[ii], j, V, G.data() + jmax*ii); CHKERRQ(ierr);
  }

  // Construct eigensolver context for that subspace
  Eigen::SelfAdjointEigenSolver<MatrixPS> eps_sub(jmax);
  PetscScalar theta = 0; // Approximation of lambda

  // Things needed in the computation loop
  Vec residual;
  VecDuplicate(Q[0][0], &residual);
  PetscScalar rnorm = 0, rnorm_old = 0, Au_norm = 0, orth_norm = 0;
  MatrixPS W; ArrayPS S;
  PetscInt base_it = maxit; PetscScalar base_eps = eps;
  ierr = PetscLogEventEnd(EIG_Initialize, 0, 0, 0, 0); CHKERRQ(ierr);
  sigma = 0;

  // The actual computation loop
  it = 0;
  while (it++ < maxit)
  {
    eps_sub.compute(G.block(0,0,j,j));
    W = eps_sub.eigenvectors();
    S = eps_sub.eigenvalues();
    Sorteig(W, S);

    while (true)
    {
      ierr = PetscLogEventBegin(EIG_Prep, 0, 0, 0, 0); CHKERRQ(ierr);
      // Get new eigenvector approximation
      ierr = VecSet(Q[0][nev_conv], 0.0); CHKERRQ(ierr);
      ierr = VecMAXPY(Q[0][nev_conv], j, W.data(), V); CHKERRQ(ierr);
      theta = S(0);
      ierr = MatMult(A[0], Q[0][nev_conv], AQ[0][nev_conv]); CHKERRQ(ierr);
      ierr = MatMult(B[0], Q[0][nev_conv], BQ[0][nev_conv]); CHKERRQ(ierr);
      ierr = VecWAXPY(residual, -theta, BQ[0][nev_conv], AQ[0][nev_conv]); CHKERRQ(ierr);
      ierr = VecScale(residual, -1.0); CHKERRQ(ierr);

      ierr = VecNorm(residual, NORM_2, &rnorm); CHKERRQ(ierr);
      ierr = VecNorm(AQ[0][nev_conv], NORM_2, &Au_norm);

      if (this->verbose >= 2)
        PetscFPrintf(comm, output, "Iteration: %4i\tLambda Approx: %14.14g\tResidual: %4.4g\n", it, theta, rnorm);
      if (isnan(theta))
      {
	if (output > 2) // Optional printing of information of projected eigenproblem encounters NaN
        {
	  if (myid == 0)
	    cout << S << "\n";
	  PetscViewer view;
	  for (int ii = 0; ii < j; ii++)
	  {
	    char filename[20];
	    sprintf(filename, "V_%ii.bin",ii+1);
	    PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &view);
	    VecView(V[ii], view);
	    PetscViewerDestroy(&view);
	  }
	  PetscViewerBinaryOpen(comm, "A.bin", FILE_MODE_WRITE, &view);
	  MatView(A[0], view);
	  PetscViewerDestroy(&view);
	  PetscViewerBinaryOpen(comm, "B.bin", FILE_MODE_WRITE, &view);
	  MatView(B[0], view);
	  PetscViewerDestroy(&view);
	  MPI_Barrier(comm);
        }
        SETERRQ(comm, PETSC_ERR_FP, "Approximate eigenvalue is not a number");
      }

      ierr = PetscLogEventEnd(EIG_Prep, 0, 0, 0, 0); CHKERRQ(ierr);
      if ( ( (rnorm/abs(theta) >= eps) && (rnorm_old != rnorm) && (rnorm/Au_norm >= 1e-12) ) || (j <= 1) )
      {
        lambda(nev_conv) = theta;
        rnorm_old = rnorm;
        break;
      }
      ierr = PetscLogEventBegin(EIG_Convergence, 0, 0, 0, 0); CHKERRQ(ierr);
      // Convergence routine
      if (this->verbose >= 2)
        PetscFPrintf(comm, output, "Eigenvalue #%i converged with residual %1.4g after %i iterations\n", nev_conv+1, rnorm, it);
      tau_num = theta;
      lambda(nev_conv) = theta;
      for (int ii = 0; ii < j; ii++)
      {
        ierr = VecCopy(V[ii], TempVecs[ii]); CHKERRQ(ierr);
        ierr = VecSet(V[ii], 0.0); CHKERRQ(ierr);
      }

      // This line is to prevent an error when debugging but should be
      // unnecessary (and costly) if this is a production build
      #if defined(PETSC_USE_DEBUG)
      MPI_Allreduce(MPI_IN_PLACE, W.data(), j*j, MPI_DOUBLE, MPI_MAX, comm);
      #endif
      for (int ii = 1; ii < j; ii++)
      {
        ierr = VecMAXPY(V[ii-1], j, W.data()+ii*j, TempVecs); CHKERRQ(ierr);
      }

      for (int ii = 0; ii < j-1; ii++)
        S(ii) = S(ii+1);
      G.block(0, 0, j-1, j-1) = S.segment(0,j-1).matrix().asDiagonal();
      W = MatrixPS::Identity(j-1, j-1);
      j--; nev_conv++;

      ierr = PetscLogEventEnd(EIG_Convergence, 0, 0, 0, 0); CHKERRQ(ierr);
      if (Done())
      {
        // Cleanup
        ierr = VecDestroy(&residual); CHKERRQ(ierr);
        ierr = Compute_Clean(); CHKERRQ(ierr);
        if (this->verbose >= 1)
        {
          compute_time = MPI_Wtime() - compute_time;
          PetscFPrintf(comm, output, "LOPGMRES found all %i eigenvalues in %i iterations, taking %1.6g seconds\n", 
                      nev_conv, it, compute_time);
        }
        return 0;
      }
    }

    // Check for restart
    if (j == jmax)
    {
      j = jmin;
      for (int ii = 0; ii < jmax; ii++)
      {
        ierr = VecCopy(V[ii], TempVecs[ii]); CHKERRQ(ierr);
        ierr = VecSet(V[ii], 0.0); CHKERRQ(ierr);
      }
      for (int ii = 0; ii < j; ii++)
      {
        ierr = VecMAXPY(V[ii], jmax, W.data()+ii*jmax, TempVecs); CHKERRQ(ierr);
      }
      G.block(0, 0, j, j) = S.segment(0,j).matrix().asDiagonal();
    }

    // Shift parameter
    ierr = PetscLogEventBegin(EIG_Update, 0, 0, 0, 0); CHKERRQ(ierr);
    vicinity = rnorm/theta < epstr; sigma_old = sigma;
    if (vicinity || tau != NUMERIC)
      sigma = lambda(nev_conv);
    else
      sigma = tau_num;

    // Call the multigrid solver to solve correction equation
    if (cycle == VCycle)
    {
      ierr = MGSolve(V[j], residual); CHKERRQ(ierr);
    }
    else if (cycle == FMGCycle)
    {
      ierr = FullMGSolve(V[j], residual); CHKERRQ(ierr);
    }
    else
      SETERRQ(comm, PETSC_ERR_SUP, "Invalid Multigrid cycle chosen");
    ierr = PetscLogEventEnd(EIG_Update, 0, 0, 0, 0); CHKERRQ(ierr);

    ierr = PetscLogEventBegin(EIG_Expand, 0, 0, 0, 0); CHKERRQ(ierr);
    // Ensure orthogonality
    ierr = Mgsm(Q[0], BQ[0], V[j], nev_conv+1); CHKERRQ(ierr);
    ierr = Icgsm(V, B[0], V[j], orth_norm, j); CHKERRQ(ierr);
    ierr = VecScale(V[j], 1/orth_norm); CHKERRQ(ierr);

    // Update search space
    ierr = MatMult(A[0], V[j], TempVecs[0]); CHKERRQ(ierr);
    ierr = VecMDot(TempVecs[0], j+1, V, G.data()+j*jmax); CHKERRQ(ierr);
    G.block(j, 0, 1, j) = G.block(0, j, j, 1).transpose();
    ierr = PetscLogEventEnd(EIG_Expand, 0, 0, 0, 0); CHKERRQ(ierr);

    j++;
    if (it == maxit && eps/base_eps < 1000)
    {
      if (this->verbose >= 1)
        PetscFPrintf(comm, output, "Only %i converged eigenvalues in %i iterations, increasing tolerance to %1.2g\n",
                     nev_conv, maxit, eps*=10);
      maxit += base_it;
    }
  }
  // Cleanup
  if (this->verbose >= 1)
  {
    compute_time = MPI_Wtime() - compute_time;
    PetscFPrintf(comm, output, "LOPGMRES only found %i eigenvalues in %i iterations, taking %1.6g seconds\n",
                nev_conv, it, compute_time);
    PetscFPrintf(comm, output, "\tThe final residual value was %1.6g, and the approximate eigenvalue was %1.6g\n", rnorm, theta);
  }
  ierr = VecDestroy(&residual); CHKERRQ(ierr);
  ierr = Compute_Clean(); CHKERRQ(ierr);
  it = 0;

  return 0;
}

/******************************************************************************/
/**                   Check if all eigenvalues have been found               **/
/******************************************************************************/
bool LOPGMRES::Done()
{
  if ( (nev_conv == 1) && (nev_type != TOTAL_NEV) )
    return false;
  ArrayPS lambda_sort = lambda.segment(0,nev_conv);
  MatrixPS empty(0,0);
  Sorteig(empty, lambda_sort);
  switch (nev_type)
  {
    case TOTAL_NEV:
      return (nev_conv == nev_req);
    case UNIQUE_NEV:
      return (((1.0 - lambda_sort.segment(1,nev_conv-1).array() /
                      lambda_sort.segment(0,nev_conv-1).array()).abs() 
              > 1e-5).cast<int>().sum() == nev_req);
    case UNIQUE_LAST_NEV:
      return ( (nev_conv > nev_req) && (abs(1.0 - 
                lambda_sort(nev_conv-2) / lambda_sort(nev_conv-1)) > 1e-5) );
  }
  return false;
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
/**                  Sort the eigenvalues of the subspace                    **/
/******************************************************************************/
Eigen::ArrayXi LOPGMRES::Sorteig(MatrixPS &W, ArrayPS &S)
{
  MatrixPS Wcopy = W;
  ArrayPS Scopy = S;
  Eigen::ArrayXi ind;
  switch (tau)
  {
    case NUMERIC:
      S = S-tau_num;
      ind = EigLab::gensort(S);
      break;
    case LM:
      S = S.cwiseAbs();
      ind = EigLab::gensort(S).reverse();
      break;
    case SM:
      S = S.cwiseAbs();
      ind = EigLab::gensort(S);
      break;
    case LR: case LA:
      ind = EigLab::gensort(S).reverse();
      break;
    case SR: case SA:
      ind = EigLab::gensort(S);
      break;
  }

  for (int ii = 0; ii < ind.size(); ii++)
  {
    S(ii) = Scopy(ind(ii));
    if (W.size() > 0)
      W.col(ii) = Wcopy.col(ind(ii));
  }

  return ind;
}

/******************************************************************************/
/**          Iterative classical M-orthogonal Gram-Schmidt                   **/
/******************************************************************************/
PetscErrorCode LOPGMRES::Icgsm(Vec *Q, Mat M, Vec u, PetscScalar &r, PetscInt k)
{
  PetscErrorCode ierr = 0;
  double alpha = 0.5;
  int itmax = 3, it = 1;
  Vec um;
  ierr = VecDuplicate(u, &um); CHKERRQ(ierr);
  ierr = MatMult(M, u, um); CHKERRQ(ierr);
  PetscScalar r0;
  ierr = VecDot(u, um, &r0); CHKERRQ(ierr);
  r0 = sqrt(r0);
  while (true)
  {
    ierr = VecMDot(um, k, Q, TempScal.data()); CHKERRQ(ierr);
    TempScal *= -1;
    ierr = VecMAXPY(u, k, TempScal.data(), Q); CHKERRQ(ierr);
    //ierr = GS(Q, um, u, k); CHKERRQ(ierr);
    ierr = MatMult(M, u, um); CHKERRQ(ierr);
    ierr = VecDot(u, um, &r); CHKERRQ(ierr);
    r = sqrt(r);
    if (r > alpha*r0 || it > itmax)
      break;
    it++; r0 = r;
  }
  if ( (r <= alpha*r0) && (myid == 0) )
    cout << "Warning, loss of orthogonality experienced in ICGSM.\n";

  ierr = VecDestroy(&um); CHKERRQ(ierr);

  return 0;
}

/******************************************************************************/
/**                  Modified M-orthogonal Gram-Schmidt                      **/
/******************************************************************************/
PetscErrorCode LOPGMRES::Mgsm(Vec* Q, Vec* BQ, Vec u, PetscInt k)
{
  PetscErrorCode ierr = 0;

  for (int ii = 0; ii < k; ii++)
  {
    ierr = VecDot(u, BQ[ii], TempScal.data()); CHKERRQ(ierr);
    ierr = VecAXPY(u, -TempScal(0), Q[ii]); CHKERRQ(ierr);
  }

  return 0;
}

/******************************************************************************/
/**         Standard M-orthogonal Gram-Schmidt orthogonalization step        **/
/******************************************************************************/
PetscErrorCode LOPGMRES::GS(Vec* Q, Vec Mu, Vec u, PetscInt k)
{
  // So far this function only hurts performance and is not yet usable
  PetscErrorCode ierr = 0;

  PetscScalar *p_Mu, *p_u, **p_Q;
  int one = 1, bn;
  ierr = VecGetLocalSize(u, &bn); CHKERRQ(ierr);
  MPI_Request *request = new MPI_Request[k];
  int *flag = new int[k];
  int allflag = 0;

  // Serial Dot products on each process, share partial result with Iallreduce
  ierr = VecGetArray(Mu, &p_Mu); CHKERRQ(ierr);
  ierr = VecGetArrays(Q, k, &p_Q); CHKERRQ(ierr);
  for (int ii = 0; ii < k; ii++)
  {
    TempScal(ii) = -ddot_(&bn, p_Mu, &one, p_Q[ii], &one);
    //MPI_Iallreduce(MPI_IN_PLACE, TempScal.data()+ii, 1, MPI_DOUBLE, MPI_SUM,
    //               comm, request+ii);
  }
  ierr = VecRestoreArray(Mu, &p_Mu); CHKERRQ(ierr);

  // Remove projections of each vector once allReduce finishes
  ierr = VecGetArray(u, &p_u); CHKERRQ(ierr);
  fill(flag, flag+k, 0);
  while (allflag < k)
  {
    for (int ii = 0; ii < k; ii++)
    {
      if (flag[ii])
        continue;
      MPI_Test(request+ii, flag+ii, MPI_STATUS_IGNORE);
      if (flag[ii])
      {
        daxpy_(&bn, TempScal.data()+ii, p_Q[ii], &one, p_u, &one);
      }
    }
    allflag = accumulate(flag, flag+k, 0);
  }

  ierr = VecRestoreArrays(Q, k, &p_Q); CHKERRQ(ierr);
  ierr = VecRestoreArray(u, &p_u); CHKERRQ(ierr);
  delete[] request;
  delete[] flag;

  return 0;
}

/******************************************************************************/
/**               Apply full multigrid for correction equation               **/
/******************************************************************************/
PetscErrorCode LOPGMRES::FullMGSolve(Vec x, Vec f)
{
  PetscErrorCode ierr = 0;
  ierr = PetscLogEventBegin(EIG_Precondition, 0, 0, 0, 0); CHKERRQ(ierr);
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Applying multigrid\n"); CHKERRQ(ierr);

  // Preallocate f, x, and D at each level (D not needed on finest level)
  ArrayPS QMatQ = lambda.segment(0,nev_conv+1) - sigma;
  xlist[0] = x;
  flist[0] = f;

  // Project f onto lowest level
  for (int ii = 0; ii < levels-1; ii++)
  {
    ierr = MatMultTranspose(P[ii], flist[ii], flist[ii+1]); CHKERRQ(ierr);
  }

  // FMG cycle
  for (int ii = levels-1; ii >= 0; ii--)
  {
    // Downcycling
    for (int jj = ii; jj < levels; jj++)
    {
      if (jj == levels-1)
      {
        // Coarse solve
        ierr = KSPSolve(ksp_coarse, flist.back(), xlist.back()); CHKERRQ(ierr);
      }
      else
      {
        ierr = WJac(QMatP[jj], QMatQ, Dlist[jj], flist[jj], xlist[jj], jj); CHKERRQ(ierr);
        ierr = ApplyOP(xlist[jj], OPx[jj], jj); CHKERRQ(ierr);
        ierr = VecAYPX(OPx[jj], -1.0, flist[jj]); CHKERRQ(ierr);
        ierr = MatMultTranspose(P[jj], OPx[jj], flist[jj+1]); CHKERRQ(ierr);
        ierr = VecSet(xlist[jj+1], 0.0); CHKERRQ(ierr);
      }
    }
    //Upcycling
    for (int jj = levels-2; jj >= ii; jj--)
    {
      ierr = MatMultAdd(P[jj], xlist[jj+1], xlist[jj], xlist[jj]); CHKERRQ(ierr);
      ierr = WJac(QMatP[jj], QMatQ, Dlist[jj], flist[jj], xlist[jj], jj); CHKERRQ(ierr);
    }
    if (ii > 0)
    {
      ierr = MatMult(P[ii-1], xlist[ii], xlist[ii-1]); CHKERRQ(ierr);
    }
  }

  ierr = PetscLogEventEnd(EIG_Precondition, 0, 0, 0, 0); CHKERRQ(ierr);
  return 0;
}     

/******************************************************************************/
/**                 Apply multigrid for correction equation                  **/
/******************************************************************************/
PetscErrorCode LOPGMRES::MGSolve(Vec x, Vec f)
{
  PetscErrorCode ierr = 0;
  ierr = PetscLogEventBegin(EIG_Precondition, 0, 0, 0, 0); CHKERRQ(ierr);
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Applying multigrid\n"); CHKERRQ(ierr);

  // Preallocate f, x, and D at each level (D not needed on finest level)
  ArrayPS QMatQ = lambda.segment(0,nev_conv+1) - sigma;
  xlist[0] = x;
  flist[0] = f;

  // Downcycle
  for (int ii = 0; ii < levels-1; ii++)
  {
    ierr = VecSet(xlist[ii], 0.0); CHKERRQ(ierr);
    ierr = WJac(QMatP[ii], QMatQ, Dlist[ii], flist[ii], xlist[ii], ii); CHKERRQ(ierr);
    ierr = ApplyOP(xlist[ii], OPx[ii], ii); CHKERRQ(ierr);
    ierr = VecAYPX(OPx[ii], -1.0, flist[ii]); CHKERRQ(ierr);
    ierr = MatMultTranspose(P[ii], OPx[ii], flist[ii+1]); CHKERRQ(ierr);
  }

  // Coarse solve
  ierr = KSPSolve(ksp_coarse, flist.back(), xlist.back()); CHKERRQ(ierr);

  // Upcycle
  for (int ii = levels-2; ii >= 0; ii--)
  {
    ierr = MatMultAdd(P[ii], xlist[ii+1], xlist[ii], xlist[ii]); CHKERRQ(ierr);
    ierr = WJac(QMatP[ii], QMatQ, Dlist[ii], flist[ii], xlist[ii], ii); CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(EIG_Precondition, 0, 0, 0, 0); CHKERRQ(ierr);
  return 0;
}

/******************************************************************************/
/**                        Weighted Jacobi smoother                          **/
/******************************************************************************/
PetscErrorCode LOPGMRES::WJac(Vec* QMatP, ArrayPS &QMatQ, Vec D, Vec y, Vec x, PetscInt level)
{
  // The y being fed in is -r, as it should be
  PetscErrorCode ierr = 0;
  ierr = PetscLogEventBegin(EIG_Jacobi, 0, 0, 0, 0); CHKERRQ(ierr);
  if (nsweep == 0)
    return 0;

  Vec r;
  ierr = VecDuplicate(y, &r); CHKERRQ(ierr);
  for (int ii = 0; ii < nsweep; ii++)
  {
    ierr = ApplyOP(x, r, level); CHKERRQ(ierr);
    ierr = VecAYPX(r, -1.0, y); CHKERRQ(ierr);
    ierr = VecPointwiseDivide(r, r, D); CHKERRQ(ierr);
    ierr = VecAXPY(x, w, r); CHKERRQ(ierr);
  }
  VecDestroy(&r);
  ierr = PetscLogEventEnd(EIG_Jacobi, 0, 0, 0, 0); CHKERRQ(ierr);

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
