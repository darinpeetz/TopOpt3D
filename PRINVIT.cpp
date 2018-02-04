#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include "PRINVIT.h"
#include "EigLab.h"

using namespace std;

PetscLogEvent EIG_Compute;
PetscLogEvent EIG_Initialize, EIG_Prep, EIG_Convergence, EIG_Expand, EIG_Update;
PetscLogEvent EIG_Comp_Init, EIG_Hierarchy, EIG_Precondition, EIG_Jacobi, *EIG_ApplyOP;

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
PRINVIT::PRINVIT()
{
  levels = 0;
  Qsize = nev_req;
  jmin = -1; jmax = -1;
  nsweep = 5;
  w = 4.0/7;
  PetscOptionsGetInt(NULL, NULL, "-PRINVIT_Verbose", &verbose, NULL);
  cycle = FMGCycle;
}

/******************************************************************************/
/**                              Main destructor                             **/
/******************************************************************************/
PRINVIT::~PRINVIT()
{
  for (unsigned int ii = 0; ii < P.size(); ii++)
    MatDestroy(P.data()+ii);
}

/******************************************************************************/
/**                       How much information to print                      **/
/******************************************************************************/
PetscErrorCode PRINVIT::Set_Verbose(PetscInt verbose)
{
  this->verbose = verbose;
  Close_File();
  PetscErrorCode ierr = PetscOptionsGetInt(NULL, NULL, "-PRINVIT_Verbose", &this->verbose, NULL);
  CHKERRQ(ierr);
  return 0;
}

/******************************************************************************/
/**                     Extract hierarchy from PCMG object                   **/
/******************************************************************************/
PetscErrorCode PRINVIT::PCMG_Extract(PC pcmg, bool isB, bool isA)
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
PetscErrorCode PRINVIT::Set_Hierarchy(const std::vector<Mat> P, const std::vector<MPI_Comm> MG_comms)
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
/**             Computes the eigenmodes of the specified system              **/
/******************************************************************************/
PetscErrorCode PRINVIT::Compute()
{
  PetscErrorCode ierr = 0;
  ierr = PetscLogEventBegin(EIG_Compute, 0, 0, 0, 0); CHKERRQ(ierr);
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Computing\n"); CHKERRQ(ierr);

  // Prep work
  ierr = PetscLogEventBegin(EIG_Initialize, 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = Create_Hierarchy(); CHKERRQ(ierr);
  ierr = Compute_Init(); CHKERRQ(ierr);

  // Initialize search subspace with interpolated coarse eigenvectors
  PetscInt j = 0;
  ierr = Initialize_V(j); CHKERRQ(ierr);
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

      ierr = Update_Preconditioner(residual, rnorm, Au_norm); CHKERRQ(ierr);

      if (this->verbose >= 2){
        ierr = Print_Status(rnorm); CHKERRQ(ierr);
      }
      if (isnan(theta))
      {
	if (false) // Optional printing of information of projected eigenproblem encounters NaN
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
          ierr = Print_Result(); CHKERRQ(ierr);
        }
        ierr = PetscLogEventEnd(EIG_Compute, 0, 0, 0, 0); CHKERRQ(ierr);
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

    // Call the multigrid solver to solve correction equation
    ierr = MGSetup(residual, rnorm); CHKERRQ(ierr);
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
    // This loop is to prevent NaN breakdown
    while (true)
    {
      // Ensure orthogonality
      ierr = Mgsm(Q[0], BQ[0], V[j], nev_conv+1); CHKERRQ(ierr);
      ierr = Icgsm(V, B[0], V[j], orth_norm, j); CHKERRQ(ierr);
      ierr = VecScale(V[j], 1/orth_norm); CHKERRQ(ierr);

      // Update search space
      ierr = MatMult(A[0], V[j], TempVecs[0]); CHKERRQ(ierr);
      ierr = VecMDot(TempVecs[0], j+1, V, G.data()+j*jmax); CHKERRQ(ierr);
      G.block(j, 0, 1, j) = G.block(0, j, j, 1).transpose();
      ierr = PetscLogEventEnd(EIG_Expand, 0, 0, 0, 0); CHKERRQ(ierr);
      
      if (isnan(G(j,j)))
      {
        ierr = VecSetRandom(V[j], NULL); CHKERRQ(ierr);
      }
      else
        break;
    }

    j++;
    if (it == maxit && eps/base_eps < 1000)
    {
      if (this->verbose >= 1)
        PetscFPrintf(comm, output, "Only %i converged eigenvalues in %i iterations, increasing tolerance to %1.2g\n", nev_conv, maxit, eps*=10);
      maxit += base_it;
    }
  }
  // Cleanup
  if (this->verbose >= 1)
  {
    ierr = Print_Result(); CHKERRQ(ierr);
  }
  ierr = VecDestroy(&residual); CHKERRQ(ierr);
  ierr = Compute_Clean(); CHKERRQ(ierr);
  it = 0;

  ierr = PetscLogEventEnd(EIG_Compute, 0, 0, 0, 0); CHKERRQ(ierr);
  return 0;
}

/******************************************************************************/
/**               Apply full multigrid for correction equation               **/
/******************************************************************************/
PetscErrorCode PRINVIT::FullMGSolve(Vec x, Vec f)
{
  PetscErrorCode ierr = 0;
  ierr = PetscLogEventBegin(EIG_Precondition, 0, 0, 0, 0); CHKERRQ(ierr);
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Applying multigrid\n"); CHKERRQ(ierr);

  // Set x and f at the top of the hierarchy
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
        ierr = Coarse_Solve(); CHKERRQ(ierr);
      }
      else
      {
        ierr = WJac(flist[jj], xlist[jj], jj); CHKERRQ(ierr);
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
      ierr = WJac(flist[jj], xlist[jj], jj); CHKERRQ(ierr);
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
PetscErrorCode PRINVIT::MGSolve(Vec x, Vec f)
{
  PetscErrorCode ierr = 0;
  ierr = PetscLogEventBegin(EIG_Precondition, 0, 0, 0, 0); CHKERRQ(ierr);
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Applying multigrid\n"); CHKERRQ(ierr);

  // Set x and f at the top of the hierarchy
  xlist[0] = x;
  flist[0] = f;

  // Downcycle
  for (int ii = 0; ii < levels-1; ii++)
  {
    ierr = VecSet(xlist[ii], 0.0); CHKERRQ(ierr);
    ierr = WJac(flist[ii], xlist[ii], ii); CHKERRQ(ierr);
    ierr = ApplyOP(xlist[ii], OPx[ii], ii); CHKERRQ(ierr);
    ierr = VecAYPX(OPx[ii], -1.0, flist[ii]); CHKERRQ(ierr);
    ierr = MatMultTranspose(P[ii], OPx[ii], flist[ii+1]); CHKERRQ(ierr);
  }

  // Coarse solve
  ierr = Coarse_Solve(); CHKERRQ(ierr);

  // Upcycle
  for (int ii = levels-2; ii >= 0; ii--)
  {
    ierr = MatMultAdd(P[ii], xlist[ii+1], xlist[ii], xlist[ii]); CHKERRQ(ierr);
    if (ii == 0)
    {
      ierr = VecMDot(xlist[ii], nev_conv+1, BQ[ii], TempScal.data()); CHKERRQ(ierr);
      TempScal *= -1;
      ierr = VecMAXPY(xlist[ii], nev_conv+1, TempScal.data(), Q[ii]); CHKERRQ(ierr);
    }
    ierr = WJac(flist[ii], xlist[ii], ii); CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(EIG_Precondition, 0, 0, 0, 0); CHKERRQ(ierr);
  return 0;
}

/******************************************************************************/
/**                        Weighted Jacobi smoother                          **/
/******************************************************************************/
PetscErrorCode PRINVIT::WJac(Vec y, Vec x, PetscInt level)
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
    ierr = VecPointwiseDivide(r, r, Dlist[level]); CHKERRQ(ierr);
    ierr = VecAXPY(x, w, r); CHKERRQ(ierr);
  }
  VecDestroy(&r);
  ierr = PetscLogEventEnd(EIG_Jacobi, 0, 0, 0, 0); CHKERRQ(ierr);

  return 0;
}
