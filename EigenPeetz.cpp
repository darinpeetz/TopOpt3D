#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include "EigenPeetz.h"
#include "EigLab.h"

using namespace std;

typedef Eigen::Matrix<PetscScalar, -1, -1> MatrixPS;
typedef Eigen::Array<PetscScalar, -1, 1>  ArrayPS;

PetscLogEvent JDMG_Initialize, JDMG_Prep, JDMG_Convergence, JDMG_Expand, JDMG_Update;
PetscLogEvent JDMG_Comp_Init, JDMG_Hierarchy, JDMG_Setup_Coarse, JDMG_Comp_Coarse;
PetscLogEvent JDMG_MGSetUp, JDMG_Precondition, JDMG_Jacobi, *JDMG_ApplyOP;
PetscLogEvent *JDMG_ApplyOP1, *JDMG_ApplyOP2, *JDMG_ApplyOP3, *JDMG_ApplyOP4;

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
JDMG::JDMG(MPI_Comm comm)
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
  PetscOptionsGetInt(NULL, NULL, "-JDMG_Verbose", &verbose, NULL);
  PetscFOpen(this->comm, "stdout", "w", &output);
  file_opened = 1;
  cycle = FMGCycle;
}

/******************************************************************************/
/**                              Main destructor                             **/
/******************************************************************************/
JDMG::~JDMG()
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
PetscErrorCode JDMG::Set_Verbose(PetscInt verbose)
{
  this->verbose = verbose;
  Close_File();
  PetscErrorCode ierr = PetscOptionsGetInt(NULL, NULL, "-JDMG_Verbose", &this->verbose, NULL);
  CHKERRQ(ierr);
  return 0;
}

/******************************************************************************/
/**              Designating an already opened file for output               **/
/******************************************************************************/
PetscErrorCode JDMG::Set_File(FILE *output)
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
PetscErrorCode JDMG::Open_File(const char filename[])
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
PetscErrorCode JDMG::Set_Operators(Mat A, Mat B)
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
PetscErrorCode JDMG::PCMG_Extract(PC pcmg, bool isB, bool isA)
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
PetscErrorCode JDMG::Set_Hierarchy(const std::vector<Mat> P, const std::vector<MPI_Comm> MG_comms)
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
void JDMG::Set_Target(Tau tau, PetscInt nev, Nev_Type ntype)
{
  if (this->verbose >= 3)
    PetscFPrintf(comm, output, "Setting target eigenvalues\n");
  this->tau = tau;
  this->nev_req = nev;
  this->nev_type = ntype;
  this->Qsize = nev_req + (this->nev_type == TOTAL_NEV ? 0 : 6);

  return;
}
void JDMG::Set_Target(PetscScalar tau, PetscInt nev, Nev_Type ntype)
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
PetscErrorCode JDMG::Create_Hierarchy()
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
  {
    ierr = PetscFPrintf(comm, output, "Creating operators at each level\n"); CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(JDMG_Hierarchy, 0, 0, 0, 0); CHKERRQ(ierr);
  for (unsigned int i = 0; i < P.size(); i++)
  {
    if (this->A[i+1] == NULL)
    {
      ierr = MatPtAP(A[i], P[i], MAT_INITIAL_MATRIX, 1.0, A.data()+i+1); CHKERRQ(ierr);
    }
    if (this->B[i+1] == NULL)
    {
      ierr = MatPtAP(B[i], P[i], MAT_INITIAL_MATRIX, 1.0, B.data()+i+1); CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(JDMG_Hierarchy, 0, 0, 0, 0); CHKERRQ(ierr);

  return 0;
}

/******************************************************************************/
/**        Preps for computing the eigenmodes of the specified system        **/
/******************************************************************************/
PetscErrorCode JDMG::Compute_Init()
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Initializing compute structures\n"); CHKERRQ(ierr);

  ierr = PetscLogEventBegin(JDMG_Comp_Init, 0, 0, 0, 0); CHKERRQ(ierr);
  // Clear any old eigenvectors
  if (nev_conv > 0)
    ierr = VecDestroyVecs(nev_conv, &phi); CHKERRQ(ierr);

  // Check if MG cycle type was set at command line
  const PetscInt ct_length = 5;
  char cycle_type[ct_length];
  PetscBool cycle_type_set = PETSC_FALSE;
  ierr = PetscOptionsGetString(NULL, NULL, "-JDMG_Cycle_Type", cycle_type,
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
      PetscPrintf(comm, "Bad JDMG_Cycle_Type given %s, should be \"FULL\" or \"V\"", cycle_type);
  }

  // Preallocate eigenvalues and eigenvectors at each level and get problem size
  Vec temp;
  Q.resize(levels); AQ.resize(levels); BQ.resize(levels);
  for (int ii = 0; ii < levels; ii++)
  {
    ierr = MatCreateVecs(A[ii], &temp, NULL); CHKERRQ(ierr);
    ierr = VecDuplicateVecs(temp, Qsize, Q.data()+ii); CHKERRQ(ierr);
    ierr = VecDuplicateVecs(temp, Qsize, AQ.data()+ii); CHKERRQ(ierr);
    ierr = VecDuplicateVecs(temp, Qsize, BQ.data()+ii); CHKERRQ(ierr);
    ierr = VecDestroy(&temp); CHKERRQ(ierr);
  }
  lambda.setOnes(Qsize);
  ierr = VecGetSize(Q[0][0], &n); CHKERRQ(ierr);
  ierr = VecGetLocalSize(Q[0][0], &nlocal); CHKERRQ(ierr);

  // Determine size of search space
  ierr = PetscOptionsGetInt(NULL, NULL, "-JDMG_jmin", &jmin, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-JDMG_jmax", &jmax, NULL); CHKERRQ(ierr);
  if (jmin < 0)
    jmin = std::min( std::max(2*nev_req, 10), std::min(n/2,10));
  if (jmax < 0)
    jmax = std::min( std::max(4*nev_req, 25), std::min(n , 50));

  // Preallocate search space, work space, and eigenvectors
  ierr = VecDuplicateVecs(Q[0][0], jmax, &V); CHKERRQ(ierr);
  PetscRandom random;
  ierr = PetscRandomCreate(comm, &random); CHKERRQ(ierr);
  for (int ii = 0; ii < jmin; ii++)
  {
    ierr = VecSetRandom(V[ii], random); CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&random);
  ierr = VecDuplicateVecs(Q[0][0], jmax, &TempVecs); CHKERRQ(ierr);
  TempScal.setZero(std::max(jmax,Qsize));
  // Check for options in MG preconditioner
  ierr = PetscOptionsGetInt(NULL, NULL, "-JDMG_Jacobi_nSweep", &nsweep, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-JDMG_Jacobi_Weight", &w, NULL); CHKERRQ(ierr);
  // Preallocate for operators
  AmsB.resize(levels); Acopy.resize(levels-1); Bcopy.resize(levels-1);
  Dlist.resize(levels-1);
  xlist.resize(levels);
  flist.resize(levels);
  QMatP.resize(levels-1);
  OPx.resize(levels-1);
  ierr = Setup_Coarse(); CHKERRQ(ierr);
  for (int ii = 0; ii < levels-1; ii++)
  {
    // Combined matrices at each level
    ierr = MatDuplicate(B[ii], MAT_DO_NOT_COPY_VALUES, AmsB.data()+ii); CHKERRQ(ierr);
    ierr = MatZeroEntries(AmsB[ii]); CHKERRQ(ierr);
    ierr = MatAXPY(AmsB[ii], 1.0, A[ii], DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    /*ierr = MatDuplicate(AmsB[ii], MAT_SHARE_NONZERO_PATTERN, Acopy.data()+ii); CHKERRQ(ierr);
    ierr = MatZeroEntries(Acopy[ii]); CHKERRQ(ierr);
    ierr = MatCopy(A[ii], Acopy[ii], DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);*/
    ierr = MatDuplicate(AmsB[ii], MAT_SHARE_NONZERO_PATTERN, Bcopy.data()+ii); CHKERRQ(ierr);
    ierr = MatZeroEntries(Bcopy[ii]); CHKERRQ(ierr);
    ierr = MatCopy(B[ii], Bcopy[ii], DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    // Vectors
    ierr = MatCreateVecs(A[ii+1], xlist.data()+ii+1, flist.data()+ii+1); CHKERRQ(ierr);
    ierr = MatCreateVecs(A[ii], Dlist.data()+ii, OPx.data()+ii); CHKERRQ(ierr);
    ierr = VecDuplicateVecs(Q[ii][0], Qsize, QMatP.data()+ii); CHKERRQ(ierr);
  }

  // Prep coarse problem
  ierr = KSPCreate(comm, &ksp_coarse); CHKERRQ(ierr);
  ierr = KSPSetType(ksp_coarse, KSPPREONLY); CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp_coarse, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1); CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp_coarse, "JDMG_coarse_"); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp_coarse); CHKERRQ(ierr);
  PC pc;
  ierr = KSPGetPC(ksp_coarse, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, PCBJACOBI); CHKERRQ(ierr);
  ierr = PCSetOptionsPrefix(pc, "JDMG_coarse_"); CHKERRQ(ierr);
  ierr = PCSetFromOptions(pc); CHKERRQ(ierr);

  ierr = KSPSetOperators(ksp_coarse, AmsB.back(), AmsB.back()); CHKERRQ(ierr);
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
  ierr = PetscLogEventEnd(JDMG_Comp_Init, 0, 0, 0, 0); CHKERRQ(ierr);

  return 0;
}

/******************************************************************************/
/**             Setup the Mat and KSP objects for coarse solve               **/
/******************************************************************************/
PetscErrorCode JDMG::Setup_Coarse()
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Setting up coarse operator\n"); CHKERRQ(ierr);

  ierr = PetscLogEventBegin(JDMG_Setup_Coarse, 0, 0, 0, 0); CHKERRQ(ierr);
  // Figure out where to put the new rows of the marix (should be on the last
  // process that has any values)
  PetscInt endrank = 0, rows, cols, lrows, lcols;
  ierr = MatGetSize(A.back(), &rows, &cols); CHKERRQ(ierr);
  ierr = MatGetLocalSize(A.back(), &lrows, &lcols); CHKERRQ(ierr);
  ncoarse = rows;
  nlcoarse = lrows; // Coarse problem size
  if (lrows > 0)
    endrank = myid;
  MPI_Allreduce(MPI_IN_PLACE, &endrank, 1, MPI_PETSCINT, MPI_MAX, comm);

  if (myid == endrank)
  {
    lrows += Qsize;
    lcols += Qsize;
  }
  // Initialize the matrix
  ierr = MatCreate(comm, &AmsB.back()); CHKERRQ(ierr);
  ierr = MatSetSizes(AmsB.back(), lrows, lcols, rows+Qsize, cols+Qsize); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(AmsB.back(), "JDMG_AmsB_"); CHKERRQ(ierr);
  ierr = MatSetFromOptions(AmsB.back()); CHKERRQ(ierr);

  // Preallocation
  PetscInt rstart = 0, rend = 0, nz;
  std::vector<PetscInt> columns;
  std::vector<PetscInt>::iterator it;
  const PetscInt *cwork;
  const PetscScalar *vwork;
  ierr = MatGetOwnershipRange(A.back(), &rstart, &rend); CHKERRQ(ierr);
  PetscInt *onDiag  = new PetscInt[lrows];
  PetscInt *offDiag = new PetscInt[lrows];
  for (int ii = rstart; ii < rend; ii++)
  {
    // Get all columns in this row
    columns.resize(0);
    ierr = MatGetRow(A.back(), ii, &nz, &cwork, NULL); CHKERRQ(ierr);
    columns.insert(columns.end(), cwork, cwork+nz);
    ierr = MatRestoreRow(A.back(), ii, &nz, &cwork, NULL); CHKERRQ(ierr);
    ierr = MatGetRow(B.back(), ii, &nz, &cwork, NULL); CHKERRQ(ierr);
    columns.insert(columns.end(), cwork, cwork+nz);
    ierr = MatRestoreRow(B.back(), ii, &nz, &cwork, NULL); CHKERRQ(ierr);
    // Get unique columns in this row
    sort(columns.begin(), columns.end());
    it = unique(columns.begin(), columns.end());
    columns.resize(distance(columns.begin(), it));
    onDiag[ii]  = (myid==endrank)*Qsize;
    offDiag[ii] = (myid!=endrank)*Qsize;
    for (it = columns.begin(); it != columns.end(); it++)
    {
      if (*it >= rstart && *it < rend)
        onDiag[ii]++;
      else
        offDiag[ii]++;
    }
  }
  if (myid == endrank)
  {
    fill(onDiag+nlcoarse, onDiag+lrows, nlcoarse+1);
    fill(offDiag+nlcoarse, offDiag+lrows, nlcoarse);
  }
  MatXAIJSetPreallocation(AmsB.back(), 1, onDiag, offDiag, NULL, NULL); CHKERRQ(ierr);
  delete[] onDiag; delete[] offDiag;

  // Add "Q" rows/columns
  if (myid == endrank)
  {
    ArrayXPI index1 = ArrayXPI::LinSpaced(rows, 0, rows-1);
    ArrayXPI index2 = ArrayXPI::LinSpaced(Qsize, rows, rows+Qsize-1);
    MatrixPS values = MatrixPS::Zero(rows, Qsize);
    ierr = MatSetValues(AmsB.back(), rows, index1.data(), Qsize, index2.data(),
          values.data(), INSERT_VALUES);
    ierr = MatSetValues(AmsB.back(), Qsize, index2.data(), rows, index1.data(),
          values.data(), INSERT_VALUES);
    for (int ii = 0; ii < Qsize; ii++){
      ierr = MatSetValue(AmsB.back(), rows+ii, rows+ii, 1.0, INSERT_VALUES); CHKERRQ(ierr);}
  }
  // Add "A-sigma*B" chunk of matrix
  for (int ii = rstart; ii < rend; ii++)
  {
    ierr = MatGetRow(A.back(), ii, &nz, &cwork, &vwork); CHKERRQ(ierr);
    ierr = MatSetValues(AmsB.back(), 1, &ii, nz, cwork, vwork, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(A.back(), ii, &nz, &cwork, &vwork); CHKERRQ(ierr);
    // This is added because the submatrix method didn't work
    ierr = MatGetRow(B.back(), ii, &nz, &cwork, &vwork); CHKERRQ(ierr);
    ierr = MatSetValues(AmsB.back(), 1, &ii, nz, cwork, vwork, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(B.back(), ii, &nz, &cwork, &vwork); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(AmsB.back(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(AmsB.back(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  // Solution and rhs vectors at coarse level
  ierr = MatCreateVecs(AmsB.back(), &x_end, &f_end); CHKERRQ(ierr);
  ierr = VecSet(f_end, 0.0); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(JDMG_Setup_Coarse, 0, 0, 0, 0); CHKERRQ(ierr);

  return 0;
}

/******************************************************************************/
/**              Computes the eigenmodes of the coarse system                **/
/******************************************************************************/
PetscErrorCode JDMG::Compute_Coarse()
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Solving for eigenvalues at coarse level\n"); CHKERRQ(ierr);

  ierr = PetscLogEventBegin(JDMG_Comp_Coarse, 0, 0, 0, 0); CHKERRQ(ierr);
  // Initialize
  ierr = EPSCreate(comm, &eps_coarse); CHKERRQ(ierr);
  if (ncoarse < 500){
    ierr = EPSSetType(eps_coarse, EPSLAPACK); CHKERRQ(ierr);}
  else {
    ierr = EPSSetType(eps_coarse, EPSKRYLOVSCHUR); CHKERRQ(ierr);}
  ierr = EPSSetProblemType(eps_coarse, EPS_GHEP); CHKERRQ(ierr);
  if (tau == NUMERIC)
    ierr = EPSSetWhichEigenpairs(eps_coarse, EPS_TARGET_REAL);
  else if (tau == SM || tau == LM)
    ierr = EPSSetWhichEigenpairs(eps_coarse, EPS_LARGEST_MAGNITUDE);
  else if (tau == SR || tau == SA || tau == LR || tau == LA)
    ierr = EPSSetWhichEigenpairs(eps_coarse, EPS_LARGEST_REAL);
  CHKERRQ(ierr);

  // Prep
  if (tau == SM || tau == SR || tau == SA)
    ierr = EPSSetOperators(eps_coarse, B.back(), A.back());
  else
    ierr = EPSSetOperators(eps_coarse, A.back(), B.back());
  CHKERRQ(ierr);
  ierr = EPSSetDimensions(eps_coarse, nev_req, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = EPSSetTolerances(eps_coarse, 1e-6, PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = EPSSetOptionsPrefix(eps_coarse, "coarse_"); CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps_coarse); CHKERRQ(ierr);

  // Solve
  ierr = EPSSolve(eps_coarse); CHKERRQ(ierr); 
  nev_conv = 0;
  ierr = EPSGetConverged(eps_coarse, &nev_conv);
  nev_conv = std::min(nev_conv,nev_req);
  if (nev_conv == 0)
    SETERRQ(comm, PETSC_ERR_NOT_CONVERGED, "Coarse solver found no eigenvalues\n");

  EPSType ceps_type;
  EPSGetType(eps_coarse, &ceps_type);

  /// Pull out the converged eigenvectors
  for (short ii = 0; ii < nev_conv; ii++)
  {
    ierr = EPSGetEigenvector(eps_coarse, ii, Q.back()[ii], 0); CHKERRQ(ierr);
  }

  for (short ii = levels-2; ii >=0; ii--)
  {
    for (short jj = 0; jj < nev_conv; jj++)
      ierr = MatMult(P[ii], Q[ii+1][jj], Q[ii][jj]); CHKERRQ(ierr);
  }
  ierr = EPSDestroy(&eps_coarse); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(JDMG_Comp_Coarse, 0, 0, 0, 0); CHKERRQ(ierr);

  return 0;
}
/******************************************************************************/
/**             Computes the eigenmodes of the specified system              **/
/******************************************************************************/
PetscErrorCode JDMG::Compute()
{
  PetscErrorCode ierr = 0;
  double compute_time = 0;
  if (this->verbose >= 1)
    compute_time = MPI_Wtime();
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Computing\n"); CHKERRQ(ierr);

  // Prep work
  ierr = PetscLogEventBegin(JDMG_Initialize, 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = Create_Hierarchy(); CHKERRQ(ierr);
  ierr = Compute_Init(); CHKERRQ(ierr);

  // Use Krylov-Schur on smallest grid to get a good starting point.
  ierr = Compute_Coarse(); CHKERRQ(ierr);

  // Initialize search subspace with interpolated coarse eigenvectors
  //PetscInt j = jmin;
  PetscInt j = std::min(nev_conv, jmax);
  for (int ii = 0; ii < j; ii++)
  {
    ierr = VecCopy(Q[0][ii], V[ii]); CHKERRQ(ierr);
  }
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
  ierr = PetscLogEventEnd(JDMG_Initialize, 0, 0, 0, 0); CHKERRQ(ierr);
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
      ierr = PetscLogEventBegin(JDMG_Prep, 0, 0, 0, 0); CHKERRQ(ierr);
      // Get new eigenvector approximation
      ierr = VecSet(Q[0][nev_conv], 0.0); CHKERRQ(ierr);
      ierr = VecMAXPY(Q[0][nev_conv], j, W.data(), V); CHKERRQ(ierr);
      theta = S(0);
      ierr = MatMult(A[0], Q[0][nev_conv], AQ[0][nev_conv]); CHKERRQ(ierr);
      ierr = MatMult(B[0], Q[0][nev_conv], BQ[0][nev_conv]); CHKERRQ(ierr);
      ierr = VecWAXPY(residual, -theta, BQ[0][nev_conv], AQ[0][nev_conv]); CHKERRQ(ierr);
      ierr = VecScale(residual, -1.0); CHKERRQ(ierr);

      // Construct parts of the new operator
      for (int ii = 1; ii < levels; ii++)
      {
        ierr = MatMultTranspose(P[ii-1], Q[ii-1][nev_conv], Q[ii][nev_conv]); CHKERRQ(ierr);
        ierr = MatMultTranspose(P[ii-1], AQ[ii-1][nev_conv], AQ[ii][nev_conv]); CHKERRQ(ierr);
        ierr = MatMultTranspose(P[ii-1], BQ[ii-1][nev_conv], BQ[ii][nev_conv]); CHKERRQ(ierr);
      }

      // Add parts to coarse operators
      PetscScalar *p_BQ;
      PetscInt col = ncoarse + nev_conv;
      ArrayXPI rows = ArrayXPI::LinSpaced(nlcoarse, 0, nlcoarse-1);
      ierr = VecGetArray(BQ.back()[nev_conv], &p_BQ); CHKERRQ(ierr);
      ierr = MatSetValues(AmsB.back(), 1, &col, nlcoarse, rows.data(),
                  p_BQ, INSERT_VALUES); CHKERRQ(ierr);
      ierr = MatSetValues(AmsB.back(), nlcoarse, rows.data(), 1, &col,
                  p_BQ, INSERT_VALUES); CHKERRQ(ierr);
      ierr = MatSetValue(AmsB.back(), col, col, 0.0, INSERT_VALUES); CHKERRQ(ierr);
      ierr = VecRestoreArray(BQ.back()[nev_conv], &p_BQ); CHKERRQ(ierr);
      ierr = MatAssemblyBegin(AmsB.back(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

      ierr = VecNorm(residual, NORM_2, &rnorm); CHKERRQ(ierr);
      ierr = VecNorm(AQ[0][nev_conv], NORM_2, &Au_norm);
      ierr = MatAssemblyEnd(AmsB.back(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


      if (this->verbose >= 2)
        PetscFPrintf(comm, output, "Iteration: %4i\tLambda Approx: %14.14g\tResidual: %4.4g\n", it, theta, rnorm);
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

      ierr = PetscLogEventEnd(JDMG_Prep, 0, 0, 0, 0); CHKERRQ(ierr);
      if ( ( (rnorm/theta >= eps) && (rnorm_old != rnorm) && (rnorm/Au_norm >= 1e-12) ) || (j <= 1) )
      {
        lambda(nev_conv) = theta;
        rnorm_old = rnorm;
        break;
      }
      ierr = PetscLogEventBegin(JDMG_Convergence, 0, 0, 0, 0); CHKERRQ(ierr);
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

      ierr = PetscLogEventEnd(JDMG_Convergence, 0, 0, 0, 0); CHKERRQ(ierr);
      if (Done())
      {
        // Cleanup
        ierr = VecDestroy(&residual); CHKERRQ(ierr);
        ierr = Compute_Clean(); CHKERRQ(ierr);
        if (this->verbose >= 1)
        {
          compute_time = MPI_Wtime() - compute_time;
          PetscFPrintf(comm, output, "JDMG found all %i eigenvalues in %i iterations, taking %1.6g seconds\n", 
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
    ierr = PetscLogEventBegin(JDMG_Update, 0, 0, 0, 0); CHKERRQ(ierr);
    vicinity = rnorm/theta < epstr; sigma_old = sigma;
    if (vicinity || tau != NUMERIC)
      sigma = lambda(nev_conv);
    else
      sigma = tau_num;

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
    ierr = PetscLogEventEnd(JDMG_Update, 0, 0, 0, 0); CHKERRQ(ierr);

    ierr = PetscLogEventBegin(JDMG_Expand, 0, 0, 0, 0); CHKERRQ(ierr);
    // Ensure orthogonality
    ierr = Mgsm(Q[0], BQ[0], V[j], nev_conv+1); CHKERRQ(ierr);
    ierr = Icgsm(V, B[0], V[j], orth_norm, j); CHKERRQ(ierr);
    ierr = VecScale(V[j], 1/orth_norm); CHKERRQ(ierr);

    // Update search space
    ierr = MatMult(A[0], V[j], TempVecs[0]); CHKERRQ(ierr);
    ierr = VecMDot(TempVecs[0], j+1, V, G.data()+j*jmax); CHKERRQ(ierr);
    G.block(j, 0, 1, j) = G.block(0, j, j, 1).transpose();
    ierr = PetscLogEventEnd(JDMG_Expand, 0, 0, 0, 0); CHKERRQ(ierr);

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
    PetscFPrintf(comm, output, "JDMG only found %i eigenvalues in %i iterations, taking %1.6g seconds\n",
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
bool JDMG::Done()
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
PetscErrorCode JDMG::Compute_Clean()
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
    ierr = VecCopy(Q[0][ii], phi[order(ii)]); CHKERRQ(ierr);
  }

  // Destroy eigenvectors at each level
  for (int ii = 0; ii < levels; ii++)
  {
    ierr = VecDestroyVecs(Qsize, Q.data()+ii);  CHKERRQ(ierr);
    ierr = VecDestroyVecs(Qsize, AQ.data()+ii); CHKERRQ(ierr);
    ierr = VecDestroyVecs(Qsize, BQ.data()+ii); CHKERRQ(ierr);
    ierr = MatDestroy(AmsB.data() + ii); CHKERRQ(ierr);
  }
  AQ.resize(0); BQ.resize(0);

  // Destroy workspace and search space
  ierr = VecDestroyVecs(jmax, &V); CHKERRQ(ierr);
  ierr = VecDestroyVecs(jmax, &TempVecs); CHKERRQ(ierr);

  // Destroy coarse problem
  ierr = KSPDestroy(&ksp_coarse); CHKERRQ(ierr);

  // Destroy Operators
  for (int ii = 0; ii < levels-1; ii++)
  {
    //ierr = MatDestroy(Acopy.data()+ii); CHKERRQ(ierr);
    ierr = MatDestroy(Bcopy.data()+ii); CHKERRQ(ierr);
    ierr = VecDestroy(xlist.data()+ii+1); CHKERRQ(ierr);
    ierr = VecDestroy(flist.data()+ii+1); CHKERRQ(ierr);
    ierr = VecDestroy(Dlist.data()+ii); CHKERRQ(ierr);
    ierr = VecDestroy(OPx.data()+ii); CHKERRQ(ierr);
    ierr = VecDestroyVecs(Qsize, QMatP.data()+ii); CHKERRQ(ierr);
  }
  ierr = VecDestroy(&x_end); CHKERRQ(ierr); ierr = VecDestroy(&f_end); CHKERRQ(ierr);
  Q.resize(0);

  return 0;
}

/******************************************************************************/
/**                  Sort the eigenvalues of the subspace                    **/
/******************************************************************************/
Eigen::ArrayXi JDMG::Sorteig(MatrixPS &W, ArrayPS &S)
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
PetscErrorCode JDMG::Icgsm(Vec *Q, Mat M, Vec u, PetscScalar &r, PetscInt k)
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
PetscErrorCode JDMG::Mgsm(Vec* Q, Vec* BQ, Vec u, PetscInt k)
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
PetscErrorCode JDMG::GS(Vec* Q, Vec Mu, Vec u, PetscInt k)
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
/**                Set up multigrid for correction equation                  **/
/******************************************************************************/
PetscErrorCode JDMG::MGSetup(Vec f, PetscReal fnorm)
{
  //TODO: Eliminate uncessary operations on Q, AQ, and BQ that aren't changing
  PetscErrorCode ierr = 0;
  ierr = PetscLogEventBegin(JDMG_MGSetUp, 0, 0, 0, 0); CHKERRQ(ierr);
  if (this->verbose >= 3)
    ierr = PetscFPrintf(comm, output, "Setting up multigrid\n"); CHKERRQ(ierr);
  // Coefficients for summing terms of diagonal
  ArrayPS sumcoeffs(6);
  //sumcoeffs << 1, -2, 1, -sigma, 2*sigma, -sigma;
  sumcoeffs << 1, -2, 1, -sigma, sigma, 0;
  PetscReal oldnorm = std::numeric_limits<PetscReal>::max();

  // Create D
  for (int ii = 0; ii < levels-1; ii++)
  {
    Vec PAPD, PAQPD, PQAQPD, PBPD, PBQPD, PQBQPD, *ALLD, WORK;
    ierr = VecDuplicateVecs(Dlist[ii], 7, &ALLD); CHKERRQ(ierr);
    PAPD = ALLD[0]; PAQPD = ALLD[1]; PQAQPD = ALLD[2];
    PBPD = ALLD[3]; PBQPD = ALLD[4]; PQBQPD = ALLD[5]; WORK = ALLD[6];

    // Subtract full Matrices
    /*ierr = MatZeroEntries(AmsB[ii]); CHKERRQ(ierr);
    ierr = MatCopy(Acopy[ii], AmsB[ii], SAME_NONZERO_PATTERN); CHKERRQ(ierr);
    ierr = MatAXPY(AmsB[ii], -sigma, Bcopy[ii], SAME_NONZERO_PATTERN); CHKERRQ(ierr);*/
    ierr = MatAXPY(AmsB[ii], sigma_old-sigma, Bcopy[ii], SAME_NONZERO_PATTERN); CHKERRQ(ierr);

    // A terms
    ierr = MatGetDiagonal(A[ii], PAPD); CHKERRQ(ierr);
    ierr = VecPointwiseMult(PAQPD, AQ[ii][0], BQ[ii][0]); CHKERRQ(ierr);
    ierr = VecPointwiseMult(PQAQPD, BQ[ii][0], BQ[ii][0]); CHKERRQ(ierr);
    ierr = VecScale(PQAQPD, lambda(0)); CHKERRQ(ierr);
    for (int jj = 1; jj < nev_conv+1; jj++)
    {
      ierr = VecPointwiseMult(WORK, AQ[ii][jj], BQ[ii][jj]); CHKERRQ(ierr);
      ierr = VecAXPY(PAQPD, 1.0, WORK); CHKERRQ(ierr);
      ierr = VecPointwiseMult(WORK, BQ[ii][jj], BQ[ii][jj]); CHKERRQ(ierr);
      ierr = VecAXPY(PQAQPD, lambda(jj), WORK); CHKERRQ(ierr);
    }

    // B terms
    ierr = MatGetDiagonal(B[ii], PBPD); CHKERRQ(ierr);
    ierr = VecPointwiseMult(PBQPD, BQ[ii][0], BQ[ii][0]); CHKERRQ(ierr);
    // PBQPD = PQBQPD
    //ierr = VecPointwiseMult(PQBQPD, BQ[ii][0], BQ[ii][0]); CHKERRQ(ierr);
    for (int jj = 1; jj < nev_conv+1; jj++)
    {
      ierr = VecPointwiseMult(WORK, BQ[ii][jj], BQ[ii][jj]); CHKERRQ(ierr);
      ierr = VecAXPY(PBQPD, 1.0, WORK); CHKERRQ(ierr);
      //ierr = VecPointwiseMult(WORK, BQ[ii][jj], BQ[ii][jj]); CHKERRQ(ierr);
      //ierr = VecAXPY(PQBQPD, 1.0, WORK); CHKERRQ(ierr);
    }

    //Total diagonal
    ierr = VecSet(Dlist[ii], 0.0); CHKERRQ(ierr);
    ierr = VecMAXPY(Dlist[ii], 5, sumcoeffs.data(), ALLD); CHKERRQ(ierr);
    ierr = VecDestroyVecs(7, &ALLD); CHKERRQ(ierr);

    //Smoothers
    for (int jj = 0; jj < nev_conv+1; jj++)
    {
      ierr = VecCopy(AQ[ii][jj], QMatP[ii][jj]); CHKERRQ(ierr);
      ierr = VecAXPY(QMatP[ii][jj], -sigma, BQ[ii][jj]); CHKERRQ(ierr);
    }

    if ((ii + nev_conv == 0) && !vicinity)
    {
      Vec x;
      ierr = VecDuplicate(f, &x); CHKERRQ(ierr); 
      ArrayPS QMatQ = lambda.segment(0,nev_conv+1) - sigma;
      ierr = VecSet(x, 0.0); CHKERRQ(ierr);
      ierr = WJac(QMatP[ii], QMatQ, Dlist[ii], f, x, ii); CHKERRQ(ierr);
      ierr = ApplyOP(QMatP[ii], QMatQ, x, OPx[ii], ii); CHKERRQ(ierr);
      ierr = VecAYPX(OPx[ii], -1.0, f); CHKERRQ(ierr);
      PetscReal OPx_norm;
      ierr = VecNorm(OPx[ii], NORM_2, &OPx_norm); CHKERRQ(ierr);
      if (OPx_norm > 10*fnorm && OPx_norm < oldnorm)
      {
        PetscFPrintf(comm, output, "F norm: %8.8g\t, OP norm: %8.8g\t, old norm %8.8g\n", fnorm, OPx_norm, oldnorm);
        oldnorm = OPx_norm;
        if (this->verbose >= 2){
          ierr = PetscFPrintf(comm, output, "Bad shift parameter, increasing shift from %1.6g to %1.6g\n",
                              sigma, sigma*10); CHKERRQ(ierr); }
        sigma *= 10;
        sumcoeffs << 1, -2, 1, -sigma, 2*sigma, -sigma;
	// Reset the shifted operator
        ierr = MatAXPY(AmsB[ii], sigma-sigma_old, Bcopy[ii], SAME_NONZERO_PATTERN); CHKERRQ(ierr);
        ii--;
      }
      ierr = VecDestroy(&x); CHKERRQ(ierr);
    }
  }

  // Coarse scale solver 
  ierr = MatShift(); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(JDMG_MGSetUp, 0, 0, 0, 0); CHKERRQ(ierr);

  return ierr;
}

/******************************************************************************/
/**               Apply full multigrid for correction equation               **/
/******************************************************************************/
PetscErrorCode JDMG::FullMGSolve(Vec x, Vec f)
{
  PetscErrorCode ierr = 0;
  ierr = PetscLogEventBegin(JDMG_Precondition, 0, 0, 0, 0); CHKERRQ(ierr);
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
        PetscScalar *p_Vec1, *p_Vec2;
        ierr = VecGetArray(flist.back(), &p_Vec1); CHKERRQ(ierr);
        ierr = VecGetArray(f_end, &p_Vec2); CHKERRQ(ierr);
        copy(p_Vec1, p_Vec1+nlcoarse, p_Vec2);
        ierr = VecRestoreArray(flist.back(), &p_Vec1); CHKERRQ(ierr);
        ierr = VecRestoreArray(f_end, &p_Vec2); CHKERRQ(ierr);
        ierr = KSPSolve(ksp_coarse, f_end, x_end); CHKERRQ(ierr);
        ierr = VecGetArray(xlist.back(), &p_Vec1); CHKERRQ(ierr);
        ierr = VecGetArray(x_end, &p_Vec2); CHKERRQ(ierr);
        copy(p_Vec2, p_Vec2+nlcoarse, p_Vec1);
        ierr = VecRestoreArray(xlist.back(), &p_Vec1); CHKERRQ(ierr);
        ierr = VecRestoreArray(x_end, &p_Vec2); CHKERRQ(ierr);
      }
      else
      {
        ierr = WJac(QMatP[jj], QMatQ, Dlist[jj], flist[jj], xlist[jj], jj); CHKERRQ(ierr);
        ierr = ApplyOP(QMatP[jj], QMatQ, xlist[jj], OPx[jj], jj); CHKERRQ(ierr);
        ierr = VecAYPX(OPx[jj], -1.0, flist[jj]); CHKERRQ(ierr);
        ierr = MatMultTranspose(P[jj], OPx[jj], flist[jj+1]); CHKERRQ(ierr);
        ierr = VecSet(xlist[jj+1], 0.0); CHKERRQ(ierr);
      }
    }
    //Upcycling
    for (int jj = levels-2; jj >= ii; jj--)
    {
      ierr = MatMultAdd(P[jj], xlist[jj+1], xlist[jj], xlist[jj]); CHKERRQ(ierr);
      if (jj == 0)
      {
        ierr = VecMDot(xlist[jj], nev_conv+1, BQ[jj], TempScal.data()); CHKERRQ(ierr);
        TempScal *= -1;
        ierr = VecMAXPY(xlist[jj], nev_conv+1, TempScal.data(), Q[jj]); CHKERRQ(ierr);
      }
      ierr = WJac(QMatP[jj], QMatQ, Dlist[jj], flist[jj], xlist[jj], jj); CHKERRQ(ierr);
    }
    if (ii > 0)
    {
      ierr = MatMult(P[ii-1], xlist[ii], xlist[ii-1]); CHKERRQ(ierr);
    }
    if (ii == 1)
    {
      ierr = VecMDot(xlist[ii-1], nev_conv+1, BQ[ii-1], TempScal.data()); CHKERRQ(ierr);
      TempScal *= -1;
      ierr = VecMAXPY(xlist[ii-1], nev_conv+1, TempScal.data(), Q[ii-1]); CHKERRQ(ierr);
    }
  }

  ierr = PetscLogEventEnd(JDMG_Precondition, 0, 0, 0, 0); CHKERRQ(ierr);
  return 0;
}     

/******************************************************************************/
/**                 Apply multigrid for correction equation                  **/
/******************************************************************************/
PetscErrorCode JDMG::MGSolve(Vec x, Vec f)
{
  PetscErrorCode ierr = 0;
  ierr = PetscLogEventBegin(JDMG_Precondition, 0, 0, 0, 0); CHKERRQ(ierr);
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
    ierr = ApplyOP(QMatP[ii], QMatQ, xlist[ii], OPx[ii], ii); CHKERRQ(ierr);
    ierr = VecAYPX(OPx[ii], -1.0, flist[ii]); CHKERRQ(ierr);
    ierr = MatMultTranspose(P[ii], OPx[ii], flist[ii+1]); CHKERRQ(ierr);
  }

  // Coarse solve
  PetscScalar *p_Vec1, *p_Vec2;
  ierr = VecGetArray(flist.back(), &p_Vec1); CHKERRQ(ierr);
  ierr = VecGetArray(f_end, &p_Vec2); CHKERRQ(ierr);
  copy(p_Vec1, p_Vec1+nlcoarse, p_Vec2);
  ierr = VecRestoreArray(flist.back(), &p_Vec1); CHKERRQ(ierr);
  ierr = VecRestoreArray(f_end, &p_Vec2); CHKERRQ(ierr);
  ierr = KSPSolve(ksp_coarse, f_end, x_end); CHKERRQ(ierr);
  ierr = VecGetArray(xlist.back(), &p_Vec1); CHKERRQ(ierr);
  ierr = VecGetArray(x_end, &p_Vec2); CHKERRQ(ierr);
  copy(p_Vec2, p_Vec2+nlcoarse, p_Vec1);
  ierr = VecRestoreArray(xlist.back(), &p_Vec1); CHKERRQ(ierr);
  ierr = VecRestoreArray(x_end, &p_Vec2); CHKERRQ(ierr);

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
    ierr = WJac(QMatP[ii], QMatQ, Dlist[ii], flist[ii], xlist[ii], ii); CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(JDMG_Precondition, 0, 0, 0, 0); CHKERRQ(ierr);
  return 0;
}

/******************************************************************************/
/**                   Subtracting matrices at coarse level                   **/
/******************************************************************************/
PetscErrorCode JDMG::MatShift()
{
  PetscErrorCode ierr = 0;

  PetscInt rstart = 0, rend = 0, nzA, nzB;
  const PetscInt *cworkA, *cworkB;
  const PetscScalar *vworkA, *vworkB;
  vector<PetscInt> allCols;
  vector<PetscScalar> allVals;
  vector<PetscInt>::iterator aCit;
  ierr = MatGetOwnershipRange(A.back(), &rstart, &rend); CHKERRQ(ierr);
  for (int ii = rstart; ii < rend; ii++)
  {
    // Get rows of each matrix
    ierr = MatGetRow(A.back(), ii, &nzA, &cworkA, &vworkA); CHKERRQ(ierr);
    ierr = MatGetRow(B.back(), ii, &nzB, &cworkB, &vworkB); CHKERRQ(ierr);
    // Combine indices
    allCols.resize(nzA+nzB);
    copy(cworkA, cworkA+nzA, allCols.data());
    copy(cworkB, cworkB+nzB, allCols.data()+nzA);
    sort(allCols.begin(), allCols.end());
    aCit = unique(allCols.begin(), allCols.end());
    // Combine Values
    allVals.resize(aCit-allCols.begin());
    fill(allVals.begin(), allVals.end(), 0.0);
    // Add A
    unsigned int col = 0;
    for (int jj = 0; jj < nzA; jj++)
    {
      while (col < allVals.size() && cworkA[jj] != allCols[col])
        col++;
      if (col == allVals.size())
        break;

      allVals[col] = vworkA[jj];
    }
    // Subtract sigma*B
    col = 0;
    for (int jj = 0; jj < nzB; jj++)
    {
      while (col < allVals.size() && cworkB[jj] != allCols[col])
        col++;
      if (col == allVals.size())
        break;

      allVals[col] -= sigma*vworkB[jj];
    }
    ierr = PetscLogFlops(nzB); CHKERRQ(ierr);

    // Insert into matrix
    ierr = MatSetValues(AmsB.back(), 1, &ii, allVals.size(), allCols.data(),
        allVals.data(), INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(A.back(), ii, &nzA, &cworkA, &vworkA); CHKERRQ(ierr);
    ierr = MatRestoreRow(B.back(), ii, &nzB, &cworkB, &vworkB); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(AmsB.back(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(AmsB.back(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  return 0;
}

/******************************************************************************/
/**                        Weighted Jacobi smoother                          **/
/******************************************************************************/
PetscErrorCode JDMG::WJac(Vec* QMatP, ArrayPS &QMatQ, Vec D, Vec y, Vec x, PetscInt level)
{
  // The y being fed in is -r, as it should be
  PetscErrorCode ierr = 0;
  ierr = PetscLogEventBegin(JDMG_Jacobi, 0, 0, 0, 0); CHKERRQ(ierr);
  if (nsweep == 0)
    return 0;

  Vec r;
  ierr = VecDuplicate(y, &r); CHKERRQ(ierr);
  for (int ii = 0; ii < nsweep; ii++)
  {
    ierr = ApplyOP(QMatP, QMatQ, x, r, level); CHKERRQ(ierr);
    ierr = VecAYPX(r, -1.0, y); CHKERRQ(ierr);
    ierr = VecPointwiseDivide(r, r, D); CHKERRQ(ierr);
    ierr = VecAXPY(x, w, r); CHKERRQ(ierr);
  }
  VecDestroy(&r);
  ierr = PetscLogEventEnd(JDMG_Jacobi, 0, 0, 0, 0); CHKERRQ(ierr);

  return 0;
}

/******************************************************************************/
/**                         Apply combined operator                          **/
/******************************************************************************/
PetscErrorCode JDMG::ApplyOP(Vec* QMatP, ArrayPS &QMatQ, Vec x, Vec y, PetscInt level)
{
  PetscErrorCode ierr = 0;
  ierr = PetscLogEventBegin(JDMG_ApplyOP[level], 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = PetscLogEventBegin(JDMG_ApplyOP1[level], 0, 0, 0, 0); CHKERRQ(ierr);
  int one = 1, bn;
  ierr = VecGetLocalSize(x, &bn); CHKERRQ(ierr);
  ArrayPS PQBx(nev_conv+1), QMatPx(nev_conv+1);
  PetscScalar *p_x, **p_BQ, **p_QMatP;
  MPI_Request request1 = MPI_REQUEST_NULL, request2 = MPI_REQUEST_NULL;

  // Local dot products
  ierr = VecGetArray(x, &p_x); CHKERRQ(ierr);
  ierr = VecGetArrays(BQ[level], nev_conv+1, &p_BQ); CHKERRQ(ierr);
  ierr = VecGetArrays(QMatP, nev_conv+1, &p_QMatP); CHKERRQ(ierr);
  for (int ii = 0; ii < nev_conv+1; ii++)
  {
    PQBx(ii) =  -ddot_(&bn, p_x, &one, p_BQ[ii], &one);
    QMatPx(ii) = ddot_(&bn, p_x, &one, p_QMatP[ii], &one);
  }
  if (bn > 0)
    ierr = PetscLogFlops(2*(nev_conv+1)*(2*bn-1)); CHKERRQ(ierr);
  ierr = VecRestoreArray(x, &p_x); CHKERRQ(ierr);
  ierr = VecRestoreArrays(BQ[level], nev_conv+1, &p_BQ); CHKERRQ(ierr);
  ierr = VecRestoreArrays(QMatP, nev_conv+1, &p_QMatP); CHKERRQ(ierr);

  // Reduction for dot products
  MPI_Iallreduce(MPI_IN_PLACE, PQBx.data(), nev_conv+1, MPI_DOUBLE,
                MPI_SUM, MG_comms[level], &request1);
  MPI_Iallreduce(MPI_IN_PLACE, QMatPx.data(), nev_conv+1, MPI_DOUBLE,
                MPI_SUM, MG_comms[level], &request2);
  // This is so VecMAXPY doesn't throw an error when debugging
  #if defined(PETSC_USE_DEBUG)
  MPI_Wait(&request1, MPI_STATUS_IGNORE);
  MPI_Wait(&request2, MPI_STATUS_IGNORE);
  MPI_Bcast(PQBx.data(), nev_conv+1, MPI_DOUBLE, 0, comm);
  MPI_Bcast(QMatPx.data(), nev_conv+1, MPI_DOUBLE, 0, comm);
  #endif

  // Term 1
  ierr = PetscLogEventEnd(JDMG_ApplyOP1[level], 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = PetscLogEventBegin(JDMG_ApplyOP2[level], 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = MatMult(AmsB[level], x, y); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(JDMG_ApplyOP2[level], 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = PetscLogEventBegin(JDMG_ApplyOP3[level], 0, 0, 0, 0); CHKERRQ(ierr);

  // Term 2
  MPI_Wait(&request1, MPI_STATUS_IGNORE);
  ierr = PetscLogEventEnd(JDMG_ApplyOP3[level], 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = PetscLogEventBegin(JDMG_ApplyOP4[level], 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = VecMAXPY(y, nev_conv+1, PQBx.data(), QMatP); CHKERRQ(ierr);

  // Terms 3 and 4
  MPI_Wait(&request2, MPI_STATUS_IGNORE);

  PQBx = PQBx.cwiseProduct(QMatQ);
  ierr = PetscLogFlops(QMatQ.size()); CHKERRQ(ierr);
  PQBx += QMatPx;
  PQBx *= -1;
  ierr = VecMAXPY(y, nev_conv+1, PQBx.data(), BQ[level]); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(JDMG_ApplyOP4[level], 0, 0, 0, 0); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(JDMG_ApplyOP[level], 0, 0, 0, 0); CHKERRQ(ierr);

  return 0;
}
