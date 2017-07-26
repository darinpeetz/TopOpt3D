#ifndef Eigen_H_INCLUDED
#define Eigen_H_INCLUDED

#include <slepceps.h>
#include <vector>
#include <Eigen/Eigen>
#include <algorithm>

typedef Eigen::Array<PetscInt, -1, -1> ArrayXXPI;
typedef Eigen::Array<PetscInt, -1,  1> ArrayXPI;
typedef Eigen::Array<PetscInt, -1, -1, Eigen::RowMajor> ArrayXXPIRM;
typedef Eigen::Matrix<double, -1, -1, Eigen::RowMajor> MatrixXdRM;
#define MPI_PETSCINT MPIU_INT

enum Tau { NUMERIC, LM, LR, LA, SM, SR, SA };
enum Nev_Type { TOTAL_NEV, UNIQUE_NEV, UNIQUE_LAST_NEV};

typedef Eigen::Matrix<PetscScalar, -1, -1> MatrixPS;
typedef Eigen::Array<PetscScalar, -1, 1>  ArrayPS;

/// The master structure containing all information to be carried between iterations
class JDMG
{
  /// Class variables
public:

  MPI_Comm comm;                               //MPI communicator
  int myid;                                    //Rank of this process
  int nprocs;                                  //Total number of processes

  /// Class methods
  // Constructors
  JDMG() {JDMG(MPI_COMM_WORLD);}
  JDMG(MPI_Comm comm);
  // Destructor
  ~JDMG();
  // How much information to print
  PetscErrorCode Set_Verbose(PetscInt verbose);
  // Set operators of eigensystem
  PetscErrorCode Set_Operators(Mat A, Mat B);
  // Get hierarchy from existing PCMG object
  PetscErrorCode PCMG_Extract(PC pcmg, bool isB=true, bool isA=false);
  PetscErrorCode Set_Hierarchy(std::vector<Mat> P);
  // Setting target eigenvalues and number to find
  void Set_Target(Tau tau, PetscInt nev, Nev_Type ntype);
  void Set_Target(PetscScalar tau, PetscInt nev, Nev_Type ntype);
  void Set_Tol(double tol) {eps = tol;}
  void Set_MaxIt(PetscInt maxit) {this->maxit = maxit;}
  PetscInt Get_Iterations() {return it;}
  // Solver
  PetscErrorCode Compute_Coarse();
  PetscErrorCode Compute();
  // Get results
  void Get_nev_conv(PetscInt &nev_conv) {nev_conv = this->nev_conv;}
  // Ownership is retained by JDMG
  void Get_EigenVectors(Vec** phi) {*phi = this->phi;}
  void Get_EigenValues(PetscScalar* lambda)
    {std::copy(this->lambda.data(), this->lambda.data()+this->nev_conv, lambda);}

/*double t_Set_Target, t_Set_Operators, t_PCMG_Extract, t_Sorteig, t_Mat_Shift,
  t_Setup_Coarse, t_GS, t_MGSM, t_ICGSM, t_Compute_Coarse, t_Compute,
  t_Compute_Init, t_Compute_Clean, t_Create_Hierarchy, t_MG, t_WJac, t_ApplyOP,
  t_MG_Setup, t_Coarse_Solve, t_Cycle, t_term0, t_term1, t_term2, t_term34;*/

private:
  // Amount of information to print (0, 1, or 2)
  int verbose;
  // Prolongation Matrices
  std::vector<Mat> P;
  // System Matrices and their coarse-grid representations
  std::vector<Mat> A, B;
  // Number of levels in hierarchy
  PetscInt levels;
  // Eigenvalues and EigenVectors
  Vec* phi;
  ArrayPS lambda;
  // Process to store PQ part of coarse operator
  PetscInt endrank;
  // Problem size
  PetscInt n, nlocal, ncoarse, nlcoarse;
  // Subspace and work array
  Vec *V, *TempVecs;
  // Scalar work space
  ArrayPS TempScal;
  // Number of requested and converged eigenvalues
  PetscInt nev_req, nev_conv;
  // Size of Q subspace in iterations
  PetscInt Qsize;
  // Requirement on unique eigenvalues
  Nev_Type nev_type;
  // Target eigenvalues
  Tau tau;
  // Numeric Target if desired
  double tau_num;
  // Convergence tolerance and tracking tolerance
  double eps, epstr;
  // Flag if sigma is getting close to lambda_max
  bool vicinity;
  // Subspace min and max size
  PetscInt jmin, jmax;
  // Maximum and total run iterations of JD scheme
  PetscInt maxit, it;
  // Flag indicating if Compute_Init needs to be run
  bool prepped;
  // EPS object for coarse scale eigenvalue problem and KSP for solver
  EPS eps_coarse;
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
  std::vector<Mat> Acopy, Bcopy, AmsB;
  std::vector<Vec*> QMatP;
  std::vector<Vec> OPx;
  PetscScalar sigma;

  /// Private methods
  // Set MPI info
  void Set_ID() {MPI_Comm_rank(comm, &myid); MPI_Comm_size(comm, &nprocs);}
  // Prepare solver for compute step
  PetscErrorCode Compute_Init();
  PetscErrorCode Setup_Coarse();
  // Create coarse grid representations
  PetscErrorCode Create_Hierarchy();
  // Check if all requested eigenvalues have been found
  bool Done();
  // Cleanup after compute step
  PetscErrorCode Compute_Clean();
  // Sort eigenvalues
  void Sorteig(MatrixPS &W, ArrayPS &S);
  // Gram-Schmidt methods
  PetscErrorCode Icgsm(Vec* Q, Mat M, Vec u, PetscScalar &r, PetscInt k);
  PetscErrorCode Mgsm(Vec* Q, Vec* Qm, Vec u, PetscInt k);
  PetscErrorCode GS(Vec* Q, Vec Mu, Vec u, PetscInt k);
  // Multigrid solver
  PetscErrorCode MG(Vec x, Vec f, PetscScalar fnorm);
  // Set up coarse grid matrix
  PetscErrorCode MatShift();
  // Weighted Jacobi smoothing
  PetscErrorCode WJac(Vec* QMatP, ArrayPS &QMatQ, Vec D, Vec y, Vec x, PetscInt level);
  // Apply Operator at a given level
  PetscErrorCode ApplyOP(Vec* QMatP, ArrayPS &QMatQ, Vec x, Vec y, PetscInt level);

};

#endif // Eigen_H_INCLUDED
