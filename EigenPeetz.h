#ifndef EigenPeetz_H_INCLUDED
#define EigenPeetz_H_INCLUDED

#include <slepceps.h>
#include <vector>
#include <Eigen/Eigen>
#include <algorithm>

typedef Eigen::Array<PetscInt, -1, -1> ArrayXXPI;
typedef Eigen::Array<PetscInt, -1,  1> ArrayXPI;
typedef Eigen::Array<PetscInt, -1, -1, Eigen::RowMajor> ArrayXXPIRM;
typedef Eigen::Matrix<double, -1, -1, Eigen::RowMajor> MatrixXdRM;
typedef Eigen::Matrix<PetscScalar, -1, -1> MatrixPS;
typedef Eigen::Array<PetscScalar, -1, 1>  ArrayPS;
#define MPI_PETSCINT MPIU_INT

extern PetscLogEvent EIG_Compute;
extern PetscLogEvent EIG_Initialize, EIG_Prep, EIG_Convergence, EIG_Expand, EIG_Update;
extern PetscLogEvent EIG_Comp_Init, EIG_Hierarchy, EIG_Setup_Coarse, EIG_Comp_Coarse;
extern PetscLogEvent EIG_MGSetup, EIG_Precondition, EIG_Jacobi, *EIG_ApplyOP;
extern PetscLogEvent *EIG_ApplyOP1, *EIG_ApplyOP2, *EIG_ApplyOP3, *EIG_ApplyOP4;

enum Tau {NUMERIC, LM, LR, LA, SM, SR, SA };
enum Nev_Type {TOTAL_NEV, UNIQUE_NEV, UNIQUE_LAST_NEV};

/// The master structure containing all information to be carried between iterations
class EigenPeetz
{
  /// Class variables
public:

  MPI_Comm comm;                               //MPI communicator
  std::vector<MPI_Comm> MG_comms;              //MPI communicator for levels of hierarchy
  int myid;                                    //Rank of this process
  int nprocs;                                  //Total number of processes

  /// Class methods
  // Constructors
  EigenPeetz();
  EigenPeetz(MPI_Comm comm);
  // Destructor
  ~EigenPeetz();
  // How much information to print
  virtual PetscErrorCode Set_Verbose(PetscInt verbose);
  // Where to print the information
  PetscErrorCode Set_File(FILE *output); // For already opened files
  PetscErrorCode Open_File(const char filename[]); // For files to be opened within EigenPeetz
  PetscErrorCode Close_File() {if (file_opened) {return PetscFClose(comm, output);} return 0;}
  // Set operators of eigensystem
  PetscErrorCode Set_Operators(Mat A, Mat B);
  // Setting target eigenvalues and number to find
  void Set_Target(Tau tau, PetscInt nev, Nev_Type ntype);
  void Set_Target(PetscScalar tau, PetscInt nev, Nev_Type ntype);
  void Set_Tol(double tol) {eps = tol;}
  void Set_MaxIt(PetscInt maxit) {this->maxit = maxit;}
  PetscInt Get_Iterations() {return it;}
  // Solver
  virtual PetscErrorCode Compute() = 0;
  // Get results
  PetscInt Get_nev_conv() {return this->nev_conv;}
  // Ownership is retained by EigenPeetz
  void Get_Eigenvectors(Vec** phi) {*phi = this->phi;}
  void Get_Eigenvalues(PetscScalar* lambda)
    {std::copy(this->lambda.data(), this->lambda.data()+this->nev_conv, lambda);}

  // Loggers
  static PetscErrorCode Initialize();
  static PetscErrorCode Finalize();

protected:
  // Amount of information to print (0, 1, or 2)
  int verbose;
  // Where to print it
  FILE *output;
  bool file_opened;
  // System Matrices and their coarse-grid representations
  std::vector<Mat> A, B;
  // Eigenvalues and EigenVectors
  Vec* phi;
  ArrayPS lambda;
  // Workspace
  Vec *TempVecs;
  ArrayPS TempScal;
  // Problem size
  PetscInt n, nlocal;
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
  // Convergence tolerance
  double eps;
  // Maximum and total run iterations
  PetscInt maxit, it;

  /// Protected methods
  // Set MPI info
  void Set_ID() {MPI_Comm_rank(comm, &myid); MPI_Comm_size(comm, &nprocs);}
  // Check if all requested eigenvalues have been found
  bool Done();
  // Sort eigenvalues
  Eigen::ArrayXi Sorteig(MatrixPS &W, ArrayPS &S);
  // Gram-Schmidt methods
  PetscErrorCode Icgsm(Vec* Q, Mat M, Vec u, PetscScalar &r, PetscInt k);
  PetscErrorCode Mgsm(Vec* Q, Vec* Qm, Vec u, PetscInt k);
  PetscErrorCode GS(Vec* Q, Vec Mu, Vec u, PetscInt k);
  // Output information
  virtual PetscErrorCode Print_Status(PetscReal rnorm) {return PetscFPrintf(comm, output, "Iteration: %4i\tLambda Approx: %14.14g\tResidual: %4.4g\n", it, lambda(nev_conv), rnorm);}
  virtual PetscErrorCode Print_Result() {return PetscFPrintf(comm, output, "Eigensolver found %i of a requested %i eigenvalues after %i iterations\n\n", nev_conv, nev_req, it);}

};

#endif // EigenPeetz_H_INCLUDED
