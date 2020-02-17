#ifndef TopOpt_H_INCLUDED
#define TopOpt_H_INCLUDED

#include <slepceps.h>
#include <vector>
#include <Eigen/Eigen>
#include "MMA.h"
#include "Functions.h"

extern "C"
{
  #include <parmetis.h>
}

typedef Eigen::Array<PetscInt, -1, -1> ArrayXXPI;
typedef Eigen::Array<PetscInt, -1,  1> ArrayXPI;
typedef Eigen::Array<PetscInt, -1, -1, Eigen::RowMajor> ArrayXXPIRM;
typedef Eigen::Matrix<PetscScalar, -1, -1, Eigen::RowMajor> MatrixXdRM;
typedef Eigen::Array<PetscScalar, -1, -1> ArrayXXPS;
typedef Eigen::Array<PetscScalar, -1, 1> ArrayXPS;
typedef Eigen::Matrix<PetscScalar, -1, -1> MatrixXPS;
typedef Eigen::Matrix<PetscScalar, -1, 1> VectorXPS;
#define MPI_PETSCINT MPIU_INT
#define MPI_PETSCSCALAR MPIU_SCALAR

enum BCTYPE { SUPPORT, LOAD, MASS, SPRING, OTHER };
enum MATINT { SIMP, SIMP_CUT, SIMP_LOGISTIC };

/// The master structure containing all information to be carried between iterations
class TopOpt
{
  /// Class variables - Setting all to public for now, but may like to change this later
public:

  //MPI communicator
  MPI_Comm comm;
  //Rank of this process
  int myid;
  //Total number of processes
  int nprocs;
  //Input file name
  std::string filename;
  //How much information to print
  int verbose;
  //How often to output results
  int print_every, last_print;
  //File for outputing information
  FILE* output;
  //Location of files for restart
  std::string folder;

  /// Mesh variables
  //Dimensionality of problem
  short numDims;
  //Nodal coordinate
  MatrixXdRM node;
  //Element Node numbers
  ArrayXXPIRM element;
  //How elements are distributed on processes
  ArrayXPI elmdist;
  //How nodes are distributed on processes
  ArrayXPI nddist;
  //Total number of nodes and elements
  PetscInt nNode, nElem;
  //Number of local nodes and elements
  PetscInt nLocElem, nLocNode;
  //Global numbering of elements and nodes stored locally
  ArrayXPI gElem, gNode;
  //Element sizes in m^numDims
  VectorXPS elemSize;
  //Flag to indicate if all elements are identical
  bool regular;

  /// FEM setup variables - only used for FEM initialization
  //Element characteristics, E0 in Pa
  double Nu0, E0;
  //Element density in kg/m
  double density;
  //Support node numbers
  ArrayXPI suppNode;
  //Boolean indicating if dof is fixed or not
  Eigen::Array<bool,-1, -1, Eigen::RowMajor> supports;
  //Spring Support node numbers
  ArrayXPI springNode;
  //Spring dof stiffnesses in Pa
  Eigen::Array<double, -1, -1, Eigen::RowMajor> springs;
  //Load nodes
  ArrayXPI loadNode;
  //Loads values in N
  Eigen::Array<double, -1, -1, Eigen::RowMajor> loads;
  //Lumped mass nodes
  ArrayXPI massNode;
  //mass values in kg
  Eigen::Array<double, -1, -1, Eigen::RowMajor> masses;

  /// FEM solution variables - used in each FEM iteration
  //Constitutive Matrix
  MatrixXPS d;
  // Element Jacobian
  double detJ;
  //B and G matrices for assembling stiffness matrices
  MatrixXPS *B, *G, *GT;
  //Integration point weights
  double *W;
  //Indices of local k matrix for constructing global K matrix
  std::vector<double> k;
  //Triplet information for assembling stiffness matrix
  std::vector<PetscInt> i, j, e;
  //Vector of values for individual elements
  std::vector<MatrixXPS> ke;
  //Force vector
  Vec F;
  //Vector of displacements from fem problem
  Vec U;
  // Maximum stiffness of elements attached to dof
  Vec MaxStiff;
  //Global indices of free local dofs
  ArrayXPI freeDof;
  //Global indices of fixed local dofs
  ArrayXPI fixedDof;
  //Global indices of local dofs with springs
  ArrayXPI springDof;
  //Global indices of local dofs without springs
  ArrayXPI springlessDof;
  //Total number of free dofs
  PetscInt nFreeDof;
  //Total number of fixed dofs
  PetscInt nFixDof;
  //Number of dofs with springs attached
  PetscInt nSpringDof;
  //Vector containing every dof number
  ArrayXXPI dofs;
  //Sparse matrix used to store stiffness of springs
  Mat spK;
  //Vector representing diagonal of spring matrix
  Vec spKVec;
  //Sparse K used to solve fem problem
  Mat K;
  //Interpolation/Restriction matrices
  std::vector<Mat> PR;
  //Communicator for each level of MG hierarchy
  std::vector<MPI_Comm> MG_comms;
  //Smoother to use with multigrid preconditioners
  std::string smoother;
  //Storing point masses
  Vec MLump;
  //The FEM solver context
  KSP KUF;
  //Solver context for dynamic ST
  KSP dynamicKSP;
  //Solver context for buckling ST
  KSP bucklingKSP;
  //Flag to use direct instead of iterative solver
  bool direct;

  /// Function information
  std::vector<Function_Base*> function_list;
  PetscBool needK, needU;

  /// Optimization variables
  //penalization factor information
  PetscScalar penal, vdPenal;
  std::vector<PetscScalar> penalties;
  std::vector<PetscScalar> void_penalties;
  //Minimum Radius Filter Matrix
  Mat P;
  //Maximum Length Scale Filter Matrix
  Mat R; Vec REdge;
  //Minimum number of voids within Rmax
  PetscScalar vdMin;
  //Material Interpolation type
  MATINT interpolation;
  std::vector<PetscScalar> interp_param;
  //Material Interpolation Values
  Vec V, dVdrho, E, dEdz, Es, dEsdz;
  //Intermediate values for material interpoloation
  Vec rhoq, y;
  //Raw densities and filtered densities, rho = P*x
  Vec x, rho;
  //Active vs. passive elements
  Eigen::Array<bool, -1, 1> active;
  //Eigenvectors
  MatrixXPS bucklingShape, dynamicShape;
  //Deflation spaces for eigenvalue problems
  Vec *bucklingDeflate, *dynamicDeflate;
  //Number of iterations for eigenvalue problems
  PetscInt bucklingIt, dynamicIt;

  /// Profiling variables
  int funcEvent, FEEvent, UpdateEvent;

  /// Class methods
  // Constructors
  TopOpt() {comm = MPI_COMM_WORLD; Initialize();}
  TopOpt(MPI_Comm comm) {this->comm = comm; Initialize();}
  TopOpt(MPI_Comm comm, short numDims)
 	{this->comm = comm; SetDimension(numDims); Initialize();}
  TopOpt(short numDims) {comm = MPI_COMM_WORLD; SetDimension(numDims); Initialize();}
  // Initialization done by each constructor
  PetscErrorCode Initialize();
  // Destructor
  ~TopOpt() {Clear();}

  // Parsing the input file
  PetscErrorCode Def_Param(MMA *optmma, VectorXPS &Dimensions, ArrayXPI &Nel,
                 PetscScalar &Rmin, PetscScalar &Rmax, bool &Normalization,
                 bool &Reorder_Mesh, PetscInt &mg_levels, PetscInt &min_size);
  PetscErrorCode Get_CL_Options();
  PetscErrorCode Set_Funcs();
  PetscErrorCode Domain(MatrixXPS &Points, Eigen::Array<bool, -1, 1> &elemValidity);
  PetscErrorCode Def_BC();
  PetscErrorCode Set_BC(ArrayXPS center, ArrayXPS radius,
                        ArrayXXPS limits, ArrayXPS values, BCTYPE TYPE);

  // Basic methods
  void MPI_Set() {MPI_Comm_rank(comm, &myid); MPI_Comm_size(comm, &nprocs);}
  PetscErrorCode PrepLog();
  void SetDimension(short numDims)
    { this->numDims = numDims; int pow2 = pow(2,numDims);
      B = new MatrixXPS[pow2]; G = new MatrixXPS[pow2];
      GT = new MatrixXPS[pow2]; W = new double[pow2]; }
  PetscErrorCode Clear();

  // Printing information
  PetscErrorCode MeshOut ( );
  PetscErrorCode MeshOut ( TopOpt *topOpt );
  PetscErrorCode StepOut ( const double &f, const VectorXPS &cons,
                           int it, long nactive );
  PetscErrorCode ResultOut ( int it );
  PetscErrorCode PrintVals ( char *name_suffix );

  // Mesh Creation
  PetscErrorCode RecFilter ( PetscInt *first, PetscInt *last, double *dx, double R,
                             ArrayXPI Nel, ArrayXPI &I, ArrayXPI &J,
                             ArrayXPS &K, PetscScalar nonzeros=0 );
  PetscErrorCode Assemble_Filter( Mat &Matrix, ArrayXPI &I, ArrayXPI &J,
                                  ArrayXPS &K, bool scale );
  PetscErrorCode LoadMesh ( VectorXPS &xIni );
  PetscErrorCode CreateMesh ( VectorXPS dimensions, ArrayXPI Nel, double Rmin,
                              double Rmax, bool Reorder_Mesh, 
                              PetscInt mg_levels, PetscInt min_size );
  PetscErrorCode Create_Interpolations ( PetscInt *first, PetscInt *last, ArrayXPI Nel,
                                         ArrayXPI *I, ArrayXPI *J, ArrayXPS *K,
                                         ArrayXPI *cList, PetscInt mg_levels );
  PetscErrorCode Create_Interpolation ( ArrayXPI &first, ArrayXPI &last, ArrayXPI &Nf,
                                        ArrayXPI &I, ArrayXPI &J, ArrayXPS &K );
  PetscErrorCode Assemble_Interpolation ( ArrayXPI *I, ArrayXPI *J, ArrayXPS *K,
                         ArrayXPI *cList, PetscInt mg_levels, PetscInt min_size );
  PetscErrorCode ApplyDomain ( Eigen::Array<bool, -1, 1> elemValidity, int padding,
                               int nInterfaceNodes,
                               ArrayXPI &MinFI, ArrayXPI &MinFJ, ArrayXPS &MinFK,
                               ArrayXPI &MaxFI, ArrayXPI &MaxFJ, ArrayXPS &MaxFK,
                               ArrayXPI *I, ArrayXPI *J, ArrayXPS *K,
                               ArrayXPI *cList, int &mg_levels );
  idx_t ReorderParMetis ( bool Reorder_Mesh, 
                          ArrayXPI &MinFI, ArrayXPI &MinFJ, ArrayXPS &MinFK,
                          ArrayXPI &MaxFI, ArrayXPI &MaxFJ, ArrayXPS &MaxFK,
                          idx_t nparts = 0, idx_t ncommonnodes = 0,
                          real_t *tpwgts = NULL, real_t *ubvec = NULL,
                          idx_t *opts = NULL, idx_t ncon = 1, idx_t *elmwgt = NULL,
                          idx_t wgtflag = 0, idx_t numflag = 0 );

  PetscErrorCode ElemDist ( Eigen::Array<idx_t, -1, 1> &partition,
                            ArrayXPI &MinFI, ArrayXPI &MinFJ, ArrayXPS &MinFK,
                            ArrayXPI &MaxFI, ArrayXPI &MaxFJ, ArrayXPS &MaxFK );
  PetscErrorCode GetElemNumbers ( PetscInt ghostStart, PetscInt ghostEnd,
                                  ArrayXPI &newElemNumber, ArrayXPI &allElemNumber );
  PetscErrorCode NodeDist ( ArrayXPI *I, ArrayXPI *J, ArrayXPS *K,
                            ArrayXPI *cList, int mg_levels );
  PetscErrorCode Expand_Elem();
  PetscErrorCode Expand_Node();
  PetscErrorCode Initialize_Vectors();
  PetscErrorCode Localize();

  // Finite Elements
  PetscErrorCode FEInitialize ( );
  PetscErrorCode FESolve ( );
  PetscErrorCode FEAssemble( );
  PetscErrorCode MatIntFnc ( const VectorXPS &y );
  // Apply filter for chain rule
  PetscErrorCode Chain_Filter(Vec dfdE, Vec dfdV);

private:
  MatrixXPS LocalK ( PetscInt el );
  PetscErrorCode Calc_Strain_Energy( ArrayXPS &energy );
  Eigen::ArrayXXd GaussPoints( );
  MatrixXPS dN(double *gaussPoint);
  void AssignB(MatrixXPS &dNdx, MatrixXPS &B);
  void AssignG(MatrixXPS &dNdx, MatrixXPS &G, MatrixXPS &GT);

};

#endif // TopOpt_H_INCLUDED
