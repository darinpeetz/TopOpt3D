#ifndef TopOpt_H_INCLUDED
#define TopOpt_H_INCLUDED

#include <slepceps.h>
#include <vector>
#include <Eigen/Eigen>
#include "MMA.h"
#include "Functions.h"
/*extern "C"
{
  #include "parmetis.h"
}*/
typedef int idx_t;

typedef Eigen::Array<PetscInt, -1, -1> ArrayXXPI;
typedef Eigen::Array<PetscInt, -1,  1> ArrayXPI;
typedef Eigen::Array<PetscInt, -1, -1, Eigen::RowMajor> ArrayXXPIRM;
typedef Eigen::Matrix<PetscScalar, -1, -1, Eigen::RowMajor> MatrixXdRM;
typedef Eigen::Array<PetscScalar, -1, 1> ArrayXPS;
#define MPI_PETSCINT MPIU_INT

enum BCTYPE { SUPPORT, LOAD, MASS, SPRING, OTHER };
enum MATINT { SIMP, SIMP_CUT, SIMP_LOGISTIC, SIMP_SMOOTH };

/// Class used to keep track of filter information before assembly
/// Implementation is available in Filter.cpp
class FilterArrays
{
public:
  PetscInt nElem;
  ArrayXXPIRM elements;
  Eigen::Array<bool, -1, 1> modified;
  Eigen::ArrayXd distances;

  void Truncate(PetscInt nElem)
  {
    this->nElem = nElem;
    elements.conservativeResize(nElem,2);
    modified.conservativeResize(nElem);
    distances.conservativeResize(nElem);
  }

  void Reset(PetscInt nElem)
  {
    this->nElem = nElem;
    elements.resize(nElem, 2); modified.setZero(nElem); distances.resize(nElem);
  }
  void Clear()
  {
    Reset(0);
  }

};

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
  //Total number of nodes, elements, and edges
  PetscInt nNode, nElem, nEdges;
  //Number of local nodes and elements
  PetscInt nLocElem, nLocNode;
  //Global numbering of elements and nodes stored locally
  ArrayXPI gElem, gNode;
  //Element sizes in m^numDims
  Eigen::VectorXd elemSize;
  //Size of Edges in m^(numDims-1)
  Eigen::VectorXd edgeSize;
  //Elements on each edge
  ArrayXXPIRM edgeElem;
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
  Eigen::MatrixXd d;
  // Element Jacobian
  double detJ;
  //B and G matrices for assembling stiffness matrices
  Eigen::MatrixXd *B, *G, *GT;
  //Integration point weights
  double *W;
  //Indices of local k matrix for constructing global K matrix
  std::vector<double> k;
  //Triplet information for assembling stiffness matrix
  std::vector<PetscInt> i, j, e;
  //Vector of values for individual elements
  std::vector<Eigen::MatrixXd> ke;
  //Force vector
  Vec F;
  //Vector of displacements from fem problem
  Vec U;
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
  double penal, pmin, pmax, pstep;
  //Filter Matrix
  Mat P;
  //Material Interpolation type
  MATINT interpolation;
  std::vector<PetscScalar> interp_param;
  //Material Interpolation Values
  Vec V, dVdy, E, dEdy, Es, dEsdy;
  //Raw densities and filtered densities, rho = P*x
  Vec x, rho;
  //Active vs. passive elements
  Eigen::Array<bool, -1, 1> active;
  //For normalizing perimeter
  PetscScalar PerimNormFactor;
  //Eigenvectors
  Eigen::MatrixXd bucklingShape, dynamicShape;
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
  PetscErrorCode Def_Param(MMA *optmma, Eigen::VectorXd &Dimensions,
                 ArrayXPI &Nel, double &Rfactor, bool &Normalization,
                 bool &Reorder_Mesh, PetscInt &mg_levels, PetscInt &min_size);
  PetscErrorCode Get_CL_Options();
  PetscErrorCode Set_Funcs();
  PetscErrorCode Domain(Eigen::ArrayXXd &Points, const Eigen::VectorXd &Box,
              Eigen::Array<bool, -1, 1> &elemValidity);
  PetscErrorCode Def_BC();
  PetscErrorCode Set_BC(Eigen::ArrayXd center, Eigen::ArrayXd radius,
              Eigen::ArrayXXd limits, Eigen::ArrayXd values, BCTYPE TYPE);

  // Basic methods
  void MPI_Set() {MPI_Comm_rank(comm, &myid); MPI_Comm_size(comm, &nprocs);}
  PetscErrorCode PrepLog();
  void SetDimension(short numDims)
    { this->numDims = numDims; int pow2 = pow(2,numDims);
      B = new Eigen::MatrixXd[pow2]; G = new Eigen::MatrixXd[pow2];
      GT = new Eigen::MatrixXd[pow2]; W = new double[pow2]; }
  PetscErrorCode Clear();

  // Printing information
  PetscErrorCode MeshOut ( );
  PetscErrorCode MeshOut (TopOpt *topOpt);
  PetscErrorCode StepOut ( const double &f, const Eigen::VectorXd &cons,
                           int it, long nactive );
  PetscErrorCode ResultOut ( int it );
  PetscErrorCode PrintVals ( char *name_suffix );

  // Mesh Creation
  void RecFilter ( PetscInt *first, PetscInt *last, double *dx, double R,
                   ArrayXPI Nel, FilterArrays &filterArrays );
  PetscErrorCode LoadMesh(Eigen::VectorXd &xIni);
  PetscErrorCode CreateMesh ( Eigen::VectorXd dimensions, ArrayXPI Nel, double R,
                   bool Reorder_Mesh, PetscInt mg_levels, PetscInt min_size );
  PetscErrorCode Create_Interpolations( PetscInt *first, PetscInt *last,
                      ArrayXPI Nel, ArrayXPI *I, ArrayXPI *J, ArrayXPS *K,
                      ArrayXPI *cList, PetscInt mg_levels );
  PetscErrorCode Create_Interpolation ( ArrayXPI &first, ArrayXPI &last,
              ArrayXPI &Nf, ArrayXPI &I, ArrayXPI &J, ArrayXPS &K );
  PetscErrorCode Assemble_Interpolation ( ArrayXPI *I, ArrayXPI *J, ArrayXPS *K,
                         ArrayXPI *cList, PetscInt mg_levels, PetscInt min_size );
  PetscErrorCode Edge_Info ( PetscInt *first, PetscInt *last, double *dx );
  PetscErrorCode ApplyDomain( Eigen::Array<bool, -1, 1> elemValidity, int padding,
                    int nInterfaceNodes, FilterArrays &filterArrays,
                    ArrayXPI *I, ArrayXPI *J, ArrayXPI *cList, int mg_levels );
  idx_t ReorderParMetis( FilterArrays &filterArrays, bool Reorder_Mesh,
                  idx_t nparts = 0, idx_t ncommonnodes = 0, double *tpwgts = NULL,
                  double *ubvec = NULL, idx_t *opts = NULL, idx_t ncon = 1,
                  idx_t *elmwgt = NULL, idx_t wgtflag = 0, idx_t numflag = 0 );

  PetscErrorCode ElemDist(FilterArrays &filterArrays,
                          Eigen::Array<idx_t, -1, 1> &partition);
  PetscErrorCode NodeDist(ArrayXPI *I, ArrayXPI *J, ArrayXPS *K,
                          ArrayXPI *cList, int mg_levels);
  PetscErrorCode Expand_Elem();
  PetscErrorCode Expand_Node();
  PetscErrorCode Initialize_Vectors();
  PetscErrorCode Localize();

  // Finite Elements
  PetscErrorCode FEInitialize ( );
  PetscErrorCode FESolve ( );
  PetscErrorCode FEAssemble( );
  PetscErrorCode MatIntFnc ( const Eigen::VectorXd &y );

private:
  Eigen::MatrixXd LocalK ( PetscInt el );
  Eigen::ArrayXXd GaussPoints( );
  Eigen::MatrixXd dN(double *gaussPoint);
  void AssignB(Eigen::MatrixXd &dNdx, Eigen::MatrixXd &B);
  void AssignG(Eigen::MatrixXd &dNdx, Eigen::MatrixXd &G,
                       Eigen::MatrixXd &GT);

};

#endif // TopOpt_H_INCLUDED
