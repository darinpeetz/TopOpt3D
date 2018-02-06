#ifndef TopOpt_H_INCLUDED
#define TopOpt_H_INCLUDED

#include <slepceps.h>
#include <vector>
#include <Eigen/Eigen>
#include "MMA.h"
#include "Functions.h"
extern "C"
{
#include "parmetis.h"
}

typedef Eigen::Array<PetscInt, -1, -1> ArrayXXPI;
typedef Eigen::Array<PetscInt, -1,  1> ArrayXPI;
typedef Eigen::Array<PetscInt, -1, -1, Eigen::RowMajor> ArrayXXPIRM;
typedef Eigen::Matrix<PetscScalar, -1, -1, Eigen::RowMajor> MatrixXdRM;
typedef Eigen::Array<PetscScalar, -1, 1> ArrayXPS;
#define MPI_PETSCINT MPIU_INT

enum BCTYPE { SUPPORT, LOAD, MASS, SPRING, OTHER };

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

  MPI_Comm comm;                               //MPI communicator
  int myid;                                    //Rank of this process
  int nprocs;                                  //Total number of processes
  std::string filename;                        //Input file name
  int verbose;                                 //How much information to print
  FILE* output;                                //File for outputing information
  std::string folder;                          //Location of files for restart

  /// Mesh variables
  short numDims;                               //Dimensionality of problem
  MatrixXdRM node;                             //Nodal coordinate
  ArrayXXPIRM element;                         //Element Node numbers
  ArrayXPI elmdist;                            //How elements are distributed on processes
  ArrayXPI nddist;                             //How nodes are distributed on processes
  PetscInt nNode, nElem, nEdges;               //Total number of nodes, elements, and edges
  PetscInt nLocElem, nLocNode;                 //Number of local nodes and elements
  ArrayXPI gElem, gNode;                       //Global numbering of elements and nodes stored locally
  Eigen::VectorXd elemSize;                    //Element sizes in m^numDims
  Eigen::VectorXd edgeSize;                    //Size of Edges in m^(numDims-1)
  ArrayXXPIRM edgeElem;                        //Elements on each edge.
  bool regular;                                // Flag to indicate if all elements are identical

  /// FEM setup variables - only used for FEM initialization
  double Nu0, E0;                              //Element characteristics, E0 in Pa
  double density;                              //Element density in kg/m
  ArrayXPI suppNode;                           //Support node numbers
  Eigen::Array<bool,-1, -1, Eigen::RowMajor> supports; //Boolean indicating if dof is fixed or not
  ArrayXPI springNode;                         //Spring Support node numbers
  Eigen::Array<double, -1, -1, Eigen::RowMajor> springs;//Spring dof stiffnesses in Pa
  ArrayXPI loadNode;                           //Load nodes
  Eigen::Array<double, -1, -1, Eigen::RowMajor> loads;//Loads values in N
  ArrayXPI massNode;                           //Lumped mass nodes
  Eigen::Array<double, -1, -1, Eigen::RowMajor> masses;//mass values in kg

  /// FEM solution variables - used in each FEM iteration
  Eigen::MatrixXd d;                           //Constitutive Matrix
  double detJ;                                 //Element Jacobian
  Eigen::MatrixXd *B, *G, *GT;                 //B and G matrices for assembling stiffness matrices
  double *W;                                   //Integration point weights
  std::vector<double> k;                       //Indices of local k matrix for constructing global K matrix
  std::vector<PetscInt> i, j, e;               //Triplet information for assembling stiffness matrix
  std::vector<Eigen::MatrixXd> ke;             //Vector of values for individual elements
  Vec F;                                       //Force vector
  Vec U;                                       //Vector of displacements from fem problem
  ArrayXPI freeDof;                            //Global indices of free local dofs
  ArrayXPI fixedDof;                           //Global indices of fixed local dofs
  ArrayXPI springDof;                          //Global indices of local dofs with springs
  ArrayXPI springlessDof;                      //Global indices of local dofs without springs
  PetscInt nFreeDof;                           //Total number of free dofs
  PetscInt nFixDof;                            //Total number of fixed dofs
  PetscInt nSpringDof;                         //Number of dofs with springs attached
  ArrayXXPI dofs;                              //Vector containing every dof number
  Mat spK;                                     //Sparse matrix used to store stiffness of springs
  Vec spKVec;                                  //Vector representing diagonal of spring matrix
  Mat K;                                       //Sparse K used to solve fem problem
  std::vector<Mat> PR;                         //Interpolation/Restriction matrices
  std::vector<MPI_Comm> MG_comms;              //Communicator for each level of MG hierarchy
  std::string smoother;                        //Smoother to use with multigrid preconditioners
  Vec MLump;                                   //Storing point masses
  KSP KUF;                                     //The FEM solver context
  KSP dynamicKSP;                              //Solver context for dynamic ST
  KSP bucklingKSP;                             //Solver context for buckling ST
  bool direct;                                 //Flag to use direct instead of iterative solver

  /// Function information
  std::vector<Function_Base*> function_list;
  PetscBool needK, needU;

  /// Optimization variables
  double penal, pmin, pmax, pstep;             //penalization factor information
  Mat P;                                       //Filter Matrix
  Vec V, dVdy, E, dEdy, Es, dEsdy;             //Material Interpolation Values
  Vec x, rho;                                  //Raw densities and filtered densities, rho = P*x
  PetscScalar PerimNormFactor;                 //For normalizing perimeter;
  Eigen::MatrixXd bucklingShape, dynamicShape; //Eigenvectors
  Vec *bucklingDeflate, *dynamicDeflate;       //Deflation spaces for eigenvalue problems
  PetscInt bucklingIt, dynamicIt;              //Number of iterations for eigenvalue problems

  /// Profiling variables
  int funcEvent, FEEvent, UpdateEvent;

  /// Class methods
  // Constructors
  TopOpt() {comm = MPI_COMM_WORLD; MPI_Info(); PrepLog();}
  TopOpt(MPI_Comm comm) {this->comm = comm; MPI_Info(); PrepLog();}
  TopOpt(MPI_Comm comm, short numDims)
 	{this->comm = comm; MPI_Info(); SetDimension(numDims); PrepLog();}
  TopOpt(short numDims) {comm = MPI_COMM_WORLD; MPI_Info(); SetDimension(numDims);}
  // Destructor
  ~TopOpt() {Clear();}

  // Parsing the input file
  PetscErrorCode Def_Param(MMA *optmma, TopOpt *topOpt, Eigen::VectorXd &Dimensions,
                 ArrayXPI &Nel, double &Rfactor, bool &Normalization,
                 bool &Reorder_Mesh, PetscInt &mg_levels, PetscInt &min_size);
  PetscErrorCode Set_Funcs();
  PetscErrorCode Domain(Eigen::ArrayXXd &Points, const Eigen::VectorXd &Box,
              Eigen::Array<bool, -1, 1> &elemValidity);
  PetscErrorCode Def_BC();
  PetscErrorCode Set_BC(Eigen::ArrayXd center, Eigen::ArrayXd radius,
              Eigen::ArrayXXd limits, Eigen::ArrayXd values, BCTYPE TYPE);

  // Basic methods
  void MPI_Info() {MPI_Comm_rank(comm, &myid); MPI_Comm_size(comm, &nprocs);}
  void PrepLog();
  void SetDimension(short numDims)
    { this->numDims = numDims; int pow2 = pow(2,numDims);
      B = new Eigen::MatrixXd[pow2]; G = new Eigen::MatrixXd[pow2];
      GT = new Eigen::MatrixXd[pow2]; W = new double[pow2]; }
  void Clear();

  // Mesh Creation
  void RecFilter ( PetscInt *first, PetscInt *last, double *dx, double R,
                   ArrayXPI Nel, FilterArrays &filterArrays );
  PetscErrorCode LoadMesh(Eigen::VectorXd &xIni);
  PetscErrorCode CreateMesh ( Eigen::VectorXd dimensions, ArrayXPI Nel, double R,
                   bool Reorder_Mesh, PetscInt mg_levels, PetscInt min_size );
  PetscErrorCode Create_Interpolations( PetscInt *first, PetscInt *last, ArrayXPI Nel,
          ArrayXPI *I, ArrayXPI *J, ArrayXPS *K, ArrayXPI *cList, PetscInt mg_levels );
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

  PetscErrorCode ElemDist(FilterArrays &filterArrays, Eigen::Array<idx_t, -1, 1> &partition);
  PetscErrorCode NodeDist(ArrayXPI *I, ArrayXPI *J, ArrayXPS *K, ArrayXPI *cList, int mg_levels);
  PetscErrorCode Expand_Elem();
  PetscErrorCode Expand_Node();
  PetscErrorCode Initialize_Vectors();
  PetscErrorCode Localize();

  // Finite Elements
  PetscErrorCode Initialize ( );
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
