#ifndef MMA_H_INCLUDED
#define MMA_H_INCLUDED

#include <mpi.h>
#include <Eigen/Eigen>
#include <vector>
#include <algorithm>
#include <iostream>

typedef unsigned int uint;
typedef unsigned long ulong;

class MMA
{
  public:
    // Constructors
    MMA() {miniter = 0; minchange = 0; iter = 0; Set_Defaults(); return;}
    MMA( ulong nvar )
          { Set_n(nvar); Initialize(); iter = 0; Set_Defaults(); return;}
    void Set_Defaults() { epsimin = 1e-7; raa0 = 1e-5; mmamove = 0.5;
                          albefa = 0.1; asyinit = 0.5; asyincr = 1.2;
                          asydecr = 0.7; fresh_start = true;}
    // Preallocate arrays of size n
    void Initialize();
    // Set MPI communicator
    void Set_Comm( MPI_Comm Mcomm ){ Comm = Mcomm; MPI_Comm_rank(Comm, &myid);
                                     MPI_Comm_size(Comm, &nproc); return; }
    // Set number of design variables
    void Set_n( ulong nvar ) { nloc = nvar; MPI_Allreduce(&nloc, &n, 1,
                               MPI_LONG, MPI_SUM, Comm); Initialize(); return; }
    // Set number of constraints
    void Set_m( uint mval );
    // Set lower bound of design variables
    void Set_Lower_Bound( Eigen::VectorXd XM ) { xmin = XM; }
    // Set upper bound of design variables
    void Set_Upper_Bound( Eigen::VectorXd XU ) { xmax = XU; }
    // Set MMA subproblem constants
    void Set_Constants( double a0val, double b0val, Eigen::VectorXd &aval,
                        Eigen::VectorXd &cval, Eigen::VectorXd &dval )
         { a0 = a0val; b0 = b0val; a = aval; c = cval; d = dval; }
    // Set KKT limit for convergence
    void Set_KKT_Limit( double lim ) { kkttol = lim; }
    // Set maximum and minimum number of iterations for convergence
    void Set_Iter_Limit_Min( uint minimum ) { miniter = minimum; }
    void Set_Iter_Limit_Max( uint maximum ) { maxiter = maximum; }
    // Set minimum DV change for convergence
    void Set_Change_Limit( double minimum ) { minchange = minimum; }
    // Set maximum setp size
    void Set_Step_Limit( double step ) { mmamove = step; }
    // Set DV values (for initialization)
    void Set_Values( Eigen::VectorXd xIni ) { xval = xIni; xold1 = xIni; xold2 = xIni; }
    // Set DV values to active or passive
    void Set_Active( std::vector<bool> &active ) { this->active = active;
                 nactive = std::count(active.begin(), active.end(), true); }
    // Set current iteration number
    void Set_It(uint it) {iter = it; return;}
    // Set a flag to check if asymptotes are valid or need to be set to defaults
    void Restart() {fresh_start = true; return;}

    // Get various items from optimizer object
    long            &Get_nloc() {return nloc;}
    ulong           &Get_n()    {return n;}
    int             &Get_m()    {return m;}
    Eigen::VectorXd &Get_x()    {return xval;}
    uint            &Get_It()   {return iter;}

    // Checking convergence
    bool Check(){ return (Check_Conv() || Check_It()); }
    bool Check_Conv(){ return ((iter > miniter) && (Change<minchange)); }
    bool Check_It() {return ++iter>=maxiter;}

    // Update the design variables
    int Update( Eigen::VectorXd &dfdx, Eigen::VectorXd &g, Eigen::MatrixXd &dgdx );


  private:
    // MMA solver
    int MMAsub( Eigen::VectorXd &dfdx, Eigen::VectorXd &g, Eigen::MatrixXd &dgdx );
    // OC solver
    int OCsub( Eigen::VectorXd &dfdx, Eigen::VectorXd &g, Eigen::MatrixXd &dgdx );
    // Auxiliary functions for MMA subsolvers
    int  DualSolve(Eigen::VectorXd &x);
    void DualResidual(Eigen::VectorXd &hvec, Eigen::VectorXd &eta,
                       Eigen::VectorXd &lambda, Eigen::VectorXd &epsvecm);
    void XYZofLam(Eigen::VectorXd &x, Eigen::VectorXd &y, double &z, Eigen::VectorXd &lambda);
    int  DualGrad(Eigen::VectorXd &ux1, Eigen::VectorXd &xl1,
                  Eigen::VectorXd &y, double &z, Eigen::VectorXd &grad);
    int  DualHess(Eigen::VectorXd &ux2, Eigen::VectorXd &xl2,
                  Eigen::VectorXd &ux3, Eigen::VectorXd &xl3,
                  Eigen::VectorXd &x,   Eigen::MatrixXd &Hess);
    void SearchDir(Eigen::MatrixXd &Hess, Eigen::VectorXd &hvec,
                   Eigen::VectorXd &lambda, Eigen::VectorXd &eta,
                   Eigen::VectorXd &dellam, Eigen::VectorXd &deleta,
                   Eigen::VectorXd &epsvec);
    double SearchDis(Eigen::VectorXd &lambda, Eigen::VectorXd &eta,
                     Eigen::VectorXd &dellam, Eigen::VectorXd &deleta);
    void primaldual_subsolve(Eigen::VectorXd &x);
    void primaldual_kktcheck( Eigen::VectorXd &dfdx, Eigen::VectorXd &g, Eigen::MatrixXd &dgdx );

    /// MPI Variables
    MPI_Comm Comm;
    int myid, nproc;
    /// Number of GLOBAL and LOCAL design variables
    ulong n;
    long nloc;
    /// Flags indicating if each variable is subject to optimization or not
    std::vector<bool> active;
    /// Number of LOCAL active variables
    long nactive;
    /// Number of constraints
    int m;
    /// Values from last three iterations
    Eigen::VectorXd xval, xold1, xold2;
    /// Iteration counter
    uint iter;
    /// Flag if moving asymptotes don't have history yet
    bool fresh_start;
    /// Convergence values
    uint maxiter, miniter;
    double kkttol, minchange;
    /// Lower and Upper bound on x
    Eigen::VectorXd xmin, xmax;
    /// MMA constants
    double a0, b0;
    Eigen::VectorXd a, b, c, d;
    double epsimin, raa0, mmamove, albefa, asyinit, asyincr, asydecr;
    /// Asymptotes
    Eigen::VectorXd low, upp;
    /// Active parts of the asymptotes
    Eigen::VectorXd *p_low, *p_upp;
    /// Subproblem variables
    Eigen::VectorXd zzz, factor, alfa, beta, p0, q0;
    Eigen::MatrixXd P, Q;
    Eigen::VectorXd plam, qlam;
    /// Lagrange Multipliers
    Eigen::VectorXd lambda, eta;
    /// Subsolve Returns for primal-dual solver
    Eigen::VectorXd ymma, lamma, xsimma, etamma, mumma, smma;
    double zmma, zet, zetmma;
    /// Residual Values
    Eigen::VectorXd residual;
    double residunorm, residumax;
    /// OC values if I ever get the OC update working right
    double OCMove, OCeta, Change;

};

#endif // MMA_H_INCLUDED
