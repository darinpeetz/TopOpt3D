#ifndef P_MMA_H_INCLUDED
#define P_MMA_H_INCLUDED

#include <mpi.h>
#include <Eigen/Eigen>
#include <vector>
#include <iostream>

typedef Eigen::VectorXd evecxd;
typedef unsigned int uint;
typedef unsigned long ulong;

class MMA
{
    public:
        MMA() {miniter = 0; minchange = 0; iter = 0; Set_Defaults(); return;}
        MMA( ulong nvar ) { Set_n(nvar); Initialize(); iter = 0; Set_Defaults(); return;}
        void Set_Defaults() { epsimin = 1e-7; raa0 = 1e-5; mmamove = 0.5;
                              albefa = 0.1; asyinit = 0.5; asyincr = 1.2;
                              asydecr = 0.7; fresh_start = true;}
        void Initialize();
        void Set_Comm( MPI_Comm Mcomm ){ Comm = Mcomm; MPI_Comm_rank(Comm, &myid);
                                         MPI_Comm_size(Comm, &nproc); return; }
        void Set_n( ulong nvar ) { nloc = nvar; MPI_Allreduce(&nloc, &n, 1,
                                   MPI_LONG, MPI_SUM, Comm); Initialize(); return; }
        void Set_m( uint mval );
        //Set lower bound of design variables
        void Set_Lower_Bound( evecxd XM ) { xmin = XM; }
        //Set upper bound of design variables
        void Set_Upper_Bound( evecxd XU ) { xmax = XU; } 
        void Set_Constants( double a0val, double b0val, Eigen::VectorXd &aval,
                            Eigen::VectorXd &cval, Eigen::VectorXd &dval )
                            { a0 = a0val; b0 = b0val; a = aval; c = cval; d = dval; }
        void Set_KKT_Limit( double lim ) { kkttol = lim; }
        void Set_Iter_Limit_Min( uint minimum ) { miniter = minimum; }
        void Set_Iter_Limit_Max( uint maximum ) { maxiter = maximum; }
        void Set_Change_Limit( double minimum ) { minchange = minimum; }
        void Set_Step_Limit( double step ) { mmamove = step; }
        void Set_Init_Values( evecxd xIni ) { xval = xIni; xold1 = xIni;
                                              xold2 = xIni; }
        void Set_It(uint it) {iter = it; return;}
        void Restart() {fresh_start = true; return;}

        ulong  &Get_nloc() {return nloc;}
        ulong  &Get_n()    {return n;}
        uint   &Get_m()    {return m;}
        evecxd &Get_x()    {return xval;}
        uint   &Get_It()   {return iter;}

        int mmasub( Eigen::VectorXd &dfdx, Eigen::VectorXd &g, Eigen::MatrixXd &dgdx );
        bool Check(){ return (Check_Conv() || Check_It()); }
        bool Check_Conv(){ return ((iter > miniter) && (Change<minchange)); }
        bool Check_It() {return ++iter>=maxiter;}
        int Update( Eigen::VectorXd &dfdx, Eigen::VectorXd &g, Eigen::MatrixXd &dgdx );
        void OCsub( Eigen::VectorXd &dfdx, Eigen::VectorXd &g, Eigen::MatrixXd &dgdx );




    private:
        int  DualSolve();
        void DualResidual(Eigen::VectorXd &hvec, Eigen::VectorXd &eta,
                           Eigen::VectorXd &lambda, Eigen::VectorXd &epsvecm);    // For internal calls
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
        void primaldual_subsolve();
        void primaldual_kktcheck( Eigen::VectorXd &dfdx, Eigen::VectorXd &g, Eigen::MatrixXd &dgdx );

        /// MPI Variables
        MPI_Comm Comm;
        int myid, nproc;
        /// Number of GLOBAL and LOCAL design variables
        ulong n, nloc;
        /// Number of constraints
        uint m;
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

#endif // P_MMA_H_INCLUDED
