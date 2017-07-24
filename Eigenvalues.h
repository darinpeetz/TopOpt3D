#ifndef PARLAB_H_INCLUDED
#define PARLAB_H_INCLUDED

#include <Eigen/Eigen>
#include <vector>
#include <math.h>
#include "EigLab.h"
#include "mpi.h"
#include "ParHelp.h"

typedef Eigen::Matrix<long, -1, 1> VectorXLI;

extern "C" void pdsaupd_(MPI_Fint *comm, int *ido, char *bmat, int *n,
    char *which, int *nev, double *tol, double *resid, int *ncv,
    double *v, int *ldv, int *iparam, int *ipntr,
    double *workd, double *workl, int *lworkl, int *info);

extern "C" void pdseupd_(MPI_Fint *comm, int *rvec, char *All, int *select,
    double *d, double *z, int *ldz, double *sigma, char *bmat, int *n,
    char *which, int *nev, double *tol, double *resid, int *ncv, double *v,
    int *ldv, int *iparam, int *ipntr, double *workd, double *workl,
    int *lworkl, int *ierr);

namespace EigLab
{
    template <typename MatrixType, typename VectorType>
    VectorType OP1(MatrixType &A, ParVars *ParV, VectorType y);

    template <typename VectorType>
    void RecoverOP(ParVars *ParV, VectorType &z);

    template <typename VectorType, typename IndexType>
    void RedVec(VectorType &v, IndexType &ind);

    template <typename VectorType, typename IndexType>
    void ExpVec(VectorType &v, IndexType &ind);

    template <typename SparseType, typename MatrixType, typename VectorType>
    int p_eigs(SparseType &A, SparseType &B, ParVars *ParV, std::string eig_mode,
              VectorType &EigenValues, MatrixType &EigenVectors, double *opts = NULL)
              //= MatrixType::Zeros(0,0), int iters = 0, double tol = 0)
    {
        MPI_Fint Fcomm = MPI_Comm_c2f(ParV->get_comm());

        double tol;
        int iters;
        bool BisSPD;
        if (opts == NULL)
        {
            tol = 0; iters = 300; BisSPD = false;
        }
        else
        {
            tol = opts[0]; iters = (int)opts[1]; BisSPD = (bool)opts[2];
        }

        typedef typename SparseType::Scalar Scalar;
        typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;

        // ido must be zero on first call
        int myid = ParV->get_rank();
        int ido = 0;
        int n = (int)A.rows(), nloc = (int)A.cols();

        // Options: "LA", "SA", "SM", "LM", "BE"
        char whch[3] = "LM";

        // Shift if iparam[6] = {3,4,5}, not used if iparam[6] = {1,2}
        RealScalar sigma = 0.0;

        if (eig_mode.length() >= 2 && isalpha(eig_mode[0]) && isalpha(eig_mode[1]))
        {
            eig_mode[0] = toupper(eig_mode[0]);
            eig_mode[1] = toupper(eig_mode[1]);

            //If "SM" is specified, invert and use "LM"
            if (eig_mode.substr(0,2) != "SM")
            {
                whch[0] = eig_mode[0];
                whch[1] = eig_mode[1];
            }
        }
        else
        {
            std::cout << "Don't specify eigenvalue clustering, I never set that up\n";
        }

        char probtype[2] = "G";
        if (B.size() == 0 || BisSPD) {probtype[0] = 'I';}

        // ARPACK mode - 1 for basic problem (or when matrix B is Symmetric Positive Definite/can be decomposed int LLT)
        int mode = (probtype[0] == 'G') + 1;
        if (eig_mode.substr(0,2) == "SM" || !(isalpha(eig_mode[0]) && isalpha(eig_mode[1]))) {mode = 3;}
        // Number of eigenvalues
        int nev = EigenValues.size();
        // Space for ARPACK to store residual
        Scalar *resid = new Scalar[nloc];
        // Number of Lanczos vectors, nev+2 < ncv <= n, 2*nev+1 is recommended
        int ncv = std::min(std::max(2*nev+1, 20), n);
        // The working n x ncv matrix, also store the final eigenvectors (if computed)
        Scalar *v = new Scalar[n*ncv];
        int ldv = n;
        // Working Space
        Scalar *workd = new Scalar[3*n];
        int lworkl = ncv*ncv+8*ncv; // Must be at least this length
        Scalar *workl = new Scalar[lworkl];

        // Some input/output parameters
        int *iparam = new int[11]; //[1] and [5] are obsolete, see pdnaupd.f for the rest
        iparam[0] = 1; // 1 has ARPACK perform shifts, 0 means we do
        if (iters > 0) {iparam[2] = iters;}
        else {iparam[2] = std::max(300, (int)std::ceil(2*n/std::max(ncv,1)));}  //Max iterations
        iparam[6] = mode; //Mode, 1 is standard, 2 is generalized, 3 is shift-and-invert, 4 is buckling

        // Used during reverse communicate to notify where arrays start
        int *ipntr = new int[11];

        // Error codes are returned in here, initial value of 0 indicates a random initial
        // residual vector is used, any other values means resid contains the initial residual
        // vector, possibly from a previous run
        int info = 0;
        Scalar scale = 1.0;

        do
        {
            pdsaupd_(&Fcomm, &ido, probtype, &nloc, whch, &nev, &tol, resid, &ncv,
                    v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);

            if (ido == -1 || ido == 1)
            {
                Scalar *in = workd + ipntr[0] - 1;
                Scalar *out = workd + ipntr[1] - 1;

                if (ido == 1 && mode != 2)
                {
                    Scalar *out2 = workd + ipntr[2] - 1;
                    if ( (mode == 1) && (B.size() == 0) )
                        Eigen::Matrix<Scalar, -1, 1>::Map(out2, nloc) = Eigen::Matrix<Scalar, -1, 1>::Map(in, nloc);
                    else
                    {
                        Eigen::Matrix<Scalar, -1, 1> Y = B * Eigen::Matrix<Scalar, -1, 1>::Map(in, nloc);
                        MPI_Allreduce(MPI_IN_PLACE, Y.data(), n, MPI_DOUBLE, MPI_SUM, ParV->get_comm());
                        for (long i = 0; i < nloc; i++)
                            out2[i] = Y(ParV->mydofs(i));
                    }

                    //in = workd + ipntr[2] - 1;
                }

                if (mode == 1)
                {
                    if (B.size() == 0) // OP = A
                    {
                        Eigen::Matrix<Scalar, -1, 1> Y = A * Eigen::Matrix<Scalar, -1, 1>::Map(in, nloc);
                        MPI_Allreduce(MPI_IN_PLACE, Y.data(), n, MPI_DOUBLE, MPI_SUM, ParV->get_comm());
                        for (long i = 0; i < nloc; i++)
                            out[i] = Y(ParV->mydofs(i));
                    }
                    else // OP = L^-1*A*L^-T
                    {
                        Eigen::Matrix<Scalar, -1, -1> vec = Eigen::Matrix<Scalar, -1, -1>::Map(in, nloc, 1);
                        Eigen::Matrix<Scalar, -1, -1>::Map(out, nloc, 1) = OP1(A, ParV, vec);
                    }
                }
                else if (mode == 2)
                {
                    Eigen::Matrix<Scalar, -1, -1> Y = A * Eigen::Matrix<Scalar, -1, 1>::Map(in, nloc);
                    MPI_Allreduce(MPI_IN_PLACE, Y.data(), n, MPI_DOUBLE, MPI_SUM, ParV->get_comm());
                    for (long i = 0; i < nloc; i++)
                    {
                        Y(i, 0) = Y(ParV->mydofs(i), 0);
                        in[i] = Y(i, 0);
                    }
                    Y.conservativeResize(nloc, 1);

                    // OP = B^{-1} A
                    //
                    ParV->solver.solve(Y);
                    Eigen::Matrix<Scalar, -1, -1>::Map(out, nloc, 1) = Y;
                }
            }
            else if (ido == 2)
            {
                Scalar *in  = workd + ipntr[0] - 1;
                Scalar *out = workd + ipntr[1] - 1;

                if (B.size() == 0)
                    Eigen::Matrix<Scalar, -1, 1>::Map(out, nloc) = Eigen::Matrix<Scalar, -1, 1>::Map(in, nloc);
                else
                {
                    Eigen::Matrix<Scalar, -1, 1> Y = B * Eigen::Matrix<Scalar, -1, 1>::Map(in, nloc);
                    MPI_Allreduce(MPI_IN_PLACE, Y.data(), n, MPI_DOUBLE, MPI_SUM, ParV->get_comm());
                    for (long i = 0; i < nloc; i++)
                        out[i] = Y(ParV->mydofs(i));
                }
            }
        } while (ido != 99);
        // Done creating Ritz vectors, time to post-process

        if (info < 0 && myid == 0)
        {
            std::cout << "\n\n***Error with pdsaupd, error code = " << info;
            std::cout << ". Check documentation of pdsaupd***\n\n";
        }
        if (info > 0 && myid == 0)
        {
            std::cout << "\n\n***pdsaupd finished unusually with code = " << info;
            std::cout << ". Check documentation of pdsaupd***\n\n";
        }
        else
        {
            int rvec = (EigenVectors.size() != 0);
            char howmny[2] = "A";
            int *select = new int[ncv]; //For picking eigenvalues to return, not used

            pdseupd_(&Fcomm, &rvec, howmny, select, EigenValues.data(), v, &ldv, &sigma,
                    probtype, &nloc, whch, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr,
                    workd, workl, &lworkl, &info);

            if (info != 0 && myid == 0)
            {
                std::cout << "\n\n***Error with pdseupd, error code = " << info;
                std::cout << ". Check documentation of pdseupd***\n\n";
            }

            if (rvec)
            {
                //Doesn't properly allocate dofs
                //EigenVectors = MatrixType::Map(v, n, nev) / scale;
                for (int i=0; i<nev; i++)
                {
                    for (int j=0; j<nloc; j++)
                        EigenVectors(j,i) = v[i*n+j] / scale;
                }

                if (mode == 1)
                {
                    RecoverOP(ParV, EigenVectors);
                }
            }
            delete select;
        }

        delete resid;
        delete v;
        delete workd;
        delete workl;
        delete iparam;
        delete ipntr;

        return info;
    }

    // Operation inv(L)*A*inv(L`)
    template <typename MatrixType, typename VectorType>
    VectorType OP1(MatrixType &A, ParVars *ParV, VectorType y)
    {
        long *iparm = ParV->solver.get_iparm();
        iparm[IPARM_TRANSPOSE_SOLVE] = API_SOLVE_FORWARD_ONLY;
        ParV->solver.solve(y);
        iparm[IPARM_TRANSPOSE_SOLVE] = API_SOLVE_BACKWARD_ONLY;

        Eigen::VectorXd x = A*y;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE,x.data(),x.size(),MPI_DOUBLE,
                      MPI_SUM, ParV->get_comm());
        //MPI_Allreduce(y.data(), x.data(), y.size(), MPI_DOUBLE,
        //              MPI_SUM, ParV->get_comm());

        for (long i = 0; i < ParV->mydofs.rows(); i++)
            y(i) = x(ParV->mydofs(i));

        ParV->solver.solve(y);

        return y;
    }

    // Recover Vector from Ritz vectors when B = LL^T
    template <typename MatrixType>
    void RecoverOP(ParVars *ParV, MatrixType &z)
    {

        long *iparm = ParV->solver.get_iparm();
        iparm[IPARM_TRANSPOSE_SOLVE] = API_SOLVE_BACKWARD_ONLY;
        ParV->solver.solve(z);
        iparm[IPARM_TRANSPOSE_SOLVE] = API_NO;

        return;
    }

    // Reduce vector from global to local form
    template <typename MatrixType, typename IndexType>
    void RedVec(MatrixType &v, IndexType &ind)
    {
        for (long i = 0; i < ind.rows(); i++)
            v.row(i) = v.row(ind(i));
        v.conservativeResize(ind.rows(),v.cols());
        return;
    }

    // Expand vector from local to global form
    template <typename MatrixType, typename IndexType>
    void ExpVec(MatrixType &v, IndexType &ind)
    {
        v.conservativeResize(ind.rows(),v.cols());
        for (long i = ind.rows()-1; i >= 0; i--)
            v.row(ind(i)) = v.row(i);
        return;
    }

/************************************************************************/
/**                     JACOBI-DAVIDSON ROUTINES                       **/
/************************************************************************/
    template <typename MatrixType>
    class JDSYM
    {
    public:
        typedef typename MatrixType::Scalar Scalar;
        typedef typename MatrixType::Index Index;
        JDSYM();    //Default Constructor

        long calc(MatrixType &A, MatrixType &B,
             Eigen::Matrix<Scalar, -1, -1> &V,
             ParVars *ParV); // Main Function

        /** Special Function Calls **/
        int calc(MatrixType &A, MatrixType &B,
             Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &V,
             short k, ParVars *ParV)
        { set_kmax(k); set_jmin(2*k); set_jmax(4*k);
          calc(A, B, V, ParV); return 0; }

        int calc(MatrixType &A, MatrixType &B,
             Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &V,
             std::string sigma, ParVars *ParV)
        { set_type(sigma); calc(A, B, V, ParV); return 0; }

        int calc(MatrixType &A, MatrixType &B,
             Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &V,
             double sigma, ParVars *ParV)
        { set_tau(sigma); set_type("SH"); calc(A, B, V, ParV); return 0; }

        int calc(MatrixType &A, MatrixType &B,
             Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &V,
             short k, std::string sigma, ParVars *ParV)
        { set_type(sigma); set_kmax(k); set_jmin(2*k);
          set_jmax(4*k); calc(A, B, V, ParV); return 0; }

        int calc(MatrixType &A, MatrixType &B,
             Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &V,
             short k, double sigma, ParVars *ParV)
        { set_type("SH"); set_tau(sigma); set_kmax(k); set_jmin(2*k);
          set_jmax(4*k); calc(A, B, V, ParV); return 0; }

        int calc(MatrixType &A, MatrixType &B, ParVars *ParV)
        { Eigen::Matrix<Scalar,-1,-1> V = InitV(A.cols()); calc(A, B, V, ParV); return 0; }

        int calc(MatrixType &A, MatrixType &B, short k, ParVars *ParV)
        { set_kmax(k); set_jmin(2*k); set_jmax(4*k);
          Eigen::Matrix<Scalar,-1,-1> V = InitV(A.cols()); calc(A, B, V, ParV); return 0; }

        int calc(MatrixType &A, MatrixType &B, std::string sigma, ParVars *ParV)
        { set_type(sigma); Eigen::Matrix<Scalar,-1,-1> V = InitV(A.cols());
          calc(A, B, V, ParV); return 0; }

        int calc(MatrixType &A, MatrixType &B, double sigma, ParVars *ParV)
        { set_tau(sigma); set_type("SH");
          Eigen::Matrix<Scalar,-1,-1> V = InitV(A.cols()); calc(A, B, V, ParV); return 0; }

        int calc(MatrixType &A, MatrixType &B, short k, std::string sigma, ParVars *ParV)
        { set_kmax(k); set_jmin(2*k); set_jmax(4*k); set_type(sigma);
          Eigen::Matrix<Scalar,-1,-1> V = InitV(A.cols()); calc(A, B, V, ParV); return 0; }

        int calc(MatrixType &A, MatrixType &B, short k, double sigma, ParVars *ParV)
        { set_kmax(k); set_jmin(2*k); set_jmax(4*k); set_tau(sigma); set_type("SH");
          Eigen::Matrix<Scalar,-1,-1> V = InitV(A.cols()); calc(A, B, V, ParV); return 0; }

        /** Set Parameter Functions **/
        void set_tau(double val) {tau = val; return;}
        void set_type(std::string val) {type = val; return;}
        void set_kmax(short val) {kmax = val; jmin = std::max(2*kmax,(int)jmin);
            jmax = std::max(4*kmax,(int)jmax); itmax = std::max((long)50*val,itmax); return;}
        void set_eps(double val) {eps = val; return;}
        void set_epstr(double val) {epstr = val; return;}
        void set_jmin(short val) {jmin = val; return;}
        void set_jmax(short val) {jmax = val; return;}
        void set_itmax(long val) {itmax = val; return;}
        void set_itmaxin(long val) {itmaxin = val; return;}
        void set_gamma(double val) {gamma = val; return;}

        Eigen::Matrix<Scalar, -1, 1> eigenvalues() {return lambda.segment(0,k);}
        Eigen::Matrix<Scalar, -1, -1> eigenvectors() {return Q.block(0,0,m,k);}
        long outiters() {return it;}
        long initers() {return init;}



    protected:
        Eigen::Matrix<Scalar, -1, -1> InitV( long nv )
        {Eigen::Matrix<Scalar,-1,-1> V = Eigen::Matrix<Scalar,-1,-1>::Random(nv, jmin); return V;}
        void sorteig(Eigen::Matrix<Scalar, -1, -1> &W, Eigen::Matrix<Scalar, -1, 1> &S);
        void Mgsm(Eigen::Matrix<Scalar, -1, 1> &u, ParVars *ParV);
        double Icgsm(const Eigen::Matrix<Scalar, -1, -1> &V, const MatrixType &M,
                     Eigen::Matrix<Scalar, -1, 1> &u, ParVars *ParV);
        template <typename Vtype>
        void MV(const MatrixType &M, Vtype &V, ParVars *ParV);
        void gmres(const MatrixType &A, Eigen::Matrix<Scalar, -1, 1>  &v,
                   double tol, short MaxIt, ParVars *ParV);
        void QkFQb(Eigen::Matrix<Scalar, -1, 1> &v, ParVars *ParV);
        void PrepPRC(const MatrixType &A, const MatrixType &B, ParVars *ParV);
        void PRC(Eigen::Matrix<Scalar, -1, 1> &v, ParVars *ParV);

double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Q, Qb, Qk, F;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> lambda, Jac;

        long init, it, itmax, itmaxin, istep, n, m;
        short j, jmin, jmax, k, kmax;
        double tau, eps, epstr, gamma;
        std::string type;

    };

    /** Default Constructor **/
    template <typename MatrixType>
    JDSYM<MatrixType>::JDSYM()
    {
        type = "LM";
        tau = 1;
        kmax = 6;
        jmin = 2*kmax;
        jmax = 4*kmax;
        eps = 1e-8;
        epstr = 1e-3;
        itmax = 300;
        itmaxin = 50;
        gamma = 2;
        init = 0;
        it = 0;
    }

    /***********************************/
    /** Main Jacobi-Davidson Function **/
    /***********************************/
    template <typename MatrixType>
    long JDSYM<MatrixType>::calc(MatrixType &A, MatrixType &B,
             Eigen::Matrix<Scalar, -1, -1> &V,
             ParVars *ParV)
    {
        /** Profiling Variables **/
        double ta = MPI_Wtime(), tb = 0, tc1 = 0, tc2 = 0, tc3 = 0, td = 0, te = 0, tf = 0, tg = 0, th = 0, ti = 0, ttemp;
        t1 = 0; t2 = 0; t3 = 0; t4 = 0; t5 = 0; t6 = 0; t7 = 0; t8 = 0;
        t9 = 0; t10 = 0; t11 = 0; t12 = 0; t13 = 0; t14 = 0; t15 = 0;

        n = A.rows(); m = A.cols();
        long istep = 0;
        double rnorm, sigma;

        /** Prepare Linear Solver Preconditioner **/
        PrepPRC(A, B, ParV);

        /** Normalize first search vector **/
        Q  = V.col(0);
        MV(B, Q, ParV);
        rnorm = (V.col(0).transpose()*Q)(0,0);
        MPI_Allreduce(MPI_IN_PLACE,&rnorm,1,MPI_DOUBLE,MPI_SUM,ParV->get_comm());
        rnorm = sqrt(rnorm);
        V.col(0) = V.col(0)/rnorm;

        /** Orthonormalize remaining search vectors **/
        for (long i = 1; i < V.cols(); i++)
        {
            j = i;
            Eigen::Matrix<Scalar,-1, 1> vec = V.col(i);
            rnorm = Icgsm( V, B, vec, ParV );
            V.col(i) = vec/rnorm;
        }

        /** Initialize Remaining Variables **/
        j = V.cols(); k = 0;
        V.conservativeResize(m, jmax);
        Eigen::Matrix<Scalar, -1, -1> G = Eigen::Matrix<Scalar, -1, -1>::Zero(jmax,jmax);
        Eigen::Matrix<Scalar, -1, -1> W(jmax,jmax), temp; // Matrices
        Eigen::Matrix<Scalar, -1, 1> S(jmax), u, ub, uk, r; // Vectors
        MatrixType sysmat = A;
        Q = V.block(0,0,m,j);
        MV(A, Q, ParV);

        MPI_Request Greqs[jmax];
        MPI_Status Gstats[jmax];
        for (long i = 0; i < j; i++)
        {
            G.block(0,i,j,1) = V.block(0,0,m,j).transpose()*Q.col(i);
            MPI_Iallreduce(MPI_IN_PLACE,G.data()+jmax*i,j,MPI_DOUBLE,MPI_SUM,ParV->get_comm(),Greqs+i);
        }
        MPI_Waitall(j,Greqs,Gstats);

        Q.setZero(m,kmax); Qb = Q; Qk = Q;
        F.setZero(kmax,kmax);
        lambda.setZero(kmax);
ta = MPI_Wtime() - ta;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, -1, -1> > eig(jmax);
        /** Outer Jacobi-Davidson Loop **/
        while (it < itmax)
        {
ttemp = MPI_Wtime();
            /** Projected Eigenvalues **/
            eig.compute(G);
            W = eig.eigenvectors();
            S = eig.eigenvalues();
            sorteig(W, S);
tb += MPI_Wtime() - ttemp; ttemp = MPI_Wtime();
            /** Ritz Approximation **/
   Cont:    ttemp = MPI_Wtime(); u = V.block(0,0,m,j)*W.block(0,0,j,1); lambda(k) = S(0);
            ub = u; MV(B, ub, ParV);
            uk = ub; PRC(uk, ParV);
            r = u;
tc1 += MPI_Wtime() - ttemp; ttemp = MPI_Wtime();
            MV(A, r, ParV);
            r -= lambda(k)*ub; rnorm = r.squaredNorm();
            MPI_Allreduce(MPI_IN_PLACE, &rnorm, 1, MPI_DOUBLE, MPI_SUM, ParV->get_comm());
            rnorm = sqrt(rnorm);
tc2 += MPI_Wtime() - ttemp; ttemp = MPI_Wtime();
            Q.col(k) = u; Qk.col(k) = uk; Qb.col(k) = ub;
            temp = Qb.block(0,0,m,k+1).transpose()*uk;
            MPI_Allreduce(MPI_IN_PLACE, temp.data(), temp.size(), MPI_DOUBLE,
                          MPI_SUM, ParV->get_comm());
            F.block(k,0,1,k+1) = temp.transpose();
            F.block(0,k,k+1,1) = temp;
tc3 += MPI_Wtime() - ttemp; ttemp = MPI_Wtime();
            /** Convergence **/
            if ( rnorm < eps && (j>1 || k == kmax-1) )
            {
ttemp = MPI_Wtime();
                if (type != "SH")
                    {tau = lambda(k);}
 if (ParV->get_rank() == 0) {std::cout << it << "\n";}
                V.block(0,0,m,j-1) = V.block(0,0,m,j)*W.block(0,1,j,j-1);
                for (short i = 0; i < j-1; i++) {S(i) = S(i+1);}
                G.setZero(jmax,jmax); G.block(0,0,j-1,j-1) = S.segment(0,j-1).asDiagonal();
                W.block(0,0,j-1,j-1).setIdentity(j-1,j-1);
                j--; k++; istep = 1;
td += MPI_Wtime() - ttemp;
                if (k == kmax)
                {
                    if (ParV->get_rank() == 0) {std::cout << "Initialization:\t" << ta << " secs on process " << ParV->get_rank() << "\n";
                    std::cout << "Projected Eigenproblem:\t" << tb << " secs on process " << ParV->get_rank() << "\n";
                    std::cout << "Ritz Approximation:\t" << tc1 << "\t" << tc2 << "\t" << tc3 << " secs on process " << ParV->get_rank() << "\n";
                    std::cout << "Convergence:\t" << td << " secs on process " << ParV->get_rank() << "\n";
                    std::cout << "Correction Setup:\t" << te << " secs on process " << ParV->get_rank() << "\n";
                    std::cout << "Correction Equation:\t" << tf << " secs on process " << ParV->get_rank() << "\n";
                    std::cout << "Update1:\t" << tg << " secs on process " << ParV->get_rank() << "\n";
                    std::cout << "Update2:\t" << th << " secs on process " << ParV->get_rank() << "\n";
                    std::cout << "Update3:\t" << ti << " secs on process " << ParV->get_rank() << "\n";
                    std::cout << "Linsolver breakdown:\n1) " << t1 << "\t2) " << t2 << "\t3) " << t3 << "\t4) " << t4 << "\n5) " << t5 << "\t6) " << t6 << "\t7) " << t7;
                    std::cout << "\t8) " << t8 << "\n9) " << t9 << "\t10) " << t10 << "\t11) " << t11 << "\t12) " << t12 << "\n13) " << t13 << "\t14) " << t14 << "\t15) " << t15 << "\n";}
                    return 0;
                }
                goto Cont;
            }
ttemp = MPI_Wtime();
            if (j == jmax)
            {
                j = jmin;
                V.block(0,0,m,j) = V.block(0,0,m,jmax)*W.block(0,0,jmax,j);
                G.setZero(jmax,jmax); G.block(0,0,j,j) = S.segment(0,j).asDiagonal();
                W.setIdentity(jmax,jmax);
            }
            if (rnorm < epstr || (type != "SH" && k == 0) )
                sigma = lambda(k);
            else
                sigma = tau;

            for (long i = 0; i < A.nonZeros(); i++) {sysmat.valuePtr()[i] = A.valuePtr()[i] - sigma*B.valuePtr()[i];}
            r = -r;
te += MPI_Wtime() - ttemp; ttemp = MPI_Wtime();
            gmres(sysmat, r, std::max( pow(gamma,-istep), eps), std::min( (long)(1.5*istep), itmaxin), ParV);
tf += MPI_Wtime() - ttemp; ttemp = MPI_Wtime();
            // r becomes the update z
            Mgsm(r, ParV);
tg += MPI_Wtime() - ttemp; ttemp = MPI_Wtime();
            rnorm = Icgsm(V, B, r, ParV); if (rnorm != 0) {r = r/rnorm;}
th += MPI_Wtime() - ttemp; ttemp = MPI_Wtime();
            V.col(j) = r;
            MV(A, r, ParV);
            temp = V.block(0,0,m,j+1).transpose()*r;
            MPI_Allreduce(MPI_IN_PLACE, temp.data(), temp.size(), MPI_DOUBLE,
                          MPI_SUM, ParV->get_comm());
ti += MPI_Wtime() - ttemp;
            G.block(j,0,1,j+1) = temp.transpose();
            G.block(0,j,j+1,1) = temp;
            j++; istep++; it++;
        }

        return 0;
    }

    template <typename MatrixType>
    void JDSYM<MatrixType>::sorteig(Eigen::Matrix<Scalar, -1, -1> &W, Eigen::Matrix<Scalar, -1, 1> &S)
    {
        Eigen::Matrix<Scalar, -1, -1> temp;
        VectorXLI i;
        if (type == "SH")
        {
            temp = (S.array()-tau).matrix().cwiseAbs();
            i = EigLab::gensort(temp);
        }
        else if (type == "LM")
        {
            temp = S.cwiseAbs();
            i = EigLab::gensort(temp);
            i.reverseInPlace();
        }
        else if (type == "SM")
        {
            temp = S.cwiseAbs();
            i = EigLab::gensort(temp);
        }
        else if (type == "LA")
        {
            temp = S;
            i = EigLab::gensort(temp);
            i.reverseInPlace();
        }
        else if (type == "SA")
        {
            temp = S;
            i = EigLab::gensort(temp);
        }

        temp.resize(j, j+1);
        for (short ii = 0; ii < j; ii++)
        {
            temp(ii,0) = S(i(ii));
            for (short jj = 0; jj < j; jj++)
            {
                temp(jj, ii+1) = W(jj, i(ii));
            }
        }
        S.segment(0,j) = temp.col(0);
        W.block(0,0,j,j) = temp.block(0, 1, j, j);
        return;
    }

    /** Modified M-Orthogonal Gram-Schmidt Orthogonalization **/
    template <typename MatrixType>
    void JDSYM<MatrixType>::Mgsm(Eigen::Matrix<Scalar, -1, 1> &u, ParVars *ParV)
    {
        Eigen::Matrix<Scalar, -1, 1> r = Qb.block(0,0,m,k+1).transpose()*u;
        MPI_Allreduce(MPI_IN_PLACE, r.data(), r.size(), MPI_DOUBLE, MPI_SUM, ParV->get_comm());
        u -= Q.block(0,0,m,k+1)*r;

        return;
    }

    /** Iterative Classical M-Orthogonal Gram-Schmidt Orthogonalization **/
    template <typename MatrixType>
    double JDSYM<MatrixType>::Icgsm(const Eigen::Matrix<Scalar, -1, -1> &V, const MatrixType &M,
                 Eigen::Matrix<Scalar, -1, 1> &u, ParVars *ParV)
    {
        double alpha = 0.5;
        short itmax = 3, it = 0;

        Eigen::Matrix<Scalar, -1, 1> um = u, h(V.cols());
        MV(M, um, ParV);

        double r1 = u.dot(um);
        MPI_Allreduce(MPI_IN_PLACE, &r1, 1, MPI_DOUBLE, MPI_SUM, ParV->get_comm());
        r1 = sqrt(r1);
        double r0 = (r1/alpha)/alpha;

        while ( (r1 <= alpha*r0) && (it < itmax) )
        {
            r0 = r1; it++;
            h.segment(0,j) = V.block(0,0,m,j).transpose()*um;
            MPI_Allreduce(MPI_IN_PLACE, h.data(), j, MPI_DOUBLE,
                          MPI_SUM, ParV->get_comm());
            u -= V.block(0,0,m,j)*h.segment(0,j);

            um = u;
            MV(M, um, ParV);

            r1 = u.dot(um);
            MPI_Allreduce(MPI_IN_PLACE, &r1, 1, MPI_DOUBLE,
                            MPI_SUM, ParV->get_comm());
            r1 = sqrt(r1);
        }
        if (r1 <= alpha*r0)
            std::cout << "\n***Warning, loss of orthogonality in Icgsm***\n\n";

        return r1;
    }

    /** Parallel Sparse Matrix-Dense Vector Product **/
    template <typename MatrixType>
    template <typename Vtype>
    void JDSYM<MatrixType>::MV(const MatrixType &M, Vtype &V, ParVars *ParV)
    {
double ttemp = MPI_Wtime();
        V = M*V;
t13 += MPI_Wtime() - ttemp; ttemp = MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, V.data(), V.size(), MPI_DOUBLE,
                      MPI_SUM, ParV->get_comm());
t14 += MPI_Wtime() - ttemp; ttemp = MPI_Wtime();
        for (long i = 0; i < ParV->mydofs.rows(); i++)
            V.row(i) = V.row(ParV->mydofs(i));
        V.conservativeResize(ParV->mydofs.rows(), V.cols());
t15 += MPI_Wtime() - ttemp; ttemp = MPI_Wtime();

        return;
    }

    /** Tailored GMRES Routine **/
    template <typename MatrixType>
    void JDSYM<MatrixType>::gmres(const MatrixType &A, Eigen::Matrix<Scalar, -1, 1>  &v,
               double tol, short MaxIt, ParVars *ParV)
    {
double ttemp;
        /** Initialization **/
        short lsit = 0;

        if ( (MaxIt<2) || (tol>=1) )
            return;

        Eigen::Matrix<Scalar,-1,-1> H = Eigen::Matrix<Scalar,-1,-1>::Zero(MaxIt+1, MaxIt);
        double rho = 1, gamma;
        Eigen::Matrix<Scalar,-1,1> Gamma = Eigen::Matrix<Scalar,-1,1>::Ones(MaxIt+1);

        /** Applying Preconditioner **/
        PRC(v, ParV);
        QkFQb(v, ParV);
        double rho0 = v.squaredNorm();
        MPI_Allreduce(MPI_IN_PLACE, &rho0, 1, MPI_DOUBLE,
                      MPI_SUM, ParV->get_comm());
        rho0 = sqrt(rho0);
        v /= rho0;

        Eigen::Matrix<Scalar,-1,-1> V = Eigen::Matrix<Scalar,-1,-1>::Zero(m, MaxIt+1);
        double tol0 = 1/(tol*tol);

        while ( (lsit<MaxIt) && (rho < tol0) )
        {
            V.conservativeResize(m,lsit+1);
            V.col(lsit) = v;
ttemp = MPI_Wtime();
            MV(A, v, ParV);
t3 += MPI_Wtime() - ttemp; ttemp = MPI_Wtime();
            PRC(v, ParV);
t4 += MPI_Wtime() - ttemp; ttemp = MPI_Wtime();
            QkFQb(v, ParV);
t5 += MPI_Wtime() - ttemp; ttemp = MPI_Wtime();
            Eigen::Matrix<Scalar,-1,1> h = V.transpose()*v;
t6 += MPI_Wtime() - ttemp;
            MPI_Allreduce(MPI_IN_PLACE, h.data(), h.size(), MPI_DOUBLE,
                      MPI_SUM, ParV->get_comm());
ttemp = MPI_Wtime();
            v -= V*h;
t8 += MPI_Wtime() - ttemp;
            gamma = v.squaredNorm();
            MPI_Allreduce(MPI_IN_PLACE, &gamma, 1, MPI_DOUBLE,
                      MPI_SUM, ParV->get_comm());
            gamma = sqrt(gamma);
            v /= gamma;

            H.block(0,lsit,lsit+1,1) = h;
            H(lsit+1, lsit) = gamma;

            if (gamma == 0) // Lucky Breakdown
                {lsit++; break;}
            else // Solve in Least Squares Sense
            {
                lsit++;
                gamma = (-Gamma.segment(0,lsit).transpose()*H.block(0,lsit-1,lsit,1))(0,0)/gamma;
                Gamma(lsit) = gamma;
                rho += gamma*gamma;
            }

        }

        if (gamma == 0) // Lucky Breakdown
        {
            Eigen::VectorXd e1 = Eigen::VectorXd::Zero(lsit); e1(0) = rho0;
            //double rnrm = 0;
            v = V.block(0,0,m,lsit)*(H.block(0,0,lsit,lsit).lu().solve(e1));
        }
        else // Solve in Least Squares Sense
        {
            Eigen::VectorXd e1 = Eigen::VectorXd::Zero(lsit+1); e1(0) = rho0;
            //double rnrm = 1/sqrt(rho);
            v = V.block(0,0,m,lsit)*(H.block(0,0,lsit+1,lsit).householderQr().solve(e1));
        }
        init += lsit;

        return;
    }

    /** v-Qkhat*(Fhat\(Qbhat'*v)) **/
    template <typename MatrixType>
    void JDSYM<MatrixType>::QkFQb(Eigen::Matrix<Scalar, -1, 1> &v, ParVars *ParV)
    {
        Eigen::VectorXd temp = Qb.block(0,0,m,k+1).transpose()*v; // Qmhat'*v
        MPI_Allreduce(MPI_IN_PLACE, temp.data(), temp.size(), MPI_DOUBLE,
                      MPI_SUM, ParV->get_comm());
        Eigen::PartialPivLU< Eigen::Matrix<Scalar,-1,-1> > Finv;
        Finv.compute(F.block(0,0,k+1,k+1)); // Fhat\Qmhat'*v
        temp = Finv.solve(temp);
        v -= Qk.block(0,0,m,k+1)*temp; // v-Qkhat*Fhat\Qm*v

        return;
    }

    template <typename MatrixType>
    void JDSYM<MatrixType>::PrepPRC(const MatrixType &A, const MatrixType &B, ParVars *ParV)
    {
        Jac.setZero(m);
        for (long i = 0; i < m; i++)
        {
            for (long ii = A.outerIndexPtr()[i]; ii < A.outerIndexPtr()[i+1]; ii++ )
            {
                if (ParV->mydofs(i) == A.innerIndexPtr()[ii])
                {
                    Jac(i) = A.valuePtr()[ii] - tau*B.valuePtr()[ii];
                    break;
                }
            }
        }
        return;
    }

    /** Jacobi Preconditioner **/
    template <typename MatrixType>
    void JDSYM<MatrixType>::PRC(Eigen::Matrix<Scalar, -1, 1> &v, ParVars *ParV)
    {
        v = v.cwiseQuotient(Jac);
        return;
    }
}


#endif // PARLAB_H_INCLUDED
