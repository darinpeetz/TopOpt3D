#include "MMA.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <cmath>

using namespace std;

/*************************************************************************/
/**                      Preallocate some arrays                        **/
/*************************************************************************/
void MMA::Initialize()
{
  /// These variables are usually not going to change, but the user should have the option
  zzz.resize(nloc);
  low.resize(nloc);
  upp.resize(nloc);
  if (xval.size() != nloc)
    xval.setZero(nloc);
  xold1 = xval; xold2 = xval;
  active.resize(nloc);
  fill(active.begin(), active.end(), true);
  nactive = nloc;
}

/*************************************************************************/
/**                     Set number of constraints                       **/
/*************************************************************************/
void MMA::Set_m( uint mval )
{
  m = mval;
  /// These parameters should be set by user, but can't hurt to have a default
  a0 = 1;
  b0 = 1;
  a = Eigen::VectorXd::Zero(m);
  b = a;
  d = Eigen::VectorXd::Ones(m);
  c = 1000*d;

  return;
}

/*************************************************************************/
/**                      Generic update routine                         **/
/*************************************************************************/
// sizes: dfdx=nx1, g=mx1, dgdx=nxm
int MMA::Update( Eigen::VectorXd &dfdx, Eigen::VectorXd &g, Eigen::MatrixXd &dgdx )
{
  int ierr = 0;
  if (nactive != dfdx.size())
  {
    if (myid == 0) {
      printf("Error dectected at line %i in file %s\n", __LINE__, __FILE__);
      printf("Objective gradient vector size (%li) does not match number of "
             "active design variables (%lu)\n", dfdx.size(), nactive);
    }
    MPI_Barrier(Comm);
    return 60;
  }
  if (nactive != dgdx.rows())
  {
    if (myid == 0) {
      printf("Error dectected at line %i in file %s\n", __LINE__, __FILE__);
      printf("Constraint gradient vectors size (%li) does not match number of "
             "active design variables (%lu)\n", dgdx.rows(), nactive);
    }
    MPI_Barrier(Comm);
    return 60;
  }
  /// At some point I plan to correctly implement OC-type update for simple problems
  Set_m(g.size());
  if (m < 2)
  {
    OCeta = 0.5;
    OCMove = 0.2;
    ierr = MMAsub(dfdx, g, dgdx);
  }
  else
    ierr = MMAsub(dfdx, g, dgdx);

  return ierr;
}

/*************************************************************************/
/**                         OC update routine                           **/
/*************************************************************************/
int MMA::OCsub( Eigen::VectorXd &dfdx, Eigen::VectorXd &g, Eigen::MatrixXd &dgdx )
{
  int ierr = 0;

  // Grab active design variables
  Eigen::VectorXd *p_x, x, *p_xmin, xmin_act, *p_xmax, xmax_act;
  if (nactive == nloc)
  {
    xold1 = xval;
    p_x = &xval;
    p_xmin = &xmin;
    p_xmax = &xmax;
  }
  else {
    x.resize(nactive); xmin_act.resize(nactive); xmax_act.resize(nactive);
    ulong ind = 0;
    for (long i = 0; i < nloc; i++) {
      if (active[i]) {
        xmin_act(ind) = xmin(i);
        xmax_act(ind) = xmax(i);
        x(ind++) = xval(i);
      }
    }
    p_x = &x;
    p_xmin = &xmin_act;
    p_xmax = &xmax_act;
    xold1 = x;
  }

  Eigen::VectorXd step = OCMove*(*p_xmax-*p_xmin);
  Eigen::VectorXd temp(nactive), B(nactive), xCnd(nactive);
  double l1 = 0, l2 = 1e6, dg;
  while (l2-l1 > 1e-4)
  {
    double lmid = 0.5*(l1+l2);
    B = (dfdx.cwiseQuotient(dgdx.transpose())).cwiseAbs()/lmid;
    for (uint i = 0; i < nactive; i++)
      temp(i) = pow(B(i), OCeta);
    xCnd = *p_xmin + (xold1-*p_xmin).cwiseProduct(temp);
    for (uint i = 0; i < nactive; i++)
      (*p_x)(i) = std::max( std::max( std::min( std::min(xCnd(i), xold1(i)+step(i)),
                                    (*p_xmax)(i)), xold1(i)-step(i)), (*p_xmin)(i));

    dg = dgdx.col(0).dot(*p_x-xold1);
    ierr = MPI_Allreduce(MPI_IN_PLACE, &dg, 1, MPI_DOUBLE, MPI_SUM, Comm);
    if (ierr != 0) {
      if (myid == 0)
        printf("Error in MPI_Allreduce at line %i in file %s\n",
               __LINE__, __FILE__);
      MPI_Barrier(Comm);
      return ierr;
    }

    if (g(0)+dg > 0)
      l1 = lmid;
    else
      l2 = lmid;
  }

  if (nactive != nloc) {
    ulong ind = 0;
    for (long i = 0; i < nloc; i++) {
      if (active[i])
        xval(i) = x(ind++);
    }
  }

  Change = ((xval-xold1).cwiseQuotient(xmax-xmin)).cwiseAbs().maxCoeff();
  return ierr;
}

/*************************************************************************/
/**                      Construct MMA subproblem                       **/
/*************************************************************************/
int MMA::MMAsub( Eigen::VectorXd &dfdx, Eigen::VectorXd &g, Eigen::MatrixXd &dgdx )
{
  int ierr = 0;

  // Grab active design variables
  Eigen::VectorXd *p_x, x, *p_x1, x1, *p_x2, x2;
  Eigen::VectorXd *p_xmin, xmin_act, *p_xmax, xmax_act;
  Eigen::VectorXd low_act, upp_act;
  if (nactive == nloc)
  {
    p_x = &xval;
    p_x1 = &xold1;
    p_x2 = &xold2;
    p_xmin = &xmin;
    p_xmax = &xmax;
    p_low = &low;
    p_upp = &upp;
  }
  else {
    x.resize(nactive); x1.resize(nactive); x2.resize(nactive);
    xmin_act.resize(nactive); xmax_act.resize(nactive);
    low_act.resize(nactive); upp_act.resize(nactive);
    ulong ind = 0;
    for (long i = 0; i < nloc; i++) {
      if (active[i]) {
        upp_act(ind) = upp(i);
        low_act(ind) = low(i);
        xmax_act(ind) = xmax(i);
        xmin_act(ind) = xmin(i);
        x2(ind) = xold2(i);
        x1(ind) = xold1(i);
        x(ind++) = xval(i);
      }
    }
    p_x = &x;
    p_x1 = &x1;
    p_x2 = &x2;
    p_xmin = &xmin_act;
    p_xmax = &xmax_act;
    p_low = &low_act;
    p_upp = &upp_act;
  }

  /// Asymptote Calculation (low and upp)
  if (fresh_start && iter < 3)
  {
    *p_low = *p_x-asyinit*(*p_xmax-*p_xmin);
    *p_upp = *p_x+asyinit*(*p_xmax-*p_xmin);
  }
  else
  {
    fresh_start = false;
    zzz = (*p_x-*p_x1).cwiseProduct(*p_x1-*p_x2);
    factor = Eigen::VectorXd::Ones(nactive);
    for (uint i = 0; i < nactive; i++)
    {
      if (zzz[i] > 0)
        factor[i] = asyincr;
      else if(zzz[i] < 0)
        factor[i] = asydecr;
    }
    *p_low = *p_x - factor.cwiseProduct(*p_x1-*p_low);
    *p_upp = *p_x + factor.cwiseProduct(*p_upp-*p_x1);
    Eigen::VectorXd lowmin = *p_x - 10 * (*p_xmax-*p_xmin);
    Eigen::VectorXd lowmax = *p_x - 0.01*(*p_xmax-*p_xmin);
    Eigen::VectorXd uppmin = *p_x + 0.01*(*p_xmax-*p_xmin);
    Eigen::VectorXd uppmax = *p_x + 10 * (*p_xmax-*p_xmin);
    for (uint i = 0; i < nactive; i++)
    {
      (*p_low)(i) = std::max((*p_low)(i), lowmin(i));
      (*p_low)(i) = std::min((*p_low)(i), lowmax(i));
      (*p_upp)(i) = std::min((*p_upp)(i), uppmax(i));
      (*p_upp)(i) = std::max((*p_upp)(i), uppmin(i));
    }
  }

  /// Calculation of the bounds alfa and beta
  Eigen::VectorXd zzz1 = *p_low + albefa*(*p_x-*p_low);
  Eigen::VectorXd zzz2 = *p_x - mmamove*(*p_xmax-*p_xmin);
  alfa.resize(nactive);
  for (uint i = 0; i < nactive; i++)
  {
    zzz(i) = std::max(zzz1(i),zzz2(i));
    alfa(i) = std::max(zzz(i),(*p_xmin)(i));
  }
  zzz1 = *p_upp - albefa*(*p_upp-*p_x);
  zzz2 = *p_x + mmamove*(*p_xmax-*p_xmin);
  beta.resize(nactive);
  for (uint i = 0; i < nactive; i++)
  {
    zzz(i) = std::min(zzz1(i),zzz2(i));
    beta(i) = std::min(zzz(i),(*p_xmax)(i));
  }

  /// Calculations of p0, q0, P, Q, and b
  Eigen::VectorXd xmami = *p_upp-*p_low;
  Eigen::VectorXd xmamieps = Eigen::VectorXd::Constant(n, 1e-5);
  for (uint i = 0; i < nactive; i++)
    xmami[i] = std::max(xmami[i],xmamieps[i]);
  Eigen::VectorXd xmamiinv = xmami.cwiseInverse();
  Eigen::VectorXd ux1 = *p_upp-*p_x;
  Eigen::VectorXd ux2 = ux1.cwiseProduct(ux1);
  Eigen::VectorXd xl1 = *p_x-*p_low;
  Eigen::VectorXd xl2 = xl1.cwiseProduct(xl1);

  p0.setZero(nactive);
  q0.setZero(nactive);
  for (uint i = 0; i < nactive; i++)
  {
    p0(i) = std::max(dfdx(i),0.0);
    q0(i) = std::max(-dfdx(i),0.0);
  }
  Eigen::VectorXd pq0 = 0.001*(p0 + q0) + 0.5*raa0*xmamiinv;
  p0 = p0 + pq0;
  q0 = q0 + pq0;
  p0 = p0.cwiseProduct(ux2);
  q0 = q0.cwiseProduct(xl2);

  P.resize(nactive, m);
  Q.resize(nactive, m);
  for (long i = 0; i < nactive; i++)
  {
    for (int j = 0; j < m; j++)
    {
      P(i, j) = std::max(dgdx(i,j),0.0);
      Q(i, j) = std::max(-dgdx(i,j),0.0);
    }
  }
  //Eigen::MatrixXd PQ = 0.001*(P + Q) + raa0*xmamiinv.replicate(1,m);
  //P = P + PQ;
  //Q = Q + PQ;
  for (int i = 0; i < m; i++)
  {
    P.col(i) = P.col(i).cwiseProduct(ux2);
    Q.col(i) = Q.col(i).cwiseProduct(xl2);
  }

  //b = P*uxinv + Q*xlinv - g; If not done in parallel
  for (int i = 0; i < m; i++)
    b(i) = (P.col(i).cwiseQuotient(ux1) + Q.col(i).cwiseQuotient(xl1)).sum();
  MPI_Allreduce(MPI_IN_PLACE, b.data(), m, MPI_DOUBLE, MPI_SUM, Comm);
  b -= g;

  *p_x2 = *p_x1; *p_x1 = *p_x;
  ///Solving the subproblem by a primal-dual Newton Method (or not)
  if (nproc > 1)
  {
    ierr = DualSolve(*p_x);
  }
  else
    ierr = DualSolve(*p_x);

  Change = ((*p_x-*p_x1).cwiseQuotient(*p_xmax-*p_xmin)).cwiseAbs().maxCoeff();
  ierr = MPI_Allreduce(MPI_IN_PLACE,&Change,1,MPI_DOUBLE,MPI_MAX,Comm);

  // Update active values in full local arrays
  if (nactive != nloc) {
    ulong ind = 0;
    for (long i = 0; i < nloc; i++) {
      if (active[i]) {
        upp(i)   = upp_act(ind);
        low(i)   = low_act(ind);
        xold2(i) = x2(ind);
        xold1(i) = x1(ind);
        xval(i)  = x(ind++);
      }
    }
  }
  return ierr;
}

/*************************************************************************/
/**            Solve the subproblem with a dual Newton method           **/
/*************************************************************************/
int MMA::DualSolve(Eigen::VectorXd &x)
{
  int ierr = 0;
  /// Dual Solver as described in Aage and Lazarov (2013)
  double epsi = 1, Theta;

  Eigen::VectorXd ux1, xl1, ux2, xl2, ux3, xl3;
  Eigen::MatrixXd Hess(m,m);
  Eigen::VectorXd epsvec(m), Grad(m);

  eta.setOnes(m); lambda = 500*eta;
  Eigen::VectorXd dellam, deleta;
  Eigen::VectorXd y;
  double z;

  plam    = p0 + P*lambda;
  qlam    = q0 + Q*lambda;
  XYZofLam(x, y, z, lambda);

  ux1   = *p_upp-x;
  ux2   = ux1.cwiseProduct(ux1);
  ux3   = ux1.cwiseProduct(ux2);
  xl1   = x-*p_low;
  xl2   = xl1.cwiseProduct(xl1);
  xl3   = xl1.cwiseProduct(xl2);
  ierr = DualGrad(ux1, xl1, y, z, Grad); if (ierr!=0) {return ierr;}

  while (epsi > epsimin)
  {
    epsvec.setConstant(epsi);

    DualResidual(Grad, eta, lambda, epsvec);

    int ittt = 0;
    while (residumax > 0.9*epsi && ittt < 100)
    {
      ittt++;

      ierr = DualHess(ux2, xl2, ux3, xl3, x, Hess); if (ierr!=0) {return ierr;}

      SearchDir(Hess, Grad, lambda, eta, dellam, deleta, epsvec);
      Theta = SearchDis(lambda, eta, dellam, deleta);

      lambda += Theta*dellam;
      eta    += Theta*deleta;

      plam    = p0 + P*lambda;
      qlam    = q0 + Q*lambda;
      XYZofLam(x, y, z, lambda);

      ux1     = *p_upp-x;
      xl1     = x-*p_low;
      ux2     = ux1.cwiseProduct(ux1);
      xl2     = xl1.cwiseProduct(xl1);
      ux3     = ux2.cwiseProduct(ux1);
      xl3     = xl2.cwiseProduct(xl1);

      ierr = DualGrad(ux1, xl1, y, z, Grad); if (ierr!=0) {return ierr;}

      DualResidual(Grad, eta, lambda, epsvec);
    }
    epsi = 0.1*epsi;
  }

  return ierr;
}

/*************************************************************************/
/**                    Residual of dual subproblem                      **/
/*************************************************************************/
void MMA::DualResidual(Eigen::VectorXd &Grad, Eigen::VectorXd &eta,
                        Eigen::VectorXd &lambda, Eigen::VectorXd &epsvec)
{
  residual.resize(2*m);
  residual.segment(0,m) = Grad + eta;
  residual.segment(m,m) = eta.cwiseProduct(lambda) - epsvec;

  residunorm = residual.norm();
  residumax = residual.cwiseAbs().lpNorm<Eigen::Infinity>();
  return;
}

/*************************************************************************/
/**       Get primal variables in terms of dual variable (lambda)       **/
/*************************************************************************/
void MMA::XYZofLam(Eigen::VectorXd &x, Eigen::VectorXd &y, double &z, Eigen::VectorXd &lambda)
{
  Eigen::VectorXd plamrt = plam.cwiseSqrt();
  Eigen::VectorXd qlamrt = qlam.cwiseSqrt();
  x = (plamrt.cwiseProduct(*p_low)+qlamrt.cwiseProduct(*p_upp)).cwiseQuotient(plamrt+qlamrt);
  for (long i = 0; i < nactive; i++)
    x(i) = std::max(std::min(x(i),beta(i)),alfa(i));

  y = (lambda-c).cwiseQuotient(d);
  for (int i = 0; i < m; i++)
    y(i) = std::max(y(i), 0.0);

  z = 10*std::max( (lambda.dot(a)-a0)/b0, 0.0);

  return;
}

/*************************************************************************/
/**                  Gradient of the dual subproblem                    **/
/*************************************************************************/
int MMA::DualGrad(Eigen::VectorXd &ux1, Eigen::VectorXd &xl1,
                   Eigen::VectorXd &y, double &z, Eigen::VectorXd &grad)
{
  int ierr = 0;
  /// fi(x(lambda))+b
  for (int i = 0; i < m; i++)
    grad(i) = (P.col(i).cwiseQuotient(ux1) + Q.col(i).cwiseQuotient(xl1)).sum();
  ierr = MPI_Allreduce(MPI_IN_PLACE, grad.data(), m, MPI_DOUBLE, MPI_SUM, Comm);
  /// -b-a*z(lambda)-y(lambda)
  grad -= b + a*z + y;

  return ierr;
}

/*************************************************************************/
/**                    Hessian of the dual subproblem                   **/
/*************************************************************************/
int MMA::DualHess(Eigen::VectorXd &ux2, Eigen::VectorXd &xl2,
                   Eigen::VectorXd &ux3, Eigen::VectorXd &xl3,
                   Eigen::VectorXd &x,   Eigen::MatrixXd &Hess)
{
  int ierr = 0;
  Eigen::MatrixXd dhdx(m,nactive);

  for (int i = 0; i < m; i++)
     dhdx.row(i) = P.col(i).cwiseQuotient(ux2) - Q.col(i).cwiseQuotient(xl2);
  Eigen::VectorXd dLdxx = (x.array()>alfa.array() && x.array()<beta.array())
            .cast<double>().matrix()
            .cwiseQuotient(2*plam.cwiseQuotient(ux3) + 2*qlam.cwiseQuotient(xl3));

  Hess = -dhdx*dLdxx.asDiagonal()*dhdx.transpose();
  ierr = MPI_Allreduce(MPI_IN_PLACE, Hess.data(), m*m, MPI_DOUBLE, MPI_SUM, Comm);

  Eigen::VectorXd Hessdy = Eigen::VectorXd::Zero(m);
  for (int i = 0; i < m; i++)
    Hessdy(i) = (double)lambda(i)>c(i);

  Hess -= Hessdy.asDiagonal();

  if (lambda.dot(a) > 0)
    Hess -= 10*(a*a.transpose());

  return ierr;
}

/*************************************************************************/
/**                  Get step direction for dual solver                 **/
/*************************************************************************/
void MMA::SearchDir(Eigen::MatrixXd &Hess, Eigen::VectorXd &Grad,
                    Eigen::VectorXd &lambda, Eigen::VectorXd &eta,
                    Eigen::VectorXd &dellam, Eigen::VectorXd &deleta,
                    Eigen::VectorXd &epsvec)
{
  Eigen::MatrixXd A = Hess - Eigen::MatrixXd((eta.cwiseQuotient(lambda)).asDiagonal());
  A += std::min(1e-4*A.trace()/m,-1e-7)*Eigen::MatrixXd::Identity(m,m);
  Eigen::VectorXd b = -Grad - epsvec.cwiseQuotient(lambda);
  dellam = A.partialPivLu().solve(b);
  deleta = -eta + epsvec.cwiseQuotient(lambda) -
           dellam.cwiseProduct(eta.cwiseQuotient(lambda));

  return;
}

/*************************************************************************/
/**                    Set step size for dual solver                    **/
/*************************************************************************/
double MMA::SearchDis(Eigen::VectorXd &lambda, Eigen::VectorXd &eta,
                      Eigen::VectorXd &dellam, Eigen::VectorXd &deleta)
{
  double Theta = 1.0;
  Eigen::VectorXd Ratio = -0.99*lambda.cwiseQuotient(dellam);
  for (int i = 0; i < m; i ++)
  {
    if (Ratio(i) >= 0)
      Theta = std::min(Theta, Ratio(i));
  }
  Ratio = -0.99*eta.cwiseQuotient(deleta);
  for (int i = 0; i < m; i ++)
  {
    if (Ratio(i) >= 0)
      Theta = std::min(Theta, Ratio(i));
  }

  return Theta;
}

/*************************************************************************/
/**            Solve the subproblem with a Primal-Dual method           **/
/*************************************************************************/
void MMA::primaldual_subsolve(Eigen::VectorXd &x)
{
  ///Primal-Dual Solver for MMA subproblem
  Eigen::MatrixXd diag1, diag2;

  double epsi = 1;
  Eigen::VectorXd epsvecn = Eigen::VectorXd::Constant(nactive, epsi);
  Eigen::VectorXd epsvecm = Eigen::VectorXd::Constant(m, epsi);
  x = 0.5*(alfa+beta);
  Eigen::VectorXd y = Eigen::VectorXd::Ones(m);
  double z = 1;
  lambda.setOnes(m);
  Eigen::VectorXd xsi = (x-alfa).cwiseInverse();
  Eigen::VectorXd eta = (beta-x).cwiseInverse();
  Eigen::VectorXd mu(m);
  for(uint i = 0; i < nactive; i++)
  {
    xsi(i) = std::max(xsi(i),1.0);
    eta(i) = std::max(eta(i),1.0);
  }
  for(int i = 0; i < m; i++)
    mu(i)  = std::max(1.0,0.5*c(i));
  zet = 1;
  Eigen::VectorXd s = Eigen::VectorXd::Ones(m);
  int itera = 0;

  while (epsi > epsimin)
  {
    epsvecn.setConstant(epsi);
    epsvecm.setConstant(epsi);
    Eigen::VectorXd ux1 = *p_upp-x;
    Eigen::VectorXd xl1 = x-*p_low;
    Eigen::VectorXd ux2 = ux1.cwiseProduct(ux1);
    Eigen::VectorXd xl2 = xl1.cwiseProduct(xl1);
    Eigen::MatrixXd uxinv1 = ux1.cwiseInverse();
    Eigen::MatrixXd xlinv1 = xl1.cwiseInverse();
    Eigen::MatrixXd plam = p0 + P*lambda;
    Eigen::MatrixXd qlam = q0 + Q*lambda;
    Eigen::MatrixXd gvec = P.transpose()*uxinv1 + Q.transpose()*xlinv1;
    Eigen::MatrixXd dpsidx = plam.cwiseQuotient(ux2) - qlam.cwiseQuotient(xl2);
    Eigen::MatrixXd rex = dpsidx - xsi + eta;
    Eigen::MatrixXd rey = c + d.cwiseProduct(y) - mu - lambda;
    double rez = a0 - zet - a.dot(lambda);
    Eigen::MatrixXd relam = gvec - a*z - y + s - b;
    Eigen::MatrixXd rexsi = xsi.cwiseProduct(x-alfa) - epsvecn;
    Eigen::MatrixXd reeta = eta.cwiseProduct(beta-x) - epsvecn;
    Eigen::MatrixXd remu = mu.cwiseProduct(y) - epsvecm;
    double rezet = zet*z - epsi;
    Eigen::MatrixXd res = lambda.cwiseProduct(s) - epsvecm;

    Eigen::VectorXd residu1(rex.rows() + rey.rows() + 1);
    residu1 <<  rex,
                rey,
                rez;

    Eigen::VectorXd residu2(relam.rows() + rexsi.rows() + reeta.rows() +
                            remu.rows() + 1 + res.rows());
    residu2 <<  relam,
                rexsi,
                reeta,
                remu,
                rezet,
                res;

    residual.resize(residu1.rows() + residu2.rows());
    residual <<   residu1,
                  residu2;

    residunorm = residual.norm();
    residumax = residual.cwiseAbs().maxCoeff();
    int ittt = 0;
    while (residumax > 0.9*epsi && ittt < 200)
    {
      ittt++;
      itera++;
      ux1 = *p_upp-x;
      xl1 = x-*p_low;
      ux2 = ux1.cwiseProduct(ux1);
      xl2 = xl1.cwiseProduct(xl1);
      Eigen::VectorXd ux3 = ux1.cwiseProduct(ux2);
      Eigen::VectorXd xl3 = xl1.cwiseProduct(xl2);
      uxinv1 = ux1.cwiseInverse();
      xlinv1 = xl1.cwiseInverse();
      Eigen::VectorXd uxinv2 = ux2.cwiseInverse();
      Eigen::VectorXd xlinv2 = xl2.cwiseInverse();
      plam = p0 + P*lambda;
      qlam = q0 + Q*lambda;
      gvec = P.transpose()*uxinv1 + Q.transpose()*xlinv1;
      Eigen::MatrixXd GG = P.transpose()*uxinv2.asDiagonal() -
                           Q.transpose()*xlinv2.asDiagonal();
      dpsidx = plam.cwiseQuotient(ux2) - qlam.cwiseQuotient(xl2);
      Eigen::VectorXd delx = dpsidx - epsvecn.cwiseQuotient(x-alfa) +
                             epsvecn.cwiseQuotient(beta-x);
      Eigen::VectorXd dely = c + d.cwiseProduct(y) - lambda - epsvecm.cwiseQuotient(y);
      double delz = a0 - a.transpose()*lambda - epsi/z;
      Eigen::VectorXd dellam = gvec - a*z - y - b + epsvecm.cwiseQuotient(lambda);
      Eigen::VectorXd diagx = plam.cwiseQuotient(ux3) + qlam.cwiseQuotient(xl3);
      diagx = 2*diagx + xsi.cwiseQuotient(x-alfa) + eta.cwiseQuotient(beta-x);
      Eigen::VectorXd diagxinv = diagx.cwiseInverse();
      Eigen::VectorXd diagy = d + mu.cwiseQuotient(y);
      Eigen::VectorXd diagyinv = diagy.cwiseInverse();
      Eigen::VectorXd diaglam = s.cwiseQuotient(lambda);
      Eigen::VectorXd diaglamyi = diaglam + diagyinv;
      Eigen::VectorXd dlam, dx;
      double dz;
      if (m < nactive)
      {
        Eigen::VectorXd blam = dellam + dely.cwiseQuotient(diagy) -
                               GG*(delx.cwiseQuotient(diagx));
        Eigen::VectorXd bb(blam.rows() + 1);
        bb <<   blam,
                delz;

        diag1 = diaglamyi.asDiagonal();
        Eigen::MatrixXd Alam = diag1 + GG*diagxinv.asDiagonal()*GG.transpose();
        Eigen::MatrixXd AA(Alam.rows() + a.cols(), Alam.cols() + a.cols());
        AA <<   Alam,               a,
                a.transpose(),  -zet/z;

        Eigen::VectorXd solut = AA.partialPivLu().solve(bb);
        dlam = solut.block(0, 0, m, 1);
        dz = solut(m);
        dx = -delx.cwiseQuotient(diagx) - (GG.transpose()*dlam).cwiseQuotient(diagx);
      }
      else
      {
        Eigen::VectorXd diaglamyiinv = diaglamyi.cwiseInverse();
        Eigen::VectorXd dellamyi = dellam + dely.cwiseQuotient(diagy);
        diag1 = diagx.asDiagonal();
        Eigen::MatrixXd Axx = diag1 + GG.transpose()*diaglamyiinv.asDiagonal()*GG;
        double azz = zet/z + a.transpose()*(a.cwiseQuotient(diaglamyi));
        Eigen::VectorXd axz = -GG.transpose()*(a.cwiseQuotient(diaglamyi));
        Eigen::VectorXd bx = delx + GG.transpose()*(dellamyi.cwiseQuotient(diaglamyi));
        double bz = delz - a.transpose()*(dellamyi.cwiseQuotient(diaglamyi));

        Eigen::MatrixXd AA(Axx.rows() + axz.cols(), Axx.cols() + axz.cols());
        AA << Axx,              axz,
              axz.transpose(),  azz;

        Eigen::VectorXd bb(bx.rows() + 1);
        bb <<   -bx,
                -bz;

        Eigen::VectorXd solut = AA.partialPivLu().solve(bb);
        dx = solut.block(0, 0, nactive, 1);
        dz = solut(nactive);
        dlam = (GG*dx).cwiseQuotient(diaglamyi) - dz*(a.cwiseQuotient(diaglamyi)) +
               dellamyi.cwiseQuotient(diaglamyi);
      }
      Eigen::VectorXd dy = -dely.cwiseQuotient(diagy) + dlam.cwiseQuotient(diagy);
      Eigen::VectorXd dxsi = -xsi + epsvecn.cwiseQuotient(x-alfa) -
                             (xsi.cwiseProduct(dx)).cwiseQuotient(x-alfa);
      Eigen::VectorXd deta = -eta + epsvecn.cwiseQuotient(beta-x) +
                             (eta.cwiseProduct(dx)).cwiseQuotient(beta-x);
      Eigen::VectorXd dmu = -mu + epsvecm.cwiseQuotient(y) -
                            (mu.cwiseProduct(dy)).cwiseQuotient(y);
      double dzet = -zet + epsi/z - zet*dz/z;
      Eigen::VectorXd ds = -s + epsvecm.cwiseQuotient(lambda) -
                           (s.cwiseProduct(dlam)).cwiseQuotient(lambda);
      Eigen::VectorXd xx(y.rows() + 1 + lambda.rows() + xsi.rows() +
                      eta.rows() + mu.rows() + 1 + s.rows());
      Eigen::VectorXd dxx(dy.rows() + 1 + dlam.rows() + dxsi.rows() +
                      deta.rows() + dmu.rows() + 1 + s.rows());
      xx <<   y,
              z,
              lambda,
              xsi,
              eta,
              mu,
              zet,
              s;

      dxx <<  dy,
              dz,
              dlam,
              dxsi,
              deta,
              dmu,
              dzet,
              ds;
      Eigen::MatrixXd stepxx = -1.01*dxx.cwiseQuotient(xx);
      double stmxx = stepxx.maxCoeff();
      Eigen::MatrixXd stepalfa = -1.01*dx.cwiseQuotient(x-alfa);
      double stmalfa = stepalfa.maxCoeff();
      Eigen::MatrixXd stepbeta = 1.01*dx.cwiseQuotient(beta-x);
      double stmbta = stepbeta.maxCoeff();
      double stmalbe = std::max(stmalfa,stmbta);
      double stmalbexx = std::max(stmalbe,stmxx);
      double stminv = std::max(stmalbexx,1.0);
      double steg = 1/stminv;
      Eigen::VectorXd xold = x;
      Eigen::VectorXd yold = y;
      double zold = z;
      Eigen::VectorXd lamold = lambda;
      Eigen::VectorXd xsiold = xsi;
      Eigen::VectorXd etaold = eta;
      Eigen::VectorXd muold = mu;
      double zetold = zet;
      Eigen::VectorXd sold = s;

      int itto = 0;
      double resinew = 2*residunorm;
      while (resinew > residunorm && itto < 50)
      {
        itto++;
        x   = xold   + steg*dx;
        y   = yold   + steg*dy;
        z   = zold   + steg*dz;
        lambda = lamold + steg*dlam;
        xsi = xsiold + steg*dxsi;
        eta = etaold + steg*deta;
        mu  = muold  + steg*dmu;
        zet = zetold + steg*dzet;
        s   = sold   + steg*ds;
        ux1 = *p_upp-x;
        xl1 = x-*p_low;
        ux2 = ux1.cwiseProduct(ux1);
        xl2 = xl1.cwiseProduct(xl1);
        uxinv1 = ux1.cwiseInverse();
        xlinv1 = xl1.cwiseInverse();
        plam = p0 + P*lambda;
        qlam = q0 + Q*lambda;
        gvec = P.transpose()*uxinv1 + Q.transpose()*xlinv1;
        dpsidx = plam.cwiseQuotient(ux2) - qlam.cwiseQuotient(xl2);
        rex = dpsidx - xsi + eta;
        rey = c + d.cwiseProduct(y) - mu - lambda;
        rez = a0 - zet - a.transpose()*lambda;
        relam = gvec - a*z - y + s - b;
        rexsi = xsi.cwiseProduct(x-alfa) - epsvecn;
        reeta = eta.cwiseProduct(beta-x) - epsvecn;
        remu = mu.cwiseProduct(y) - epsvecm;
        rezet = zet*z - epsi;
        res = lambda.cwiseProduct(s) - epsvecm;
        residu1 <<  rex,
                    rey,
                    rez;

        residu2 <<  relam,
                    rexsi,
                    reeta,
                    remu,
                    rezet,
                    res;

        residual <<   residu1,
                      residu2;

        resinew = residual.norm();
        steg = steg/2;
      }
      residunorm = resinew;
      residumax = residual.cwiseAbs().maxCoeff();
      steg = 2*steg;
    }
    epsi = 0.1*epsi;
  }

  xval = x;
  ymma = y;
  zmma = z;
  lamma = lambda;
  xsimma = xsi;
  etamma = eta;
  mumma = mu;
  zetmma = zet;
  smma = s;
  return;
}

/*************************************************************************/
/**                 KKT check for Primal-Dual method                    **/
/*************************************************************************/
void MMA::primaldual_kktcheck( Eigen::VectorXd &dfdx, Eigen::VectorXd &g, Eigen::MatrixXd &dgdx )
{
  Eigen::VectorXd rex = dfdx + dgdx.transpose()*lamma - xsimma + etamma;
  Eigen::VectorXd rey = c + d.cwiseProduct(ymma) - mumma - lamma;
  double rez = a0 - zetmma - a.dot(lamma);
  Eigen::VectorXd relam = g - a*zmma - ymma + smma;
  Eigen::VectorXd rexsi = xsimma.cwiseProduct(xval-xmin);
  Eigen::VectorXd reeta = etamma.cwiseProduct(xmax-xval);
  Eigen::VectorXd remu = mumma.cwiseProduct(ymma);
  double rezet = zetmma*zmma;
  Eigen::VectorXd res = lamma.cwiseProduct(smma);

  Eigen::VectorXd residu1(rex.rows() + rey.rows() + 1);
  Eigen::VectorXd residu2(relam.rows() + rexsi.rows() + reeta.rows() +
                          remu.rows() + 1 + res.rows());
  residu1 <<  rex,
              rey,
              rez;

  residu2 <<  relam,
              rexsi,
              reeta,
              remu,
              rezet,
              res;

  residual <<   residu1,
                residu2;

  residunorm = residual.norm();
  residumax = residual.cwiseAbs().maxCoeff();
  return;
}
