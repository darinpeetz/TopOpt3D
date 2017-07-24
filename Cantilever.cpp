#include <Eigen/Eigen>
#include <iostream>
#include <fstream>
#include "mpi.h"
#include "EigLab.h"
#include "Inputs.h"
#include "Functions.h"
#include "Domain.h"
#include <cmath>

using namespace std;
typedef Eigen::Matrix<double, -1, -1, Eigen::RowMajor> MatrixXdRM;

/// 3D cantilever input file
namespace Input
{
  void Def_Param(TopOpt *topOpt, MMA *optmma, Eigen::VectorXd &Dimensions,
                 ArrayXPI &Nel, double &R, Eigen::VectorXd &zIni)
  {
      Dimensions.resize(6,1);
      Dimensions << 0, 0.1, 0, 0.05, 0, 0.05;
      Nel.resize(3,1);
      Nel << 60, 30, 30;
      topOpt->Nu0 = 0.3; topOpt->E0 = 1e11;
      topOpt->density = 1000;
      topOpt->direct = false;

      topOpt->pmin = 1.0; topOpt->pstep = 1; topOpt->pmax = 4.0;

      optmma->Set_KKT_Limit(0);
      optmma->Set_Iter_Limit_Min(5);
      optmma->Set_Iter_Limit_Max(25);
      optmma->Set_Change_Limit(0.01);

      R = 1.5*(Dimensions(1)-Dimensions(0))/Nel(0);
  }

  void Funcs (TopOpt *topOpt, double &f, Eigen::VectorXd &dfdx,
              Eigen::VectorXd &g, Eigen::MatrixXd &dgdx)
  {
    /// Determines objective function, constraint functions from within program
    PetscInt nevals = 6;

    f = 0;
    dfdx.setZero(topOpt->nLocElem);
    g.setZero(1);
    dgdx.setZero(topOpt->nLocElem,1);

    double temp;
    Eigen::VectorXd dtemp = Eigen::VectorXd::Zero(topOpt->nLocElem);
    Functions::Compliance( topOpt, temp, dtemp.data() );
    f += 0.93*temp;
    dfdx += 0.93*dtemp;
    Functions::Perimeter( topOpt, temp, dtemp.data() );
    f += 0.0*temp;
    dfdx += 0.0*dtemp;

    /*/// Buckling subproblem
    PetscOptionsGetInt(NULL,"-nev",&nevals,NULL);
    Functions::Buckling( topOpt, g.data(), dgdx.data(), nevals );
    g.segment(0, nevals); dgdx.block(0, 0, topOpt->nLocElem, nevals);
    for (short i = nevals-1; i > 0; i--)
        g(i) -= 0.99*g(i-1);
    f = g(0); dfdx = dgdx.col(0);*/

    /// Dynamic subproblem
    /*PetscOptionsGetInt(NULL,NULL,"-nev",&nevals,NULL);
    Functions::Dynamic( topOpt, g.data(), dgdx.data(), nevals );
    double W = 1e6;
    g.conservativeResize(nevals);
    g *= W;
    dgdx.conservativeResize(topOpt->nLocElem, nevals);
    dgdx *= W;
    for (short i = nevals-1; i > 0; i--)
        g(i) -= pow(0.99,i)*g(0);
    if (nevals > 0)
    {
      f = g(0); dfdx = dgdx.col(0);
    }
    else
    {
      f = 0; g.setZero(1); dgdx.setZero(topOpt->nLocElem,1);
    }*/

    /// Volume Subproblem
    Functions::Volume( topOpt, g(0), dgdx.data() );
    g(0) -= 0.2;

    return;
  }

  void Domain(Eigen::ArrayXXd &Points, const Eigen::VectorXd &Box,
              Eigen::Array<bool, -1, 1> &elemValidity)
  {
    elemValidity.setOnes(Points.rows());

    return;
  }

  void Def_BC(TopOpt *topOpt, const Eigen::VectorXd &Box,
              ArrayXPI Nel)
  {
    Eigen::ArrayXd dx(topOpt->numDims);
    for (short i = 0; i < topOpt->numDims; i++)
      dx(i) = (Box(2*i+1)-Box(2*i))/Nel(i);
    double buffer = min(2.5, Nel.cast<double>().minCoeff()/4);
    double eps = 1e-4*dx.minCoeff();

    /// Support Specifications - fix left edge of domain
    int suppInd = 0;
    topOpt->suppNode.resize((Nel(0)+1)*(Nel(1)+1)*(Nel(2))+1);
    //int loadInd = 0;
    //topOpt->loadNode.resize(2);
    int massInd = 0;
    topOpt->massNode.resize(10);
    for (int nd = 0; nd < topOpt->nLocNode; nd++)
    {
      if (topOpt->node(nd,0) < eps)
        topOpt->suppNode(suppInd++) = nd;
      if ( (Box(1) - topOpt->node(nd,0) < eps) &&
           (abs(Box(3)/2-topOpt->node(nd,1))<0.6*dx(1)) &&
           (abs(Box(5)/2-topOpt->node(nd,2))<0.6*dx(2)) )
        topOpt->massNode(massInd++) = nd;
      /*if (topOpt->node(nd,1) > Box(3)-eps)
        if (topOpt->node(nd,0) < eps || topOpt->node(nd,0) > Box(1)-eps)
          topOpt->loadNode(loadInd++) = nd;*/

    }
    /*topOpt->loadNode.conservativeResize(loadInd);
    topOpt->loads.setZero(loadInd,2);
    topOpt->loads.col(1).setOnes();
    topOpt->loads *= -1;*/

    topOpt->suppNode.conservativeResize(suppInd);
    topOpt->supports.setOnes(suppInd,topOpt->numDims);
    topOpt->massNode.conservativeResize(massInd);
    topOpt->masses.setOnes(massInd,topOpt->numDims);
    topOpt->masses *= 0.05;

    return;
  }
}
