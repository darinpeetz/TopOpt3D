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

/// 2D cantilever input file
namespace Input
{
  void Def_Param(TopOpt *topOpt, MMA *optmma, Eigen::VectorXd &Dimensions,
                 ArrayXPI &Nel, double &R, Eigen::VectorXd &zIni)
  {
      Dimensions.resize(4,1);
      Dimensions << 0, 8, 0, 5;
      Nel.resize(2,1);
      Nel << 40, 25;
      topOpt->Nu0 = 0.3; topOpt->E0 = 1e7;
      topOpt->density = 1000;
      topOpt->direct = false;

      topOpt->pmin = 1.0; topOpt->pstep = 1; topOpt->pmax = 4.0;

      optmma->Set_KKT_Limit(0);
      optmma->Set_Iter_Limit_Min(20);
      optmma->Set_Iter_Limit_Max(100);
      optmma->Set_Change_Limit(0.01);

      R = 1.5*(Dimensions(1)-Dimensions(0))/Nel(0);
  }

  int Funcs (TopOpt *topOpt, double &f, Eigen::VectorXd &dfdx,
              Eigen::VectorXd &g, Eigen::MatrixXd &dgdx)
  {
    /// Determines objective function, constraint functions from within program
    PetscInt nevals = 10;

    dfdx.setZero(topOpt->nLocElem);
    g.setZero(10);
    dgdx.setZero(topOpt->nLocElem,10);

    /*Functions::Compliance( topOpt, f, dfdx.data() );
    Functions::Perimeter( topOpt, g(1), dgdx.data()+topOpt->nLocElem );
    g(1) += 100;*/

    /// Buckling subproblem
    /*PetscOptionsGetInt(NULL,NULL,"-nev",&nevals,NULL);
    Functions::Buckling( topOpt, g.data(), dgdx.data(), nevals );
    g.segment(0, nevals); dgdx.block(0, 0, topOpt->nLocElem, nevals);
    for (short i = 1; i < nevals; i++)
        g(i) -= pow(0.99,i)*g(0);
    f = g(0); dfdx = dgdx.col(0);*/

    /// Dynamic subproblem
    PetscOptionsGetInt(NULL,NULL,"-nev",&nevals,NULL);
    Functions::Dynamic( topOpt, g.data(), dgdx.data(), nevals );
    g.segment(0, nevals); dgdx.block(0, 0, topOpt->nLocElem, nevals);
    for (short i = nevals-1; i > 0; i--)
        g(i) -= 0.99*g(i-1);
    f = g(0); dfdx = dgdx.col(0);

    /// Volume Subproblem
    Functions::Volume( topOpt, g(0), dgdx.data() );
    g(0) -= 0.5;// g(0) *= 100; dgdx *= 100;

    return 0;
  }

  void Domain(Eigen::ArrayXXd &Points, const Eigen::VectorXd &Box,
              Eigen::Array<bool, -1, 1> &elemValidity)
  {
    elemValidity.setOnes(Points.rows());

    return;
  }

  void Def_BC(TopOpt *topOpt, const Eigen::VectorXd &Box, ArrayXPI Nel)
  {
    Eigen::ArrayXd dx(topOpt->numDims);
    for (short i = 0; i < topOpt->numDims; i++)
      dx(i) = (Box(2*i+1)-Box(2*i))/Nel(i);
    double buffer = min(2.5, Nel.cast<double>().minCoeff()/4);
    double eps = 1e-4*dx.minCoeff();

    /// Support Specifications - fix left edge of domain
    int suppInd = 0;
    topOpt->suppNode.resize((Nel(0)+1)*(Nel(1)+1));
    int loadInd = 0;
    topOpt->loadNode.resize(2);

    for (int nd = 0; nd < topOpt->nLocNode; nd++)
    {
      if (topOpt->node(nd,0) < eps)
        topOpt->suppNode(suppInd++) = nd;
      if (topOpt->node(nd,0) > Box(1)-eps)
        if (abs(topOpt->node(nd,1) - (Box(3)+Box(2))/2) - eps < dx(1)/2)
          topOpt->loadNode(loadInd++) = nd;

    }
    topOpt->loadNode.conservativeResize(loadInd);
    topOpt->loads.setZero(loadInd,2);
    topOpt->loads.col(1).setOnes();
    topOpt->loads *= -1;

    topOpt->suppNode.conservativeResize(suppInd);
    topOpt->supports.setOnes(suppInd,2);

    return;
  }
}
