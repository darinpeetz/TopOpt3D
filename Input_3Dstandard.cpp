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
      Dimensions << 0, 1, 0, 1, 0, 1;
      Nel.resize(3,1);
      Nel << 10, 10, 10;
      topOpt->Nu0 = 0.3; topOpt->E0 = 1e7;

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
    f *= 100; dfdx *= 100;
    Functions::Perimeter( topOpt, g(1), dgdx.data()+topOpt->nLocElem );
    g(1) += 100;*/

    /*/// Buckling subproblem
    PetscOptionsGetInt(NULL,"-nev",&nevals,NULL);
    Functions::Buckling( topOpt, g.data(), dgdx.data(), nevals );
    g.segment(0, nevals); dgdx.block(0, 0, topOpt->nLocElem, nevals);
    for (short i = nevals-1; i > 0; i--)
        g(i) -= 0.99*g(i-1);
    f = g(0); dfdx = dgdx.col(0);*/

    /// Dynamic subproblem
    PetscOptionsGetInt(NULL,"-nev",&nevals,NULL);
    Functions::Dynamic( topOpt, g.data(), dgdx.data(), nevals );
    g.segment(0, nevals); dgdx.block(0, 0, topOpt->nLocElem, nevals);
    for (short i = nevals-1; i > 0; i--)
        g(i) -= 0.99*g(i-1);
    f = g(0); dfdx = dgdx.col(0);
MPI_Barrier(topOpt->comm);
MPI_Abort(topOpt->comm, 42);
    /// Volume Subproblem
    Functions::Volume( topOpt, g(0), dgdx.data() );
    g(0) -= 0.5; g(0) *= 100; dgdx *= 100;

    return 0;
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
int a = 1;
std::cout << a++ << "\n";
    /// Support Specifications - fix left edge of domain
    int suppInd = 0;
    topOpt->suppNode.resize((Nel(0)+1)*(Nel(1)+1)*(Nel(2))+1);
    int loadInd = 0;
    topOpt->loadNode.resize(2);
std::cout << a++ << "\n";
    for (int nd = 0; nd < topOpt->nLocNode; nd++)
    {
      if (topOpt->node(nd,1) < eps)
        topOpt->suppNode(suppInd++) = nd;
      if (topOpt->node(nd,1) > Box(3)-eps)
        if (topOpt->node(nd,0) < eps || topOpt->node(nd,0) > Box(1)-eps)
          topOpt->loadNode(loadInd++) = nd;

    }
    topOpt->loadNode.conservativeResize(loadInd);
    topOpt->loads.setZero(loadInd,2);
    topOpt->loads.col(1).setOnes();
    topOpt->loads *= -1;
std::cout << a++ << "\n";
    topOpt->suppNode.conservativeResize(suppInd);
    topOpt->supports.setOnes(suppInd,2);

    return;
  }
}
