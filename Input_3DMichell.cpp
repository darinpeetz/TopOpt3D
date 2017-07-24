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

/// 3D Michell input file
namespace Input
{
  void Def_Param(TopOpt *topOpt, MMA *optmma, Eigen::VectorXd &Dimensions,
                 Eigen::Array<PetscInt, 3, 1> &Nel, double &R,
                 Eigen::VectorXd &zIni)
  {
      Dimensions.resize(6,1);
      Dimensions << 0, 2, 0, 1, 0, 1;
      Nel.resize(3,1);
      Nel << 40, 20, 20;
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
    g.setZero(1);
    dgdx.setZero(topOpt->nLocElem,1);

    Functions::Compliance( topOpt, f, dfdx.data() );
    /*Functions::Perimeter( topOpt, g(1), dgdx.data()+topOpt->locel, Comm );
    if (req == 2) /// Get final results and clean up
        PetscFPrintf(Comm, "Results.txt", "Perimeter value:\n%1.12g\n", g(1));
    g(1) += 4;*/

    /* /// Buckling subproblem
    PetscOptionsGetInt(NULL,"-nev",&nevals,NULL);
    //Functions::Buckling( topOpt, g.data(), dgdx.data(), nevals, Comm );
    //g.segment(0, nevals) *= 10; dgdx.block(0, 0, topOpt->locel, nevals) *= 10;
    for (short i = nevals-1; i > 0; i--)
        g(i) -= 0.99*g(i-1);
    f = g(0); dfdx = dgdx.col(0); */

    /// Volume Subproblem
    Functions::Volume( topOpt, g(0), dgdx.data() );
    g(0) -= 0.4;

    return 0;
  }

  void Domain(Eigen::ArrayXXd &Points, const Eigen::VectorXd &Box,
              Eigen::Array<bool, -1, 1> &elemValidity)
  {
    elemValidity.setOnes(Points.rows());

    return;
  }

  void Def_BC(TopOpt *topOpt, const Eigen::VectorXd &Box,
              const Eigen::Array<PetscInt, 3, 1> Nel)
  {
    Eigen::ArrayXd dx(3);
    dx << (Box(1)-Box(0))/Nel(0), (Box(3)-Box(2))/Nel(1), (Box(5)-Box(4))/Nel(2);
    double eps = 1e-4*dx.minCoeff();
    int loadInd = 0;
    topOpt->loadNode.resize(Nel(1)+1);

    /// Support Specifications - fix left edge of domain
    int suppInd = 0;
    topOpt->suppNode.resize((Nel(1)+1)*(Nel(2)+1));
    for (int nd = 0; nd < topOpt->nLocNode; nd++)
    {
      if (topOpt->node(nd,2) < eps && abs(topOpt->node(nd,0)-Box(1)) < eps)
      {
        //if (topOpt->node(nd,1) > 2.5*dx(1) && topOpt->node(nd,1)<(Box(3)-2.5*dx(1)) )
          topOpt->loadNode(loadInd++) = nd;
      }
      if (topOpt->node(nd,0) < eps)
        topOpt->suppNode(suppInd++) = nd;

    }
    topOpt->loadNode.conservativeResize(loadInd);
    topOpt->loads.setZero(loadInd,3);
    topOpt->loads.col(2).setOnes();
    topOpt->loads *= -1;

    topOpt->suppNode.conservativeResize(suppInd);
    topOpt->supports.setOnes(suppInd,3);

    return;
  }
}
