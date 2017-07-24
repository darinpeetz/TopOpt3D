#ifndef INPUT_H_INCLUDED
#define INPUT_H_INCLUDED

#include <Eigen/Dense>
#include "TopOpt.h"
#include "MMA.h"

namespace Input
{
  void Def_Param(TopOpt *topOpt, MMA *optmma, Eigen::VectorXd &Dimensions,
                 ArrayXPI &Nel, double &Rfactor,
                 Eigen::VectorXd &zIni);
  int Funcs (TopOpt *topOpt, double &f, Eigen::VectorXd &dfdx,
              Eigen::VectorXd &g, Eigen::MatrixXd &dgdx);
  void Domain(Eigen::ArrayXXd &Points, const Eigen::VectorXd &Box,
              Eigen::Array<bool, -1, 1> &elemValidity);
  void Def_BC(TopOpt *topOpt, const Eigen::VectorXd &Box,
              ArrayXPI Nel);
}

#endif // FEM_H_INCLUDED
