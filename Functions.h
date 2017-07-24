#ifndef FUNCTIONS_H_INCLUDED
#define FUNCTIONS_H_INCLUDED

#include "TopOpt.h"

namespace Functions
{
    int FunctionCall( TopOpt *topOpt, double &f, Eigen::VectorXd &dfdx,
                      Eigen::VectorXd &g, Eigen::MatrixXd &dgdx );
    int Compliance( TopOpt *topOpt, double &obj, double *grad );
    int Volume( TopOpt *topOpt, double &obj, double *grad );
    int Perimeter( TopOpt *topOpt, double &obj, double *grad );
    int Buckling( TopOpt *topOpt, double *obj, double *grad, PetscInt &nevals );
    int Dynamic( TopOpt *topOpt, double *lambda, double *grad, PetscInt &nevals );
}

#endif // FUNCTIONS_H_INCLUDED
