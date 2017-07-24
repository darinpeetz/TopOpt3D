#ifndef DOMAIN_H_INCLUDED
#define DOMAIN_H_INCLUDED

#include <Eigen/Dense>
#include "EigLab.h"

namespace Domain
{
    Eigen::VectorXd Circle(const Eigen::MatrixXd &Nodes, const double xc, const double yc, const double r);
    Eigen::VectorXd Rectangle(const Eigen::MatrixXd &Nodes, const double xl, const double xr, const double yb, const double yt);
    Eigen::VectorXd Line(const Eigen::MatrixXd &Nodes, const double x1, const double y1, const double x2, const double y2);

    Eigen::VectorXd Union(const Eigen::VectorXd &d1, const Eigen::VectorXd &d2);
    Eigen::VectorXd Intersect(const Eigen::VectorXd &d1, const Eigen::VectorXd &d2);
    Eigen::VectorXd Difference(const Eigen::VectorXd &d1, const Eigen::VectorXd &d2);
}

#endif // DOMAIN_H_INCLUDED
