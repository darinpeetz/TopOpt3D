#include "Domain.h"
#include <iostream>
#include <Eigen/Eigen>

namespace Domain
{
    Eigen::VectorXd Circle(const Eigen::MatrixXd &Nodes, const double xc, const double yc, const double r)
    {
        Eigen::VectorXd Keep(Nodes.rows());
        for (long int i = 0; i < Nodes.rows(); i++)
            Keep(i) = (Nodes(i, 0)-xc)*(Nodes(i, 0)-xc) + (Nodes(i, 1)-yc)*(Nodes(i, 1)-yc) - r*r;
        return Keep;
    }

    Eigen::VectorXd Rectangle(const Eigen::MatrixXd &Nodes, const double xl, const double xr, const double yb, const double yt)
    {
        Eigen::VectorXd Keep(Nodes.rows());
        for (long int i = 0; i < Nodes.rows(); i++)
            Keep(i) = std::max(std::max(std::max(xl-Nodes(i,0),Nodes(i,0)-xr),yb-Nodes(i,1)),Nodes(i,1)-yt);
        return Keep;
    }

    Eigen::VectorXd Line(const Eigen::MatrixXd &Nodes, const double x1, const double y1, const double x2, const double y2)
    {
        Eigen::Vector2d a;
        Eigen::VectorXd Keep(Nodes.rows());
        Eigen::MatrixXd b(Nodes.rows(),Nodes.cols());
        a << x2-x1, y2-y1;
        a = a/a.norm();
        b.col(0) = Nodes.col(0)-Eigen::VectorXd::Constant(Nodes.rows(),x1); b.col(1) = Nodes.col(1)-Eigen::VectorXd::Constant(Nodes.rows(),y1);
        Keep = b.col(0)*a(1)-b.col(1)*a(0);
        return Keep;
    }

    Eigen::VectorXd Union(const Eigen::VectorXd &d1, const Eigen::VectorXd &d2)
    {
        Eigen::VectorXd dNew(d1.rows());
        for (long int i = 0; i < d1.rows(); i++)
            dNew(i) = std::min(d1(i),d2(i));
        return dNew;
    }

    Eigen::VectorXd Intersect(const Eigen::VectorXd &d1, const Eigen::VectorXd &d2)
    {
        Eigen::VectorXd dNew(d1.rows());
        for (long int i = 0; i < d1.rows(); i++)
            dNew(i) = std::max(d1(i),d2(i));
        return dNew;
    }

    Eigen::VectorXd Difference(const Eigen::VectorXd &d1, const Eigen::VectorXd &d2)
    {
        Eigen::VectorXd dNew(d1.rows());
        for (long int i = 0; i < d1.rows(); i++)
        {
            if (d1(i)>0)
                dNew(i) = d1(i);
            else
                dNew(i) = -d2(i);
        }
        return dNew;
    }

}
