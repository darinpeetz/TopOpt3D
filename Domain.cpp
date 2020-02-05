#include "Domain.h"
#include <iostream>
#include <Eigen/Eigen>

namespace Domain
{
  ArrayXPS Ellipsoid(const Eigen::MatrixXd &Nodes, const ArrayXPS &xc, const ArrayXPS &r)
  {
    ArrayXPS Keep = (Nodes.col(0).array()-xc[0]).cwiseProduct(Nodes.col(0).array()-xc[0]) / (r[0]*r[0]);
    for (int i = 1; i < Nodes.cols(); i++)
      Keep += (Nodes.col(i).array()-xc[i]).cwiseProduct(Nodes.col(i).array()-xc[i]) / (r[i]*r[i]);
    return Keep - 1;
  }

  ArrayXPS Cylinder(const Eigen::MatrixXd &Nodes, const ArrayXPS &xc, const VectorXPS &normal,
                     const double h, const double r)
  {
    MatrixXPS Offset = Nodes;
    for (int i = 0; i < Offset.cols(); i++)
      Offset.col(i).array() -= xc[i];
    VectorXPS Axial = Offset * normal;
    Offset -= Axial * normal.transpose();
    VectorXPS Transverse = Offset.cwiseProduct(Offset).rowwise().sum().cwiseSqrt().array() - r;

    return Transverse.array().cwiseMax(Axial.cwiseAbs().array()-h);
  }

  ArrayXPS Hexahedron(const Eigen::MatrixXd &Nodes, const ArrayXPS &low, const ArrayXPS &high)
  {
    ArrayXPS Keep = (low[0] - Nodes.col(0).array()).cwiseMax(Nodes.col(0).array() - high[0]);
    for (int i = 1; i < Nodes.cols(); i++)
      Keep = Keep.cwiseMax((low[i] - Nodes.col(i).array()).cwiseMax(Nodes.col(i).array() - high[i]));
    return Keep;
  }

  ArrayXPS Plane(const Eigen::MatrixXd &Nodes, const ArrayXPS &base, const VectorXPS &normal)
  {
    MatrixXPS Offset = Nodes;
    for (int i = 0; i < Offset.cols(); i++)
      Offset.col(i).array() -= base[i];
    return (Offset * normal).array();
  }

  ArrayXPS Union(const VectorXPS &d1, const VectorXPS &d2)
  {
    ArrayXPS dNew(d1.rows());
    for (int i = 0; i < d1.rows(); i++)
      dNew(i) = std::min(d1(i),d2(i));
    return dNew;
  }

  ArrayXPS Intersect(const VectorXPS &d1, const VectorXPS &d2)
  {
    ArrayXPS dNew(d1.rows());
    for (int i = 0; i < d1.rows(); i++)
      dNew(i) = std::max(d1(i),d2(i));
    return dNew;
  }

  ArrayXPS Difference(const VectorXPS &d1, const VectorXPS &d2)
  {
    ArrayXPS dNew(d1.rows());
    for (int i = 0; i < d1.rows(); i++)
    {
      if (d1(i)>0)
        dNew(i) = d1(i);
      else
        dNew(i) = -d2(i);
    }
    return dNew;
  }

}
