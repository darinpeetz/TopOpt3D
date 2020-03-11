#ifndef DOMAIN_H_INCLUDED
#define DOMAIN_H_INCLUDED


#include <slepceps.h>
#include <vector>
#include <Eigen/Dense>
#include "EigLab.h"

typedef Eigen::Array<PetscScalar, -1, 1> ArrayXPS;
typedef Eigen::Matrix<PetscScalar, -1, -1> MatrixXPS;
typedef Eigen::Matrix<PetscScalar, -1, 1> VectorXPS;

namespace Domain
{
  ArrayXPS Ellipsoid(const Eigen::MatrixXd &Nodes, const ArrayXPS &xc,
                     const ArrayXPS &r);
  ArrayXPS Cylinder(const Eigen::MatrixXd &Nodes, const ArrayXPS &xc,
                    const VectorXPS &normal, const double h, const double r);
  ArrayXPS Hexahedron(const Eigen::MatrixXd &Nodes, const ArrayXPS &low,
                     const ArrayXPS &high);
  ArrayXPS Plane(const Eigen::MatrixXd &Nodes, const ArrayXPS &base,
                 const VectorXPS &normal);

  ArrayXPS Union(const VectorXPS &d1, const VectorXPS &d2);
  ArrayXPS Intersect(const VectorXPS &d1, const VectorXPS &d2);
  ArrayXPS Difference(const VectorXPS &d1, const VectorXPS &d2);
}

#endif // DOMAIN_H_INCLUDED
