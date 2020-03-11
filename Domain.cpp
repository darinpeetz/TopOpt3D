#include "Domain.h"
#include <iostream>
#include <Eigen/Eigen>

namespace Domain
{
/********************************************************************
 * Determine if points are inside an ellipsoid
 * 
 * @param Nodes: Coordinates of the points to evaluate
 * @param xc: Centroid of the ellipsoid
 * @param r: Radius of the ellipsoid in each direction
 * 
 * @return Keep: Values < 1 indicate they are inside the ellipsoid
 * 
 *******************************************************************/
  ArrayXPS Ellipsoid(const Eigen::MatrixXd &Nodes, const ArrayXPS &xc, const ArrayXPS &r)
  {
    ArrayXPS Keep = (Nodes.col(0).array()-xc[0]).cwiseProduct(Nodes.col(0).array()-xc[0]) / (r[0]*r[0]);
    for (int i = 1; i < Nodes.cols(); i++)
      Keep += (Nodes.col(i).array()-xc[i]).cwiseProduct(Nodes.col(i).array()-xc[i]) / (r[i]*r[i]);
    return Keep - 1;
  }

/********************************************************************
 * Determine if points are inside a cylinder
 * 
 * @param Nodes: Coordinates of the points to evaluate
 * @param xc: Centroid of the bottom circle defining the cylinder
 * @param normal: Vector pointing from bottom to the top of the cylinder
 * @param h: Height of the cylinder
 * @param r: Radius of cylinder
 * 
 * @return Keep: Values < 1 indicate they are inside the cylinder
 * 
 *******************************************************************/
  ArrayXPS Cylinder(const Eigen::MatrixXd &Nodes, const ArrayXPS &xc, const VectorXPS &normal,
                     const double h, const double r)
  {
    MatrixXPS Offset = Nodes;
    for (int i = 0; i < Offset.cols(); i++)
      Offset.col(i).array() -= xc[i];
    VectorXPS Axial = Offset * normal.segment(0,Offset.cols());
    Offset -= Axial * normal.segment(0,Offset.cols()).transpose();
    VectorXPS Transverse = Offset.cwiseProduct(Offset).rowwise().sum().cwiseSqrt().array() - r;

    return Transverse.array().cwiseMax(Axial.cwiseAbs().array()-h);
  }

/********************************************************************
 * Determine if points are inside a hexahedron
 * 
 * @param Nodes: Coordinates of the points to evaluate
 * @param low: Coordinates lower corner of the hexahedron (x_min, y_min, z_min)
 * @param high: Coordinates upper corner of the hexahedron (x_max, y_max, z_max)
 * 
 * @return Keep: Values < 1 indicate they are inside the hexahedron
 * 
 *******************************************************************/
  ArrayXPS Hexahedron(const Eigen::MatrixXd &Nodes, const ArrayXPS &low, const ArrayXPS &high)
  {
    ArrayXPS Keep = (low[0] - Nodes.col(0).array()).cwiseMax(Nodes.col(0).array() - high[0]);
    for (int i = 1; i < Nodes.cols(); i++)
      Keep = Keep.cwiseMax((low[i] - Nodes.col(i).array()).cwiseMax(Nodes.col(i).array() - high[i]));
    return Keep;
  }

/********************************************************************
 * Determine if points are on one side of a plane or the other
 * 
 * @param Nodes: Coordinates of the points to evaluate
 * @param base, coordinates of a point on the plane
 * @param normal: Vector pointing to the "right" of the plane
 * 
 * @return Keep: Values < 1 indicate they are on the "left" of the plane
 * 
 *******************************************************************/
  ArrayXPS Plane(const Eigen::MatrixXd &Nodes, const ArrayXPS &base, const VectorXPS &normal)
  {
    MatrixXPS Offset = Nodes;
    for (int i = 0; i < Offset.cols(); i++)
      Offset.col(i).array() -= base[i];
    return (Offset * normal.segment(0,Offset.cols())).array();
  }

/********************************************************************
 * Union of two sets of points
 * 
 * @param d1: Indicator of whether points are in object 1 
 * @param d2: Indicator of whether points are in object 2
 * 
 * @return Keep: Values < 1 indicate they are inside either object
 * 
 *******************************************************************/
  ArrayXPS Union(const VectorXPS &d1, const VectorXPS &d2)
  {
    ArrayXPS dNew(d1.rows());
    for (int i = 0; i < d1.rows(); i++)
      dNew(i) = std::min(d1(i),d2(i));
    return dNew;
  }

/********************************************************************
 * Intersection of two sets of points
 * 
 * @param d1: Indicator of whether points are in object 1 
 * @param d2: Indicator of whether points are in object 2
 * 
 * @return Keep: Values < 1 indicate they are inside both objects
 * 
 *******************************************************************/
  ArrayXPS Intersect(const VectorXPS &d1, const VectorXPS &d2)
  {
    ArrayXPS dNew(d1.rows());
    for (int i = 0; i < d1.rows(); i++)
      dNew(i) = std::max(d1(i),d2(i));
    return dNew;
  }

/********************************************************************
 * Points in object 1 that are not in object 2
 * 
 * @param d1: Indicator of whether points are in object 1 
 * @param d2: Indicator of whether points are in object 2
 * 
 * @return Keep: Values < 1 indicate they are inside 1 but not 2
 * 
 *******************************************************************/
  ArrayXPS Difference(const VectorXPS &d1, const VectorXPS &d2)
  {
    ArrayXPS dNew(d1.rows());
    for (int i = 0; i < d1.rows(); i++) {
      if (d1(i)>0)
        dNew(i) = d1(i);
      else
        dNew(i) = -d2(i);
    }
    return dNew;
  }

}
