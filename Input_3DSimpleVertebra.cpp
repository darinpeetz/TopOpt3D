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

namespace Input
{
  /// An extra function for specifying the location of the annulus and nucleus
  void AnnulusNucleus(Eigen::ArrayXXd &Points, const Eigen::VectorXd &Box,
                      Eigen::Array<bool, -1, 1> &annulus,
                      Eigen::Array<bool, -1, 1> &nucleus, double buffer);

  void Def_Param(TopOpt *topOpt, MMA *optmma, Eigen::VectorXd &Dimensions,
                 ArrayXPI &Nel, double &R, Eigen::VectorXd &zIni)
  {
      Dimensions.resize(6,1);
      Dimensions << 0, 43, 0, 40, 0, 30;
      Nel.resize(3,1);
      Nel << 86, 80, 60;
      topOpt->Nu0 = 0.3; topOpt->E0 = 1e7;
      topOpt->density = 1000;
      topOpt->direct = false;

      topOpt->pmin = 3.0; topOpt->pstep = 0.25; topOpt->pmax = 3.0;

      optmma->Set_KKT_Limit(0);
      optmma->Set_Iter_Limit_Min(5);
      optmma->Set_Iter_Limit_Max(100);
      optmma->Set_Change_Limit(0.0001);

      R = 1.5*(Dimensions(1)-Dimensions(0))/Nel(0);
  }

  int Funcs (TopOpt *topOpt, double &f, Eigen::VectorXd &dfdx,
              Eigen::VectorXd &g, Eigen::MatrixXd &dgdx)
  {
    /// Determines objective function, constraint functions from within program
    //PetscInt nevals = 10;

    f = 0;
    dfdx.setZero(topOpt->nLocElem);
    g.setZero(1);
    dgdx.setZero(topOpt->nLocElem,1);

    double temp;
    Eigen::VectorXd dtemp = Eigen::VectorXd::Zero(topOpt->nLocElem);
    Functions::Compliance( topOpt, temp, dtemp.data() );
    f += 0.07*temp;
    dfdx += 0.07*dtemp;
    Functions::Perimeter( topOpt, temp, dtemp.data() );
    f -= 0.93*temp;
    dfdx -= 0.93*dtemp;

    /* /// Buckling subproblem
    PetscOptionsGetInt(NULL,"-nev",&nevals,NULL);
    //Functions::Buckling( topOpt, g.data(), dgdx.data(), nevals );
    //g.segment(0, nevals) *= 10; dgdx.block(0, 0, topOpt->nLocElem, nevals) *= 10;
    for (short i = nevals-1; i > 0; i--)
        g(i) -= 0.99*g(i-1);
    f = g(0); dfdx = dgdx.col(0); */

     /// Dynamic subproblem
    /*PetscOptionsGetInt(NULL,"-nev",&nevals,NULL);
    Functions::Dynamic( topOpt, g.data(), dgdx.data(), nevals );
    //g.segment(0, nevals) *= 10; dgdx.block(0, 0, topOpt->nLocElem, nevals) *= 10;
    for (short i = nevals-1; i > 0; i--)
        g(i) -= 0.99*g(i-1);
    f = g(0); dfdx = dgdx.col(0);*/


    /// Volume Subproblem
    Functions::Volume( topOpt, g(0), dgdx.data() );
    g(0) -= 0.2;

    return 0;
  }

  void Domain(Eigen::ArrayXXd &Points, const Eigen::VectorXd &Box,
              Eigen::Array<bool, -1, 1> &elemValidity)
  {
    // Find which elements are in the annulus and which are in the nucleus
    Eigen::Array<bool, -1, 1> annulus, nucleus;
    AnnulusNucleus(Points, Box, annulus, nucleus, 0);

    // For constructing the domain, it doesn't matter which group the elements
    // is in
    elemValidity = annulus || nucleus;

    return;
  }

  void Def_BC(TopOpt *topOpt, const Eigen::VectorXd &Box, ArrayXPI Nel)
  {
    // Only check nodes owned by this process
    Eigen::ArrayXXd Points = Eigen::ArrayXXd::Zero(topOpt->nLocElem, topOpt->numDims);
    for (int el = 0; el < topOpt->nLocElem; el++)
    {
      for (int nd = 0; nd < topOpt->element.cols(); nd++)
        Points.row(el) += topOpt->node.row(topOpt->element(el,nd)).array();
      Points.row(el) /= topOpt->element.cols();
    }

    // Find which elements are in the annulus and which are in the nucleus
    Eigen::ArrayXd dx(3);
    dx << (Box(1)-Box(0))/Nel(0), (Box(3)-Box(2))/Nel(1), (Box(5)-Box(4))/Nel(2);
    double buffer = 3*dx.maxCoeff();
    Eigen::Array<bool, -1, 1> annulus, nucleus;
    AnnulusNucleus(Points, Box, annulus, nucleus, buffer);

    // Fill in load values for nodes on edge of domain
    MatrixXdRM tempLoad = MatrixXdRM::Zero(topOpt->node.rows(),3);
    double eps = 1e-10*(Box(1)-Box(0))*(Box(3)-Box(2))*(Box(5)-Box(4));
    for (int el = 0; el < topOpt->nLocElem; el++)
    {
      if (annulus(el) || nucleus(el))
      {
        for (int nd = 0; nd < topOpt->element.cols(); nd++)
        {
          PetscInt node = topOpt->element(el,nd);
          if ( std::abs(topOpt->node(node,2) - Box(4)) < eps )
            tempLoad(node, 2) += 0.25;
          else if ( std::abs(topOpt->node(node,2) - Box(5)) < eps )
            tempLoad(node, 2) -= 0.25;
        }
      }
    }

    // Construct vectors to share load values with processes that own non-local nodes
    std::vector<double> *nonLocalLoads = new std::vector<double>[topOpt->nprocs];
    std::vector<PetscInt> *nonLocalNodes = new std::vector<PetscInt>[topOpt->nprocs];
    int ind = 0;
    for (int i = topOpt->nLocNode; i < tempLoad.rows(); i++)
    {
      if (tempLoad.row(i).cwiseAbs().sum() != 0)
      {
        PetscInt globalNum = topOpt->gNode(i);
        int proc = 0;
        while (globalNum >= topOpt->nddist(proc+1))
          proc++;
        nonLocalLoads[proc].insert(nonLocalLoads[proc].end(), tempLoad.data() +
              topOpt->numDims*i, tempLoad.data() + topOpt->numDims*(i+1));
        nonLocalNodes[proc].push_back(globalNum);
      }
    }

    /// Perform all the sends
    MPI_Request *requests = new MPI_Request[topOpt->nprocs];
    requests[topOpt->myid] = MPI_REQUEST_NULL;
    for (short proc = 0; proc < topOpt->nprocs; proc++)
    {
      MPI_Issend(nonLocalNodes[proc].data(), nonLocalNodes[proc].size(),
                 MPI_PETSCINT, proc, 0, topOpt->comm, requests + proc);
      MPI_Issend(nonLocalLoads[proc].data(), nonLocalLoads[proc].size(),
                 MPI_DOUBLE, proc, 1, topOpt->comm, requests + proc);
    }

    /// Receive all the Information
    // First probe all messsages to see how much the elements array needs to be
    // expanded by, then receive all messages in the new buffer
    Eigen::ArrayXi flags = Eigen::ArrayXi::Zero(topOpt->nprocs);
    MPI_Status *statuses = new MPI_Status[topOpt->nprocs];
    Eigen::ArrayXi recvCount(topOpt->nprocs);
    flags(topOpt->myid) = 1; recvCount(topOpt->myid) = 0;
    while (flags.sum() < topOpt->nprocs)
    {
      for (short proc = 0; proc < topOpt->nprocs; proc++)
      {
        if (proc != topOpt->myid && flags(proc) == 0)
        {
          // Receive either the number or node message
          MPI_Iprobe(proc, MPI_ANY_TAG, topOpt->comm, flags.data()+proc, statuses+proc);
          // Determine number of elements contained in this message
          if (statuses[proc].MPI_TAG == 0)
            MPI_Get_count(statuses+proc, MPI_PETSCINT, recvCount.data() + proc);
          if (statuses[proc].MPI_TAG == 1)
          {
            MPI_Get_count(statuses+proc, MPI_DOUBLE, recvCount.data() + proc);
            recvCount(proc) /= topOpt->numDims;
          }
        }
      }
    }

    // Now do the receives
    ArrayXPI recvNodes(recvCount.sum());
    MatrixXdRM recvLoads(recvCount.sum(), topOpt->numDims);
    ind = 0;
    for (short proc = 0; proc < topOpt->nprocs; proc++)
    {
      if (proc == topOpt->myid)
        continue;
      MPI_Recv(recvNodes.data() + ind, recvCount(proc), MPI_PETSCINT,
               proc, 0, topOpt->comm, MPI_STATUS_IGNORE);
      MPI_Recv(recvLoads.data() + ind*topOpt->numDims, topOpt->numDims*recvCount(proc),
               MPI_DOUBLE, proc, 1, topOpt->comm, MPI_STATUS_IGNORE);
      ind += recvCount(proc);
    }

    /// Add the received information to the previous array
    for (int i = 0; i < recvNodes.rows(); i++)
      tempLoad.row(recvNodes(i)-topOpt->nddist(topOpt->myid)) += recvLoads.row(i);

    // Remove zeros from the array
    ind = 0;
    topOpt->loadNode.resize(tempLoad.rows());
    for (int i = 0; i < topOpt->nLocNode; i++)
    {
      if (tempLoad.row(i).cwiseAbs().sum() != 0)
      {
        topOpt->loadNode(ind) = i;
        tempLoad.row(ind++) = tempLoad.row(i);
      }
    }
    topOpt->loadNode.conservativeResize(ind);
    topOpt->loads = tempLoad.block(0, 0, ind, topOpt->numDims);

    /// Spring Specifications - use same nodes as loads
    topOpt->springNode = topOpt->loadNode;
    topOpt->springs = 2*Eigen::Array<double, -1, 3>::Ones(topOpt->springNode.rows(),3);

    return;
  }

  /// An extra function for specifying the location of the annulus and nucleus
  void AnnulusNucleus(Eigen::ArrayXXd &Points, const Eigen::VectorXd &Box,
                      Eigen::Array<bool, -1, 1> &annulus,
                      Eigen::Array<bool, -1, 1> &nucleus, double buffer)
  {
    double width = Box(1) - Box(0) - 2*buffer;
    double length = Box(3) - Box(2) - 2*buffer;
    double height = Box(5) - Box(4) - 2*buffer;
    double radius;
    Eigen::Vector3d center;
    center(0) = (Box(0)+Box(1))/2;
    center(1) = (Box(2)+Box(3))/2;
    center(2) = (Box(4)+Box(5))/2;

    /// See .dwg file for drawing of domain and boundary Conditions
    /// Need to scale dimensions appropriately so that a box sized 43x32x30
    /// fits in the domain and then allow for 2 element boundary
    double factor = std::max(std::max(43/width,40/length),30/height);
    for (short dim = 0; dim < 3; dim++)
    {
      Points.col(dim) = (Points.col(dim) - center(dim))*factor + center(dim);
    }
    buffer = 0;

    annulus.setOnes(Points.rows());

    /// Big circle
    radius = 21.5;
    center << Box(0) + 21.5, Box(2) + 21.5, 0;
    annulus = annulus && (((Points.col(0)-center(0)).square() +
                         (Points.col(1)-center(1)).square()) < pow(radius,2));

    /// Cutout for canal
    radius = 13;
    center << Box(0) + 21.5, Box(3) + 5, 0;
    annulus = annulus && (((Points.col(0)-center(0)).square() +
                         (Points.col(1)-center(1)).square()) > pow(radius,2));

    // Just set nucleus equal to the annulus
    nucleus = annulus;
    return;
  }
}
