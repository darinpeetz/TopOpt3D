#ifndef RECMESH_H_INCLUDED
#define RECMESH_H_INCLUDED

#include <Eigen/Eigen>
#include "TopOpt.h"
#include "mpi.h"
extern "C"
{
#include "parmetis.h"
}

namespace RecMesh
{
    void Node_Elem ( ArrayXXPI &element, ArrayXXPI &ndEl, PetscInt nElem,
                     ArrayXPI &elmdist, ArrayXPI &nddist, MPI_Comm Comm );
}

#endif // RECMESH_H_INCLUDED
