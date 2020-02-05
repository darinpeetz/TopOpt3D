#include "TopOpt.h"
#include <numeric>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

typedef Eigen::Array<PetscInt, -1, -1> ArrayXXPI;
typedef Eigen::Array<PetscInt, -1, 1> ArrayXPI;
typedef Eigen::Array<PetscScalar, -1, 1> ArrayXPS;
typedef Eigen::Matrix<PetscScalar, -1, -1> MatrixXPS;

/// This method calls Create_Interpolation repeatedly to construct all interpolation
/// matrices for the full multigrid hierarchy.  The matrices are not assembled to make
/// any renumberings from element redistribution easier.
// I: row indices (the fine nodes)
// J: col indices (the coarse nodes)
// K: interpolation coefficients
// cList: the coarse nodes on each level
int TopOpt::Create_Interpolations( PetscInt *first, PetscInt *last, ArrayXPI Nel,
            ArrayXPI *I, ArrayXPI *J, ArrayXPS *K, ArrayXPI *cList, PetscInt mg_levels )
{
  // Number of nodes in each direction on current grid (Nf), and node
  // breakdown on this process (firstf and lastf)
  ArrayXPI Nf = Nel;
  ArrayXPI firstf(3), lastf(3);
  copy(first, first+3, firstf.data()); copy(last, last+3, lastf.data());
  Nf.segment(0,numDims) += 1;
  if (myid != 0)
    firstf[numDims-1]++;
  lastf.segment(0,numDims) += 1;

  // Coarse node numbers in each direction
  ArrayXPI xBase = ArrayXPI::LinSpaced(Nf(0), 0, Nf(0)-1);
  ArrayXPI yBase = ArrayXPI::LinSpaced(Nf(1), 0, Nf(1)-1);
  ArrayXPI zBase = ArrayXPI::LinSpaced(Nf(2), 0, Nf(2)-1);

  for (int i = 0; i < mg_levels-1; i++)
  {  
    // Create the interpolation triplets for this restriction

    PetscErrorCode ierr = Create_Interpolation(firstf, lastf,
                                               Nf, I[i], J[i], K[i]);
    CHKERRQ(ierr);
    // Setup the node breakdown on the new coarse grid
    firstf = (firstf+1)/2;
    lastf.segment(0,numDims) = (lastf.segment(0,numDims)+1)/2;
    // Replace relative node numberings in these arrays with actual node numbers
    // Fine nodes first (if necessary)
    if (i > 0)
    {
      for (int j = 0; j < I[i].size(); j++)
        I[i](j) = cList[i-1](I[i](j));
    }

    // Make the new conversion list
    for (int j = 0; j < xBase.size(); j+=2)
      xBase(j/2) = xBase(j);
    xBase.conservativeResize((xBase.size()+1)/2);
    for (int j = 0; j < yBase.size(); j+=2)
      yBase(j/2) = yBase(j);
    yBase.conservativeResize((yBase.size()+1)/2);
    for (int j = 0; j < zBase.size(); j+=2)
      zBase(j/2) = zBase(j);
    zBase.conservativeResize((zBase.size()+1)/2);

    cList[i] = xBase.replicate((int)yBase.size(),1).replicate(zBase.size(),1);
    ArrayXXPI temp = (Nel(0)+1)*yBase.transpose().replicate(xBase.size(),1);
    temp.resize(temp.size(),1);
    cList[i] += temp.replicate(zBase.size(),1);
    temp = (Nel(0)+1)*(Nel(1)+1)*zBase.transpose()
           .replicate(xBase.size(),1).replicate(yBase.size(),1);
    temp.resize(temp.size(),1);
    cList[i] += temp;

    // Apply the new list to the coarse nodes
    for (int j = 0; j < J[i].size(); j++) 
      J[i](j) = cList[i](J[i](j));
  }

  return 0;
}

/// This method produces creates i,j,k indices to assemble an interpolation
/// matrix from a fine to a coarse grid.  The matrix is not assembled to make
/// any renumberings from element redistribution easier
int TopOpt::Create_Interpolation ( ArrayXPI &first, ArrayXPI &last,
                     ArrayXPI &Nf, ArrayXPI &I, ArrayXPI &J, ArrayXPS &K )
{
  ArrayXPI Nc = (Nf+1)/2;
  if ((Nf < last).any())
    SETERRQ6(comm, PETSC_ERR_ARG_INCOMP, "Local elements (%i, %i, %i)  extend beyond total elements(%i, %i, %i)",
            last(0), last(1), last(2), Nf(0), Nf(1), Nf(2));

  PetscInt numvals = pow(3, numDims)*Nc(0)*Nc(1)*Nc(2);
  I.resize(numvals); J.resize(numvals); K.resize(numvals); 
  ArrayXPS vals(3); vals << 0.5, 1, 0.5;

  // Loop over local COARSE nodes
  PetscInt ind = 0;
  PetscInt ckmax = last(2), ckmin = ((first(2)+1)/2)*2;//(last[2]==Nf(2) ? last[2]+1 : last[2]);
  PetscInt cjmax = last(1), cjmin = ((first(1)+1)/2)*2;//(last[1]==Nf(1) ? last[1]+1 : last[1]);
  PetscInt cimax = last(0), cimin = ((first(0)+1)/2)*2;//(last[0]==Nf(0) ? last[0]+1 : last[0]);
  for (PetscInt ck = ckmin; ck < ckmax; ck+=2)
  {
    for (PetscInt cj = cjmin; cj < cjmax; cj+=2)
    {
      for (PetscInt ci = cimin; ci < cimax; ci+=2)
      {
        PetscInt cnode = ci/2 + (cj/2)*Nc(0) + (ck/2)*Nc(0)*Nc(1);
        PetscInt fkmin = ck > 0 ? -1 : 0;
        PetscInt fkmax = ck < Nf[2]-1 ? 1 : 0;
        PetscInt fjmin = cj > 0 ? -1 : 0;
        PetscInt fjmax = cj < Nf[1]-1 ? 1 : 0;
        PetscInt fimin = ci > 0 ? -1 : 0;
        PetscInt fimax = ci < Nf[0]-1 ? 1 : 0;
        for (PetscInt fk = fkmin; fk <= fkmax; fk++)
        {
          for (PetscInt fj = fjmin; fj <= fjmax; fj++)
          {
            for (PetscInt fi = fimin; fi <= fimax; fi++)
            {
              I(ind) = (ci+fi) + (cj+fj)*Nf(0) + (ck+fk)*Nf(0)*Nf(1);
              J(ind) = cnode;
              K(ind++) = vals(fi+1) * vals(fj+1) * vals(fk+1);
            }
          }
        }
      }
    }
  }

  I.conservativeResize(ind); J.conservativeResize(ind); K.conservativeResize(ind);
  Nf = Nc;

  return 0;
}

/// This method assembles the interpolation matrices
int TopOpt::Assemble_Interpolation ( ArrayXPI *I, ArrayXPI *J, ArrayXPS *K, ArrayXPI *cList, PetscInt mg_levels, PetscInt min_size )
{
  PetscErrorCode ierr = 0;
  // Preallocation arrays
  PetscInt *onDiag  = new PetscInt[nNode];
  PetscInt *offDiag = new PetscInt[nNode];
  // Matrix distribution information
  ArrayXPI lRows = nddist, lCols(nprocs+1);
  // Renumbering information
  ArrayXPI invind(nNode), ind(cList[0].size());
  // MPI Reduction Requests
  MPI_Request on_req, off_req;

  // List of interpolation matrices
  this->PR.resize(mg_levels-1);
  // List of communicators on each level
  this->MG_comms.resize(mg_levels);
  this->MG_comms[0] = this->comm;
  // How the dof are split between processors at each level
  ArrayXPI nddist = this->nddist;

  if (min_size <= 0)
    min_size = cList[mg_levels-2].size();
  
  for (int i = 0; i < mg_levels-1; i++)
  {
    // Sort all the coarse nodes to determine their numbers at this level
    // e.g. fine node numbers are 8,1,5, so need to set coarse numbers to 3,1,2
    ind.segment(0,cList[i].size()) = ArrayXPI::LinSpaced(cList[i].size(), 0, cList[i].size()-1);
    ierr = PetscSortIntWithPermutation(cList[i].size(), cList[i].data(), ind.data());
    CHKERRQ(ierr);
    for (PetscInt j = 0; j < cList[i].size(); j++)
      invind(cList[i](ind(j))) = j;

    // Create Projection matrices and define global sizes
    PetscInt gRows = (i==0 ? nNode : cList[i-1].size());
    PetscInt gCols = cList[i].size();
    ierr = MatCreate(comm, this->PR.data()+i); CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(this->PR[i], "PR_"); CHKERRQ(ierr);
    ierr = MatSetFromOptions(this->PR[i]); CHKERRQ(ierr);

    // Create communicators for each level of hierarchy
    PetscInt nprocs = std::max(gCols/min_size,1);
    if (this->verbose >= 2)
      ierr = PetscPrintf(comm, "Level %i is split over %i processors\n", i,
                         min(this->nprocs, nprocs)); CHKERRQ(ierr);
    if (i == mg_levels-2)
      MPI_Comm_split(this->comm, myid == 0 ? 0 : MPI_UNDEFINED, 0, this->MG_comms.data()+i+1);
    if (nprocs < this->nprocs)
    {
      MPI_Comm_split(this->comm, myid < nprocs ? 0 : MPI_UNDEFINED, 0, this->MG_comms.data()+i+1);
      nddist.setConstant(this->nNode);
      nddist.segment(0, nprocs+1) = ArrayXPI::LinSpaced(nprocs+1, 0, this->nNode);
    }
    else
      this->MG_comms[i] = this->comm;

    // Initialize preallocation arrays
    fill(onDiag, onDiag+gRows, 0);
    fill(offDiag, offDiag+gRows, 0);

    // Set local row array
    if (i > 0)
      lRows = lCols;

    // Coarsest interpolator works a little different
    if (i == mg_levels-2)
    {
      // Set matrix size
      lCols.setConstant(gCols); lCols(0) = 0;
      ierr = MatSetSizes(this->PR[i], numDims*(lRows(myid+1)-lRows(myid)),
                 numDims*(lCols(myid+1)-lCols(myid)), numDims*gRows, numDims*gCols);
      CHKERRQ(ierr);

      // Get Preallocation sizes
      for (int j = 0; j < I[i].size(); j++)
        onDiag[I[i](j)]++;
      MPI_Iallreduce(MPI_IN_PLACE, onDiag, gRows, MPI_PETSCINT, MPI_SUM, comm, &on_req);

      // Update I and J arrays with level-specific node numbers
      for (int j = 0; j < J[i].size(); j++)
        J[i](j) = invind(J[i](j));

      // Want coarsest matrix to be all on proc 0
      if (myid == 0)
      {
        ArrayXPI zero = ArrayXPI::Zero(gRows);
        MPI_Wait(&on_req, MPI_STATUS_IGNORE);
        ierr = MatXAIJSetPreallocation(this->PR[i], this->numDims, onDiag+lRows(myid), zero.data(), 0, 0); CHKERRQ(ierr);
      }
      else
      {
        ArrayXPI zero = ArrayXPI::Zero(gRows);
        MPI_Wait(&on_req, MPI_STATUS_IGNORE);
        ierr = MatXAIJSetPreallocation(this->PR[i], this->numDims, zero.data(), onDiag+lRows(myid), 0, 0); CHKERRQ(ierr);
      }
    }
    else // All other interpolators
    {
      // Set local column numbers
      lCols.setZero();
      for (int j = 0; j < cList[i].size(); j++)
      {
        int proc = 1;
        while (cList[i](j) >= nddist(proc))
          proc++;
        lCols(proc)++;
      }
      partial_sum(lCols.data()+1, lCols.data()+this->nprocs+1, lCols.data()+1);

      // Get preallocation sizes and update J arrays with level-specific node numbers;
      PetscInt oldcol = 0, newcol = 0, proc = 0;
      for (int j = 0; j < J[i].size(); j++)
      {
        if (J[i](j) != oldcol)
        {
          oldcol = J[i](j);
          newcol = invind(oldcol);
          proc = 0;
          while (oldcol >= nddist(proc+1))
            proc++; // process that the column is on
        }
        J[i](j) = newcol;
        if ( (I[i](j) >= lRows(proc)) && (I[i](j) < lRows(proc+1)) )
          onDiag[I[i](j)]++;
        else
          offDiag[I[i](j)]++;
      }

      // Share all preallocation information
      MPI_Iallreduce(MPI_IN_PLACE, onDiag, gRows, MPI_PETSCINT, MPI_SUM, comm, &on_req);
      MPI_Iallreduce(MPI_IN_PLACE, offDiag, gRows, MPI_PETSCINT, MPI_SUM, comm, &off_req);

      // Update I array of next level
      for (int j = 0; j < I[i+1].size(); j++)
        I[i+1](j) = invind(I[i+1](j));

      MPI_Wait(&on_req, MPI_STATUS_IGNORE);
      MPI_Wait(&off_req, MPI_STATUS_IGNORE);

      // Set matrix size and preallocatei
      ierr = MatSetSizes(this->PR[i], numDims*(lRows(myid+1)-lRows(myid)),
                 numDims*(lCols(myid+1)-lCols(myid)), numDims*gRows, numDims*gCols); CHKERRQ(ierr);
      ierr = MatXAIJSetPreallocation(this->PR[i], this->numDims, onDiag+lRows(myid), offDiag+lRows(myid), 0, 0); CHKERRQ(ierr);
    }

    // Assemble the matrix
    for (int j = 0; j < I[i].size(); j++)
    {
      MatrixXPS vals = K[i](j)*MatrixXPS::Identity(numDims, numDims);
      ierr = MatSetValuesBlocked(this->PR[i], 1, I[i].data()+j, 1, J[i].data()+j,
        vals.data(), INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(this->PR[i], MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(this->PR[i], MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    
    // Scale rows
    Vec rowSum, Ones;
    ierr = MatCreateVecs(this->PR[i], NULL, &rowSum); CHKERRQ(ierr);
    ierr = VecDuplicate(rowSum, &Ones); CHKERRQ(ierr);
    ierr = VecSet(Ones, 1.0); CHKERRQ(ierr);
    ierr = MatGetRowSum(this->PR[i], rowSum); CHKERRQ(ierr);
    PetscInt location; PetscScalar minimum;
    ierr = VecMin(rowSum, &location, &minimum); CHKERRQ(ierr);
    if (minimum == 0.0)
      SETERRQ2(comm, PETSC_ERR_FP, "Fine node #%i in interpolation matrix %i is not attached to any coarse nodes", location, i);
    ierr = VecPointwiseDivide(rowSum, Ones, rowSum); CHKERRQ(ierr);
    ierr = MatDiagonalScale(this->PR[i], rowSum, NULL); CHKERRQ(ierr);
    ierr = VecDestroy(&rowSum); CHKERRQ(ierr);
    ierr = VecDestroy(&Ones); CHKERRQ(ierr);
  }
  delete[] onDiag; delete[] offDiag;

  return 0;
}
