#include <iostream>
#include <cmath>
#include <fstream>
#include "TopOpt.h"

using namespace std;

typedef Eigen::Array<PetscInt, -1, -1> ArrayXXPI;
typedef Eigen::Array<PetscInt, -1, 1> ArrayXPI;

/********************************************************************
 *  This method produces a density filter on regular, rectangular grids
 * 
 * @param first: first element in each dimension on this process
 * @param last: last element in each dimension on this process
 * @param dx: element spacing in each dimension
 * @param R: Filter radius
 * @param nel: Number of elements in each dimension
 * @param Filter: The Mat object to be used as the filter matrix
 * 
 * @return ierr: PetscErrorCode
 *******************************************************************/
PetscErrorCode TopOpt::RecFilter ( PetscInt *first, PetscInt *last,
                                   double *dx, double R, ArrayXPI Nel,
                                   Mat &Filter, PetscScalar nonzeros )
{
  PetscErrorCode ierr = 0;

  // Number of elements in either direction within radius (not including)
  // the element at the center
  short N[3] = {0, 0, 0};
  short nNbrhd = 1;
  for (short i = 0; i < numDims; i++)
  {
    N[i] = R/dx[i];
    nNbrhd *= 2*N[i]+1;
  }

  // Distances of all neighborhood elements
  Eigen::ArrayXd dist((2*N[0]+1)*(2*N[1]+1)*(2*N[2]+1));
  // Indicator if the elements are within radius R
  Eigen::Array<bool, -1, 1> nbrhd((2*N[0]+1)*(2*N[1]+1)*(2*N[2]+1));
  // Element number template for adding elements to array
  ArrayXPI elemTemplate((2*N[0]+1)*(2*N[1]+1)*(2*N[2]+1));
  for (int k = -N[2]; k < N[2]+1; k++)
  {
    for (int j = -N[1]; j < N[1]+1; j++)
    {
      for (int i = -N[0]; i < N[0]+1; i++)
      {
        int ind = i+N[0] + (j+N[1])*(2*N[0]+1) + (k+N[2])*(2*N[0]+1)*(2*N[1]+1);
        dist[ind] = sqrt( pow(i*dx[0],2) + pow(j*dx[1],2) + pow(k*dx[2],2) );
        nbrhd[ind] = dist[ind] < R;
        elemTemplate[ind] = i + j*Nel(0) + k*Nel(0)*Nel(1);
      }
    }
  }

  // Arrays of connected elements and their distances
  int filterInd = 0;
  Eigen::Array<PetscInt, -1, 1> I(nLocElem*nNbrhd);
  Eigen::Array<PetscInt, -1, 1> J(nLocElem*nNbrhd);
  Eigen::Array<PetscScalar, -1, 1> K(nLocElem*nNbrhd);
  // First three loops are over local elements
  for (int elk = first[2]; elk < last[2]; elk++)
  {
    for (int elj = first[1]; elj < last[1]; elj++)
    {
      for (int eli = first[0]; eli < last[0]; eli++)
      {
        int el = eli + elj*Nel(0) + elk*Nel(0)*Nel(1);
        // Next three loops are over neighborhood elements
        for (int k = -N[2]; k < N[2]+1; k++)
        {
          for (int j = -N[1]; j < N[1]+1; j++)
          {
            for (int i = -N[0]; i < N[0]+1; i++)
            {
              // Connected element number in neighborhood
              int ind = i+N[0] + (j+N[1])*(2*N[0]+1) + (k+N[2])*(2*N[0]+1)*(2*N[1]+1);
              // If element is within radius and in the same row/column
              bool valid = (nbrhd[ind]) && (i+eli>=0) && (i+eli<Nel(0)) &&
                  (j+elj>=0) && (j+elj<Nel(1)) && (k+elk>=0) && (k+elk<Nel(2));
              if (valid)
              {
                // Add that element to list
                I(filterInd) = el;
                J(filterInd) = elemTemplate[ind]+el;
                if (nonzeros)
                  K(filterInd) = nonzeros;
                else
                  K(filterInd) = 1-dist[ind]/R;
                filterInd++;
              }
            }
          }
        }

      }
    }
  }
  I.conservativeResize(filterInd);
  J.conservativeResize(filterInd);
  K.conservativeResize(filterInd);

  /// Assemble the filter matrix
  ierr = MatCreate(comm, &Filter); CHKERRQ(ierr);
  ierr = MatSetSizes(Filter, this->nLocElem, this->nLocElem,
                     this->nElem, this->nElem); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(Filter, "Filter_"); CHKERRQ(ierr);
  ierr = MatSetFromOptions(Filter); CHKERRQ(ierr);

  // Set preallocation
  ArrayXPI onDiag = ArrayXPI::Zero(this->nLocElem);
  ArrayXPI offDiag = ArrayXPI::Zero(this->nLocElem);
  for (int el = 0; el < I.size(); el++)
  {
    if (J(el) >= this->elmdist(this->myid) &&
        J(el) < this->elmdist(this->myid+1) )
    {
      onDiag(I(el)-this->elmdist(this->myid))++;
    }
    else
    {
      offDiag(I(el)-this->elmdist(this->myid))++;
    }
  }

  // Set the preallocation
  ierr = MatXAIJSetPreallocation(Filter, 1, onDiag.data(),
                                 offDiag.data(), 0, 0); CHKERRQ(ierr);

  // Insert values into matrix
  for (int el = 0; el < I.size(); el++)
  {
    ierr = MatSetValue(Filter, I(el), J(el), K(el), ADD_VALUES); CHKERRQ(ierr);
  }

  // Begin assembly (finish just before returning from function)
  ierr = MatAssemblyBegin(Filter, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  // Finish assembly of the matrix before continuing
  ierr = MatAssemblyEnd(Filter, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  // Scale Rows
  Vec rowSum, Ones;
  ierr = VecCreateMPI(comm, nLocElem, nElem, &rowSum); CHKERRQ(ierr);
  ierr = VecDuplicate(rowSum, &Ones); CHKERRQ(ierr);
  ierr = VecSet(Ones, 1.0); CHKERRQ(ierr);
  ierr = MatGetRowSum(P, rowSum); CHKERRQ(ierr);
  ierr = VecPointwiseDivide(rowSum, Ones, rowSum); CHKERRQ(ierr);
  ierr = MatDiagonalScale(P, rowSum, NULL); CHKERRQ(ierr);
  ierr = VecDestroy(&rowSum); CHKERRQ(ierr);
  ierr = VecDestroy(&Ones); CHKERRQ(ierr);

  return ierr;
}
