#include "mpi.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include "TopOpt.h"

using namespace std;

typedef Eigen::Array<PetscInt, -1, -1> ArrayXXPI;
typedef Eigen::Array<PetscInt, -1, 1> ArrayXPI;

/// This method produces a density filter on regular, rectangular grids
void TopOpt::RecFilter ( PetscInt *first, PetscInt *last, double *dx, double R,
                         ArrayXPI Nel, FilterArrays &filterArrays )
{
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
  double *dist = new double[(2*N[0]+1)*(2*N[1]+1)*(2*N[2]+1)];
  // Indicator if the elements are within radius R
  bool *nbrhd = new bool[(2*N[0]+1)*(2*N[1]+1)*(2*N[2]+1)];
  // Element number template for adding elements to array
  int *elemTemplate = new int[(2*N[0]+1)*(2*N[1]+1)*(2*N[2]+1)];
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
  filterArrays.Reset(nLocElem*nNbrhd);
  int filterInd = 0;
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
                filterArrays.elements.row(filterInd) << el, elemTemplate[ind]+el;
                filterArrays.distances(filterInd) = 1-dist[ind]/R;
                filterInd++;
              }
            }
          }
        }

      }
    }
  }
  filterArrays.Truncate(filterInd);
  delete[] dist;
  delete[] nbrhd;
  delete[] elemTemplate;

  return;
}
