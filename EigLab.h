#ifndef EIGLAB_H_INCLUDED
#define EIGLAB_H_INCLUDED

#include <Eigen/Eigen>
#include <vector>
#include <iostream>

template <class T> class xind
{
  public:
  T x; long ind;

  xind(T x, long ind) : x(x), ind(ind) {}
  ~xind() {};
  bool operator<(xind const &other) const {
    return x < other.x;}
};

namespace EigLab
{
/*************************************************************************************************/
  template <typename MatrixTypeA>
  Eigen::ArrayXi gensort ( MatrixTypeA &x )
  {
    typedef typename MatrixTypeA::Scalar Scalar;

    MatrixTypeA result(x.rows(), x.cols());
    Eigen::ArrayXi ind(x.rows(),x.cols());
    std::vector< xind<Scalar> > sorter;
    for (long i = 0; i < x.rows(); i++)
        sorter.push_back( xind<Scalar>(x(i, 0), i) );
    std::stable_sort(sorter.begin(), sorter.end());
    for (unsigned long j = 0; j < sorter.size(); j++)
    {
      result(j) = x(sorter[j].ind);
      ind(j) = sorter[j].ind;
    }
    x = result;

    return ind;
  }
/*************************************************************************************************/
  template <typename MatrixTypeA>
  Eigen::ArrayXi sort ( MatrixTypeA &x, short dim )
  {
    typedef typename MatrixTypeA::Scalar Scalar;

    Eigen::ArrayXi ind;
    long m, n;
    m = x.rows();
    n = x.cols();
    if (dim == 1) // sort rows
    {
      ind = Eigen::ArrayXi::LinSpaced(m, 0, m-1);
      Eigen::ArrayXi newind(m);
      MatrixTypeA result(m, n);
      for (long i = n-1; i >= 0; i-- )
      {
        std::vector< xind<Scalar> > sorter;
        for (long j = 0; j < m; j++)
            sorter.push_back( xind<Scalar>(x(j,i), j) );
        std::stable_sort(sorter.begin(), sorter.end());
        for (unsigned long j = 0; j < sorter.size(); j++)
        {
          result.row(j) = x.row(sorter[j].ind);
          newind(j) = ind(sorter[j].ind);
        }
        x = result;
        ind = newind;
      }
    }
    else if (dim == 2) // sort cols
    {
      ind = Eigen::ArrayXi::LinSpaced(n, 0, n-1);
      Eigen::ArrayXi newind(n);
      MatrixTypeA result(m, n);
      for (long i = m-1; i >= 0; i-- )
      {
        std::vector< xind<Scalar> > sorter;
        for (long j = 0; j < n; j++)
            sorter.push_back( xind<Scalar>(x(i,j), j) );
        std::stable_sort(sorter.begin(), sorter.end());
        for (unsigned long j = 0; j < sorter.size(); j++)
        {
          result.col(j) = x.col(sorter[j].ind);
          newind(j) = ind(sorter[j].ind);
        }
        x = result;
        ind = newind;
      }
    }

    return ind;
  }
/*************************************************************************************************/
  template <typename MatrixTypeA, typename MatrixTypeB>
  void IndRemove ( MatrixTypeA &matrix, MatrixTypeB &indices, const short dim )
  {
    //Keeps indicated indices
    std::sort(indices.data(), indices.data()+indices.size());
    long int nrows = matrix.rows(), ncols = matrix.cols();
    if (dim == 1)
    {
      nrows = indices.size();
      for (long int i = 0; i < indices.size(); i++)
          matrix.row(i) = matrix.row(indices(i));
    }
    if (dim == 2)
    {
      ncols = indices.size();
      for (long int i = 0; i < indices.size(); i++)
          matrix.col(i) = matrix.col(indices(i));
    }
    matrix.conservativeResize(nrows,ncols);

    return;
  }
/*************************************************************************************************/
  template <typename MatrixTypeA, typename MatrixTypeB>
  void BoolRemove ( MatrixTypeA &matrix, MatrixTypeB &indices, const short dim )
  {
    //Keeps nonzero values
    long int index = 0;
    long int nrows = matrix.rows(), ncols = matrix.cols();

    if (dim == 1)
    {
      for (long int i = 0; i < matrix.rows(); i++)
      {
        if (indices(i) != 0)
          matrix.row(index++) = matrix.row(i);
      }
      nrows = index;
    }
    if (dim == 2)
    {
      for (long int i = 0; i < matrix.cols(); i++)
      {
        if (indices(i) != 0)
          matrix.col(index++) = matrix.col(i);
      }
      ncols = index;
    }
    matrix.conservativeResize(nrows,ncols);

    return;
  }
/*************************************************************************************************/
  template <typename MatrixTypeA, typename MatrixTypeB>
  void RemoveSlices ( MatrixTypeA &matrix, MatrixTypeB &indices, const short dim )
  {
    if (matrix.size() == 0)
      return;
    // Removes rows or columns from a matrix - operates with 0 base for row numbers, dim = 1 for rows, 2 for cols
    if ( dim == 1 )  //Removing Rows
    {
      if (indices.rows() == matrix.rows())          //boolean remove, 1 keep, 0 remove
        BoolRemove ( matrix, indices, dim );
      else                                         //Keep indicated indices
        IndRemove ( matrix, indices, dim );
    }
    else if ( dim == 2 ) //Removing Cols
    {
      if (indices.cols() == matrix.cols())         //boolean remove, 1 keep, 0 remove
        BoolRemove ( matrix, indices, dim );
      else                                         //Keep indicated indices
        IndRemove ( matrix, indices, dim );
    }
    else
      std::cout << "Bad dimension given to RemoveRows, dimension must be 1 for rows or 2 for columns\n";
    return;
  }
/*************************************************************************************************/
  template <typename MatrixTypeA>
  void Unique ( MatrixTypeA &matrix, short dim )
  {
    if (matrix.size() == 0)
      return;

    sort(matrix, dim);
    if (dim == 1)
    {
      MatrixTypeA matA = matrix.block(0, 0, matrix.rows()-1, matrix.cols());
      MatrixTypeA matB = matrix.block(1, 0, matrix.rows()-1, matrix.cols());
      Eigen::Matrix<bool, -1, 1> check = (matA.cwiseNotEqual(matB)).rowwise().any();
      check.reverseInPlace();
      check.conservativeResize(check.rows()+1,check.cols());
      check.tail(1) << 1;
      check.reverseInPlace();
      RemoveSlices(matrix, check, dim);
    }
    else if (dim == 2)
    {
      MatrixTypeA matA = matrix.block(0, 0, matrix.rows(), matrix.cols()-1);
      MatrixTypeA matB = matrix.block(0, 1, matrix.rows(), matrix.cols()-1);
      Eigen::Matrix<bool, -1, 1> check = (matA.cwiseNotEqual(matB)).colwise().any();
      check.reverseInPlace();
      check.conservativeResize(check.rows(),check.cols()+1);
      check.tail(1) << 1;
      check.reverseInPlace();
      RemoveSlices(matrix, check, dim);
    }
    else
      std::cout << "Bad dimension given to Unique, dimension must be 1 for rows or 2 for columns\n";
    return;
  }
/*************************************************************************************************/
  template <typename MatrixTypeA, typename T>
  Eigen::ArrayXXi find ( MatrixTypeA &matrix, const T value)
  {
    typedef typename MatrixTypeA::Scalar Scalar;
    Scalar *start = matrix.data();
    Scalar *finish = matrix.data()+matrix.size();
    Scalar *point;
    std::vector<long> colind;
    while ( start <= finish )
    {
      point = std::find(start, finish, value);
      colind.push_back(point-matrix.data());
      start = point+1;
    }
    colind.pop_back();
    Eigen::ArrayXXi indices(colind.size(), 2);
    long n = matrix.rows();
    for (unsigned int i = 0; i < colind.size(); i++)
    {
      indices(i,0) = colind[i] % n;
      indices(i,1) = (colind[i]-indices(i,0))/n;
    }
    return indices;
  }
/*************************************************************************************************/
}

#endif // EIGLAB_H_INCLUDED
