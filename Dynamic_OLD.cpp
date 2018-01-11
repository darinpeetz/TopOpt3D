#include <iostream>
#include <fstream>
#include <numeric>
#include "Functions.h"
#include "TopOpt.h"
#include "EigLab.h"
#include <unsupported/Eigen/KroneckerProduct>

using namespace std;
typedef Eigen::Matrix<PetscScalar, -1, -1> MatrixPS;
typedef Eigen::Matrix<PetscScalar, -1, 1>  VectorPS;

namespace Functions
{
    int DiagMassFnc( TopOpt *topOpt, Mat &M, Eigen::VectorXd &dMdy );
    void VectorMassFnc( TopOpt *topOpt, Mat &M, Eigen::VectorXd &dMdy );
    void MassFnc( TopOpt *topOpt, Mat &M, Eigen::VectorXd &dMdy );

    int Dynamic( TopOpt *topOpt, double *lambda, double *grad, PetscInt &nevals )
    {
        short NE = topOpt->element.cols(), DN = topOpt->numDims, DE = NE*DN;

        /// Assemble stress stiffness matrix and get sensitivity information
        Eigen::VectorXd dMdy;
        Mat M;
        DiagMassFnc( topOpt, M, dMdy );
        //VectorMassFnc( topOpt, M, dMdy );

        /// Create eigenvectors
        Vec *phi;
        VecDuplicateVecs(topOpt->U, nevals, &phi);

        /// Remove fixed and spring dof from M (and K if necessary)
        MatZeroRowsColumns(M, topOpt->fixedDof.size(),
                           topOpt->fixedDof.data(), 0, NULL, NULL);
        if (topOpt->nSpringDof > 0)
        {
          MatZeroRowsColumns(M, topOpt->springDof.size(),
                             topOpt->springDof.data(), 0, NULL, NULL);
          MatZeroRowsColumns(topOpt->K, topOpt->springDof.size(),
                             topOpt->springDof.data(), 1, NULL, NULL);
        }

        /// Set up the standard eigenvalue problem
        /*EPS eps; // The eigensolver context
        EPSCreate(topOpt->comm, &eps);
        EPSSetOperators(eps, M, NULL);
        EPSSetType(eps, EPSARPACK);
        EPSSetProblemType(eps, EPS_NHEP);
        EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE);*/

        /// Set up the generalized eigenvalue problem
        EPS eps; // The eigensolver context
        EPSCreate(topOpt->comm, &eps);
        EPSSetType(eps, EPSKRYLOVSCHUR);
        EPSSetProblemType(eps, EPS_GHEP);
        EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL);
        EPSSetOperators(eps, M, topOpt->K);
        EPSSetDimensions(eps, nevals, PETSC_DEFAULT, PETSC_DEFAULT);
        EPSSetTolerances(eps, 1e-4, 300);
        EPSSetFromOptions(eps);

        /// For using spectrum slicing eigenvalue method
        //EPSSetWhichEigenpairs(eps, EPS_ALL);
        /*RG rg;
        EPSGetRG(eps, &rg);
        RGSetType(rg, RGELLIPSE);
        RGEllipseSetParameters(rg, 3e-5, 2.5e-5, 1e-10);
        //EPSCISSSetSizes(eps, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT,
            PETSC_DEFAULT, PETSC_DEFAULT, PETSC_TRUE);*/

        /// Set up the Spectral Transformation (Matrix inversion operation)
        if (topOpt->nSpringDof == 0)
        {
          ST eps_st;
          EPSGetST(eps, &eps_st);
          STSetKSP(eps_st, topOpt->KUF);
        }
        else
        {
          ST eps_st;
          EPSGetST(eps, &eps_st);
          if (topOpt->dynamicShape.size() == 0)
          {
            KSPCreate(topOpt->comm, &topOpt->dynamicKSP);
            KSPSetType(topOpt->dynamicKSP, KSPPREONLY);
            KSPSetFromOptions(topOpt->dynamicKSP);
            PC pc;
            KSPGetPC(topOpt->dynamicKSP, &pc);
            PCSetType(pc, PCCHOLESKY);
            PCSetFromOptions(pc);
          }
          STSetKSP(eps_st, topOpt->dynamicKSP);
        }

        /// Check if this routine has been called before and if we need to initialize eigenvectors
        if (topOpt->dynamicShape.size() == 0)
        {
          topOpt->dynamicShape.setZero(topOpt->node.size(), nevals);
          topOpt->dynamicIt = 300;
          for (int i = 0; i < nevals; i++)
          {
            VecPlaceArray(phi[i], topOpt->dynamicShape.data() +
                          i*topOpt->dynamicShape.rows());
          }
        }
        else
        {
          for (int i = 0; i < nevals; i++)
          {
            VecPlaceArray(phi[i], topOpt->dynamicShape.data() +
                          i*topOpt->dynamicShape.rows());
          }
          EPSSetInitialSpace(eps, nevals, phi);
        }

        /// Perform the solve and look at the results
        EPSSolve(eps);

        /// Look at results if requested
        PetscBool verbose = PETSC_FALSE;
        PetscInt requested = nevals;
        EPSGetConverged(eps, &nevals);
        nevals = std::min(nevals, requested);
        if (requested != nevals)
        {
          topOpt->dynamicIt *= pow(1.1, requested-nevals);
          PetscPrintf(topOpt->comm, "Only %li of %li eigenvalues converged\n", nevals, requested);
          PetscPrintf(topOpt->comm, "Max iterations increased to: %li\n", topOpt->dynamicIt);
        }

        /// Display more information if requested
        PetscOptionsHasName(NULL,NULL,"-eig_verbose",&verbose);
        if (verbose == PETSC_TRUE)
        {
            /// Read out what options were set
            EPSProblemType Ptype;
            EPSGetProblemType(eps, &Ptype);
            PetscPrintf(topOpt->comm, "Problem type: %i\n", Ptype);

            EPSType type;
            EPSGetType(eps, &type);
            PetscPrintf(topOpt->comm, "Solution method: %s\n", type);

            /// Display convergence information
            PetscInt iters;
            EPSGetIterationNumber(eps, &iters);
            PetscPrintf(topOpt->comm, "Eigenvalue solver converged after %li iterations\n",iters);
        }

        /// Pull out the converged eigenvalues
        for (short i = 0; i < nevals; i++)
        {
          EPSGetEigenpair(eps, i, lambda+i, 0, phi[i], 0);

          // M-normalize eigenvectors - only if using standard eigenvalue problem
          /*Vec temp;
          PetscScalar norm;
          VecDuplicate(phi_full[i], &temp);
          MatMult(topOpt->K, phi_full[i], temp);
          VecDot(phi_full[i], temp, &norm);
          VecScale(phi_full[i], 1.0/sqrt(norm));

          MatMult(topOpt->K, phi_full[i], temp);
          VecDot(phi_full[i], temp, &norm);*/

          // Update ghost positions
          VecGhostUpdateBegin(phi[i], INSERT_VALUES, SCATTER_FORWARD);
          VecGhostUpdateEnd(phi[i], INSERT_VALUES, SCATTER_FORWARD);
          VecResetArray( phi[i] );
          if (topOpt->myid == 0)
            std::cout << lambda[i] << "\t";
        }
        if (topOpt->myid == 0)
          std::cout << "\n";

        /// Destroy EPS objects
        VecDestroyVecs(topOpt->dynamicShape.cols(), &phi);
        ST eps_st;
        EPSGetST(eps, &eps_st);
        KSP empty;
        KSPCreate(topOpt->comm, &empty);
        STSetKSP(eps_st, empty);
        EPSDestroy(&eps);

        /// Dot product of eigenvectors expanded to triplet form
        /// to match unassembled stiffness matrices
        Eigen::MatrixXd phim( (DE*DE)*topOpt->nLocElem, nevals );
        for (long el = 0; el < topOpt->nLocElem; el++)
        {
          ArrayXPI eDof(DE);
          for (int i = 0; i < NE; i++)
          {
            for (int j = 0; j < DN; j++)
              eDof(i*DN + j) = DN*topOpt->element(el, i) + j;
          }

          for (int i = 0; i < DE; i++){
            for (int j = 0; j < DE; j++){
                phim.row( (DE*DE)*el + DE*i + j) =
                  topOpt->dynamicShape.block(eDof(j),0,1,nevals).cwiseProduct(
                                  topOpt->dynamicShape.block(eDof(i),0,1,nevals));
            }
          }
        }

        /// Construct sensitivity of material stiffness matrix
        const PetscScalar *p_dEdy;
        VecGetArrayRead(topOpt->dEdy, &p_dEdy);
        Eigen::Map< const Eigen::VectorXd > dEdy(p_dEdy, topOpt->nLocElem);
        Eigen::MatrixXd dKdy;
        if (topOpt->regular)
        {
          Eigen::Map< Eigen::VectorXd > ke(topOpt->ke[0].data(), DE*DE);
          dKdy = Eigen::kroneckerProduct(dEdy, ke);
        }
        else
        {
          /// TODO: COMBINE THIS AND PREVIOUS LOOP FOR EFFICIENCY
          PetscInt ind = 0;
          for (unsigned int el = 0; el < topOpt->ke.size(); el++)
            ind += topOpt->ke[el].size();
          dKdy.resize(ind, 1);
          ind = 0;
          Eigen::Map< Eigen::VectorXd > ke(topOpt->ke[0].data(), DE*DE);
          for (unsigned int el = 0; el < topOpt->ke.size(); el++)
          {
            new (&ke)Eigen::Map< Eigen::VectorXd >(topOpt->ke[el].data(),topOpt->ke[el].size());
            dKdy.block(ind, 0, ke.size(), 1) = dEdy(el)*ke;
          }
        }
        VecRestoreArrayRead(topOpt->dEdy, &p_dEdy);

        /// Construct sensitivity
        Eigen::MatrixXd df((DE*DE)*topOpt->nLocElem,nevals);
        for (short i = 0; i < nevals; i++)
            df.col(i) = phim.col(i).cwiseProduct(dMdy-lambda[i]*dKdy);

        for (long el = 0; el < topOpt->nLocElem; el++)
            df.row(el) = df.block(el*(DE*DE), 0, (DE*DE), nevals).colwise().sum();
        df.conservativeResize(topOpt->nLocElem, nevals);

        /// dCdrhof*drhofdrho
        Vec PETSc_grad, dlamdy;
        VecCreateMPI( topOpt->comm, topOpt->nLocElem, topOpt->nElem, &PETSc_grad );
        VecDuplicate( PETSc_grad, &dlamdy );
        for (short i = 0; i < nevals; i++)
        {
            VecPlaceArray( dlamdy, df.data()+i*df.rows() );
            VecPlaceArray( PETSc_grad, grad+i*topOpt->nLocElem );
            MatMultTranspose( topOpt->P, dlamdy, PETSc_grad );

            VecResetArray(dlamdy);
            VecResetArray(PETSc_grad);
        }
        VecDestroy( &PETSc_grad ); VecDestroy( &dlamdy );

        return 0;
    }

    /********************************************************************/
    /*                  Creates the diagonal mass matrix               **/
    /********************************************************************/
    int DiagMassFnc( TopOpt *topOpt, Mat &M, Eigen::VectorXd &dMdy )
  {
    PetscErrorCode ierr = 0;

    // Initialize M
    ierr = MatCreate(topOpt->comm, &M); CHKERRQ(ierr);
    ierr = MatSetSizes(M, topOpt->numDims*topOpt->nLocNode, topOpt->numDims*topOpt->nLocNode,
                topOpt->numDims*topOpt->nNode, topOpt->numDims*topOpt->nNode); CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(M,"M_"); CHKERRQ(ierr);
    ierr = MatSetFromOptions(M); CHKERRQ(ierr);
    ArrayXPI onDiag = ArrayXPI::Ones(topOpt->nLocNode);
    ArrayXPI offDiag = ArrayXPI::Zero(topOpt->nLocNode);
    ierr = MatXAIJSetPreallocation(M, topOpt->numDims, onDiag.data(), offDiag.data(), 0, 0); CHKERRQ(ierr);

    // Mesh characteristics
    const short NE = topOpt->element.cols(), DN = topOpt->numDims, DE = NE*DN;
    // Track construction of Ks, dKs
    long dMmarker = 0;
    dMdy.resize( topOpt->nLocElem*(long)pow(DE,2) );

    // Get pointers to Petsc vectors
    const PetscScalar *p_V, *p_dVdy;
    ierr = VecGetArrayRead(topOpt->V, &p_V); CHKERRQ(ierr);
    ierr = VecGetArrayRead(topOpt->dVdy, &p_dVdy); CHKERRQ(ierr);

    MatrixPS mMat = 1.0/pow(2,topOpt->numDims)/topOpt->numDims*
            topOpt->elemSize(0)*topOpt->density*Eigen::MatrixXd::Identity(DE, DE);
    Eigen::Map< VectorPS > mVec(mMat.data(), mMat.size());
    MatrixPS nodeMat(topOpt->numDims, topOpt->numDims);
    /// Loop over elements
    for (long el = 0; el < topOpt->element.rows(); el++)
    {
      if (!topOpt->regular)
      {
        mMat.setIdentity();
        mMat *= 1.0/pow(2,topOpt->numDims)/topOpt->numDims *
            topOpt->density * topOpt->elemSize(0);
      }

      /// Fill in the sensitivity dMdy
      if (el < topOpt->nLocElem)
      {
        dMdy.segment(dMmarker, mVec.size()) = p_dVdy[el] * mVec;
        dMmarker += mVec.size();
      }

      /// Loop over indices to fill in M
      for (int n = 0; n < NE; n++) // Looping over rows
      {
        PetscInt node = topOpt->element(el,n);
        if (node < topOpt->nLocNode) // If node is local to this process
        {
          nodeMat = p_V[el]*mMat.block(n*topOpt->numDims, n*topOpt->numDims,
              topOpt->numDims, topOpt->numDims);
          PetscInt row = topOpt->gNode(node);
          ierr = MatSetValuesBlocked(M, 1, &row, 1, &row, nodeMat.data(), ADD_VALUES); CHKERRQ(ierr);
        }
      }
    }

    ierr = MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(topOpt->V, &p_V); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(topOpt->dVdy, &p_dVdy); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatDiagonalSet(M, topOpt->MLump, ADD_VALUES); CHKERRQ(ierr);

    return 0;
  }

    /********************************************************************/
    /*      Gets M as a vector and returns M = M\K (freeDof only)      **/
    /********************************************************************/
    void VectorMassFnc( TopOpt *topOpt, Mat &M, Eigen::VectorXd &dMdy )
    {
      // Initialize M
      // Note Mvec is diagonal of M, mVec is m (element mass matrix) stored as a
      // vector.  This is probably confusing and should be revised
      Vec MVec;
      VecDuplicate(topOpt->U, &MVec);

      // Mesh characteristics
      const short NE = topOpt->element.cols(), DN = topOpt->numDims, DE = NE*DN;
      // Track construction of Ks, dKs
      long dMmarker = 0;
      dMdy.resize( topOpt->nLocElem*(long)pow(DE,2) );

      // Get pointers to Petsc vectors
      const PetscScalar *p_V, *p_dVdy;
      VecGetArrayRead(topOpt->V, &p_V);
      VecGetArrayRead(topOpt->dVdy, &p_dVdy);
      Eigen::MatrixXd mMat = 1.0/pow(2,topOpt->numDims)*
              topOpt->elemSize(0)*topOpt->density*Eigen::MatrixXd::Identity(DE, DE);
      Eigen::Map< Eigen::VectorXd > mVec(mMat.data(), mMat.size());
      /// Loop over elements
      for (long el = 0; el < topOpt->element.rows(); el++)
      {
        if (!topOpt->regular)
        {
          mMat.setIdentity();
          mMat *= 1.0/pow(2,topOpt->numDims)*topOpt->density*topOpt->elemSize(0);
        }

        /// Fill in the sensitivity dMdy
        if (el < topOpt->nLocElem)
        {
          dMdy.segment(dMmarker, mVec.size()) = p_dVdy[el] * mVec;
          dMmarker += mVec.size();
        }

        /// Loop over indices to fill in M
        for (int n = 0; n < NE; n++) // Looping over rows
        {
          PetscInt node = topOpt->element(el,n);
          if (node < topOpt->nLocNode) // If node is local to this process
          {
            PetscScalar v = p_V[el] * mMat(n,n);
            for (int d = 0; d < DN; d++)
              VecSetValue(MVec, DN*topOpt->gNode(node)+d, v, ADD_VALUES);
          }
        }
      }
      VecAssemblyBegin(MVec);
      VecRestoreArrayRead(topOpt->V, &p_V);
      VecRestoreArrayRead(topOpt->dVdy, &p_dVdy);
      VecAssemblyEnd(MVec);
      VecAXPY(MVec, 1, topOpt->MLump);

      // Apply Dirichlet B.C.'s
      IS sub;
      ISCreateGeneral(topOpt->comm, topOpt->freeDof.size(),
                      topOpt->freeDof.data(), PETSC_USE_POINTER, &sub);
      MatGetSubMatrix(topOpt->K, sub, sub, MAT_INITIAL_MATRIX, &M);

      // Apply inverse of M matrix to K matrix
      Vec ones;
      Vec Msub;
      VecGetSubVector(MVec, sub, &Msub);
      VecDuplicate(Msub, &ones);
      VecSet(ones, 1.0);
      VecPointwiseDivide(Msub, ones, Msub);
      MatDiagonalScale(M, Msub, NULL);
      VecRestoreSubVector(MVec, sub, &Msub);

      return;
    }

    /********************************************************************/
    /*     Not ready yet, intended to make a consistent mass matrix    **/
    /********************************************************************/
    void MassFnc( TopOpt *topOpt, Mat &M, Eigen::VectorXd &dMdy )
    {
      // Initialize M
      MatDuplicate( topOpt->K, MAT_SHARE_NONZERO_PATTERN, &M );

      // Mesh characteristics
      const short NE = topOpt->element.cols(), DN = topOpt->numDims, DE = NE*DN;
      // Track construction of Ks, dKs
      long dMmarker = 0;
      dMdy.resize( topOpt->nLocElem*(long)pow(DE,2) );

      // Get pointers to Petsc vectors
      const PetscScalar *p_V, *p_dVdy;
      VecGetArrayRead(topOpt->V, &p_V);
      VecGetArrayRead(topOpt->dVdy, &p_dVdy);

      Eigen::MatrixXd mMat = 1.0/pow(2,topOpt->numDims)*
                topOpt->elemSize(0)*topOpt->density*Eigen::MatrixXd::Identity(DE, DE);
      Eigen::Map< Eigen::VectorXd > mVec(mMat.data(), mMat.size());
      /// Loop over elements
      for (long el = 0; el < topOpt->element.rows(); el++)
      {
        if (!topOpt->regular)
        {
          mMat.setIdentity();
          mMat *= 1.0/pow(2,topOpt->numDims)*topOpt->density*topOpt->elemSize(0);
        }

        /// Fill in the sensitivity dMdy
        if (el < topOpt->nLocElem)
        {
          dMdy.segment(dMmarker, mVec.size()) = p_dVdy[el] * mVec;
          dMmarker += mVec.size();
        }

        /// Loop over indices to fill in M
        for (int n = 0; n < NE; n++) // Looping over rows
        {
          PetscInt node = topOpt->element(el,n);
          if (node < topOpt->nLocNode) // If node is local to this process
          {
            PetscScalar v = p_V[el] * mMat(n,n);
            for (int d = 0; d < DN; d++)
              MatSetValue(M, DN*topOpt->gNode(node)+d, DN*topOpt->gNode(node)+d, v, ADD_VALUES);
          }
        }
      }

      MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
      VecRestoreArrayRead(topOpt->V, &p_V);
      VecRestoreArrayRead(topOpt->dVdy, &p_dVdy);
      MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
      MatDiagonalSet(M, topOpt->MLump, ADD_VALUES);

      return;
    }
}
