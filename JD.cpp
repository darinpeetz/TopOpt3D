#include <iostream>
#include <fstream>
#include <numeric>
#include "Functions.h"
#include "TopOpt.h"
#include "EigLab.h"
#include <unsupported/Eigen/KroneckerProduct>

using namespace std;

namespace Functions
{
  int DiagMassFnc( TopOpt *topOpt, Mat &M, Eigen::VectorXd &dMdy );
  int VectorMassFnc( TopOpt *topOpt, Mat &M, Eigen::VectorXd &dMdy );
  int MassFnc( TopOpt *topOpt, Mat &M, Eigen::VectorXd &dMdy );

  int Dynamic( TopOpt *topOpt, double *lambda, double *grad, PetscInt &nevals )
  {
    PetscErrorCode ierr;
    short NE = topOpt->element.cols(), DN = topOpt->numDims, DE = NE*DN;

    /// Assemble stress stiffness matrix and get sensitivity information
    Eigen::VectorXd dMdy;
    Mat M;
    ierr = DiagMassFnc( topOpt, M, dMdy ); CHKERRQ(ierr);
    //VectorMassFnc( topOpt, M, dMdy );

    /// Create eigenvectors
    Vec *phi;
    ierr = VecDuplicateVecs(topOpt->U, nevals, &phi);

    for (int i = topOpt->nddist(topOpt->myid)*topOpt->numDims; i < topOpt->nddist(topOpt->myid+1)*topOpt->numDims; i ++)
      MatSetValue(M, i, i, i+1, INSERT_VALUES);
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    /// Remove fixed and spring dof from M (and K if necessary)
    ierr = MatZeroRowsColumns(M, topOpt->fixedDof.size(),
                       topOpt->fixedDof.data(), 1.0, NULL, NULL); CHKERRQ(ierr);
    if (topOpt->nSpringDof > 0)
    {
      ierr = MatZeroRowsColumns(M, topOpt->springDof.size(),
                         topOpt->springDof.data(), 0.0, NULL, NULL); CHKERRQ(ierr);
      ierr = MatZeroRowsColumns(topOpt->K, topOpt->springDof.size(),
                         topOpt->springDof.data(), 10000.0, NULL, NULL); CHKERRQ(ierr);
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
    ierr = EPSCreate(topOpt->comm, &eps); CHKERRQ(ierr);
    ierr = EPSSetType(eps, EPSKRYLOVSCHUR); CHKERRQ(ierr);
    ierr = EPSSetProblemType(eps, EPS_GHEP); CHKERRQ(ierr);
    ierr = EPSSetOperators(eps, M, topOpt->K); CHKERRQ(ierr);
    ierr = EPSSetDimensions(eps, nevals, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = EPSSetTolerances(eps, 1e-8, PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);

    /// For using spectrum slicing eigenvalue method
    //EPSSetWhichEigenpairs(eps, EPS_ALL);
    /*RG rg;
    EPSGetRG(eps, &rg);
    RGSetType(rg, RGELLIPSE);
    RGEllipseSetParameters(rg, 3e-5, 2.5e-5, 1e-10);
    //EPSCISSSetSizes(eps, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT,
        PETSC_DEFAULT, PETSC_DEFAULT, PETSC_TRUE);*/

    /// Check if this routine has been called before and if we need to
    /// initialize eigenvectors and deflation space
    if (topOpt->dynamicShape.size() == 0)
    {
      topOpt->dynamicShape.setZero(topOpt->node.size(), nevals);
      for (int i = 0; i < nevals; i++)
      {
        ierr = VecPlaceArray(phi[i], topOpt->dynamicShape.data() +
                      i*topOpt->dynamicShape.rows()); CHKERRQ(ierr);
      }

      // Create deflation space
      /*int NRBM = (topOpt->numDims*topOpt->numDims + topOpt->numDims)/2, i;
      PetscScalar *p_RBM;
      ierr = VecDuplicateVecs(topOpt->U, NRBM, &topOpt->dynamicDeflate); CHKERRQ(ierr);
      // Translation RBM
      for (i = 0; i < topOpt->numDims; i++)
      {
        ierr = VecSet(topOpt->dynamicDeflate[i], 0.0); CHKERRQ(ierr);
        ierr = VecGetArray(topOpt->dynamicDeflate[i], &p_RBM); CHKERRQ(ierr);
        for (int j = 0; j < topOpt->nLocNode; j++)
          p_RBM[topOpt->numDims*j+i] = 1.0;
        ierr = VecRestoreArray(topOpt->dynamicDeflate[i], &p_RBM); CHKERRQ(ierr);
      }
      // Rotation RBM
      int ind = i;
      for ( ; i < NRBM; i++)
      {
        for (int j = 0; j < i-topOpt->numDims; j++)
        {
          ierr = VecSet(topOpt->dynamicDeflate[ind], 0.0); CHKERRQ(ierr);
          ierr = VecGetArray(topOpt->dynamicDeflate[ind], &p_RBM); CHKERRQ(ierr);
          for (int k = 0; k < topOpt->nLocNode; k++)
          {
            p_RBM[topOpt->numDims*k + j] = topOpt->node(k,i-topOpt->numDims);
            p_RBM[topOpt->numDims*k + i-topOpt->numDims] = -topOpt->node(k,j);
          }
          ierr = VecRestoreArray(topOpt->dynamicDeflate[ind], &p_RBM); CHKERRQ(ierr);
          ind++;
        }
      }*/

    }
    else
    {
      for (int i = 0; i < nevals; i++)
      {
        ierr = VecPlaceArray(phi[i], topOpt->dynamicShape.data() +
                      i*topOpt->dynamicShape.rows()); CHKERRQ(ierr);
      }
      ierr = EPSSetInitialSpace(eps, nevals, phi); CHKERRQ(ierr);
    }

    /// Set up the Spectral Transformation (Matrix inversion operation)
    /*if (topOpt->nSpringDof == 0)
    {*/
      ST eps_st;
      ierr = EPSGetST(eps, &eps_st); CHKERRQ(ierr);
      ierr = STSetKSP(eps_st, topOpt->KUF); CHKERRQ(ierr);
      ierr = KSPSetTolerances(topOpt->KUF, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, 1e6); CHKERRQ(ierr);
    /*}
    else
    {
      KSPSetOperators(this->KUF, this->K, this->K);
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


        KSPSetType(topOpt->dynamicKSP, KSPCG);
        KSPSetTolerances(topOpt->dynamicKSP, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, 1e6);
        PC pc;
        KSPGetPC(topOpt->dynamicKSP, &pc);
        PCSetType(pc, PCGAMG);
        PCSetFromOptions(pc);
      }
      STSetKSP(eps_st, topOpt->dynamicKSP);
    }*/

    /// Perform the solve and look at the results
    /*int NRBM = (topOpt->numDims*topOpt->numDims + topOpt->numDims)/2;
    cout << NRBM << "\n";
    EPSSetDeflationSpace(eps, NRBM, topOpt->dynamicDeflate);*/
    ierr = EPSSetFromOptions(eps);// CHKERRQ(ierr);
    cout << "Calling EPS\n";
    ierr = EPSSolve(eps);// CHKERRQ(ierr);

    /// Look at results if requested
    PetscBool verbose = PETSC_FALSE;
    PetscInt requested = nevals;
    ierr = EPSGetConverged(eps, &nevals); CHKERRQ(ierr);
    nevals = std::min(nevals, requested);
    if (requested != nevals)
    {
      ierr = PetscPrintf(topOpt->comm, "Only %li of %li eigenvalues converged\n",
                         nevals, requested); CHKERRQ(ierr);
    }

    /// Display more information if requested
    ierr = PetscOptionsHasName(NULL,NULL,"-eig_verbose",&verbose); CHKERRQ(ierr);
    if (verbose == PETSC_TRUE)
    {
      /// Read out what options were set
      EPSProblemType Ptype;
      ierr = EPSGetProblemType(eps, &Ptype); CHKERRQ(ierr);
      ierr = PetscPrintf(topOpt->comm, "Problem type: %i\n", Ptype); CHKERRQ(ierr);

      EPSType type;
      ierr = EPSGetType(eps, &type); CHKERRQ(ierr);
      ierr = PetscPrintf(topOpt->comm, "Solution method: %s\n", type); CHKERRQ(ierr);

      EPSWhich which;
      ierr = EPSGetWhichEigenpairs(eps, &which); CHKERRQ(ierr);
      ierr = PetscPrintf(topOpt->comm, "Target eigenpairs: %i\n", which); CHKERRQ(ierr);

      /// Display convergence information
      EPSConvergedReason reason;
      EPSGetConvergedReason(eps, &reason);
      ierr = PetscPrintf(topOpt->comm, "EPS terminated on criteria: %i\n", reason); CHKERRQ(ierr);
      PetscInt iters;
      ierr = EPSGetIterationNumber(eps, &iters); CHKERRQ(ierr);
      ierr = PetscPrintf(topOpt->comm, "Eigenvalue solver converged after %li iterations\n",iters); CHKERRQ(ierr);
    }

    /// Pull out the converged eigenvalues
    for (short i = 0; i < nevals; i++)
    {
      ierr = EPSGetEigenpair(eps, i, lambda+i, 0, phi[i], 0); CHKERRQ(ierr);

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
      ierr = VecGhostUpdateBegin(phi[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecGhostUpdateEnd(phi[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecResetArray( phi[i] ); CHKERRQ(ierr);
      if (topOpt->myid == 0)
        cout << lambda[i] << "\n";
    }
      if (topOpt->myid == 0)
        cout << "\n";

    /// Destroy EPS objects
    ierr = VecDestroyVecs(topOpt->dynamicShape.cols(), &phi); CHKERRQ(ierr);
    //ST eps_st;
    ierr = EPSGetST(eps, &eps_st); CHKERRQ(ierr);
    KSP empty;
    ierr = KSPCreate(topOpt->comm, &empty); CHKERRQ(ierr);
    ierr = STSetKSP(eps_st, empty); CHKERRQ(ierr);
    ierr = EPSDestroy(&eps); CHKERRQ(ierr);
    ierr = KSPSetTolerances(topOpt->KUF, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 1e6); CHKERRQ(ierr);

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
    ierr = VecCreateMPI( topOpt->comm, topOpt->nLocElem, topOpt->nElem, &PETSc_grad ); CHKERRQ(ierr);
    ierr = VecDuplicate( PETSc_grad, &dlamdy ); CHKERRQ(ierr);
    for (short i = 0; i < nevals; i++)
    {
      ierr = VecPlaceArray( dlamdy, df.data()+i*df.rows() ); CHKERRQ(ierr);
      ierr = VecPlaceArray( PETSc_grad, grad+i*topOpt->nLocElem ); CHKERRQ(ierr);
      ierr = MatMultTranspose( topOpt->P, dlamdy, PETSc_grad ); CHKERRQ(ierr);

      ierr = VecResetArray(dlamdy); CHKERRQ(ierr);
      ierr = VecResetArray(PETSc_grad); CHKERRQ(ierr);
    }
    ierr = VecDestroy( &PETSc_grad ); VecDestroy( &dlamdy ); CHKERRQ(ierr);

    return 0;
  }

  /********************************************************************/
  /*                  Creates the diagonal mass matrix               **/
  /********************************************************************/
  int DiagMassFnc( TopOpt *topOpt, Mat &M, Eigen::VectorXd &dMdy )
  {
    // Initialize M
    MatCreate(topOpt->comm, &M);
    MatSetSizes(M, topOpt->numDims*topOpt->nLocNode, topOpt->numDims*topOpt->nLocNode,
                topOpt->numDims*topOpt->nNode, topOpt->numDims*topOpt->nNode);
    MatSetOptionsPrefix(M,"M_");
    MatSetFromOptions(M);
    ArrayXPI onDiag = ArrayXPI::Ones(topOpt->numDims*topOpt->nLocNode);
    ArrayXPI offDiag = ArrayXPI::Zero(topOpt->numDims*topOpt->nLocNode);
    MatXAIJSetPreallocation(M, 1, onDiag.data(), offDiag.data(), 0, 0);

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

    return 0;
  }

  /********************************************************************/
  /*      Gets M as a vector and returns M = M\K (freeDof only)      **/
  /********************************************************************/
  int VectorMassFnc( TopOpt *topOpt, Mat &M, Eigen::VectorXd &dMdy )
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

    return 0;
  }

  /********************************************************************/
  /*     Not ready yet, intended to make a consistent mass matrix    **/
  /********************************************************************/
  int MassFnc( TopOpt *topOpt, Mat &M, Eigen::VectorXd &dMdy )
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

    return 0;
  }
}
