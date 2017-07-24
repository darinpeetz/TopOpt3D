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
  void StressFnc( TopOpt *topOpt, Mat &Ks, Eigen::VectorXd &dKsdy );
  Eigen::MatrixXd sigtos(Eigen::VectorXd sigma);

  void Buckling( TopOpt *topOpt, double *lambda, double *grad, PetscInt &nevals )
  {
    short NE = topOpt->element.cols(), DN = topOpt->numDims, DE = NE*DN;

    /// Assemble stress stiffness matrix and get sensitivity information
    Eigen::VectorXd dKsdy;
    Mat Ks;
    MatDuplicate( topOpt->K, MAT_SHARE_NONZERO_PATTERN, &Ks );
    StressFnc( topOpt, Ks, dKsdy );

    /// Create eigenvectors
    Vec *phi;
    VecDuplicateVecs(topOpt->U, nevals, &phi);

    /// Apply Dirichlet B.C.'s
    MatZeroRowsColumns(Ks, topOpt->fixedDof.size(), topOpt->fixedDof.data(),
                       0.0, NULL, NULL);
    if (topOpt->nSpringDof > 0)
    {
     MatZeroRowsColumns(Ks, topOpt->springDof.size(),
                        topOpt->springDof.data(), 0.0, NULL, NULL);
     MatZeroRowsColumns(topOpt->K, topOpt->springDof.size(),
                        topOpt->springDof.data(), 1.0, NULL, NULL);
    }

    /// Set up the generalized eigenvalue problem
    EPS eps; // The eigensolver context
    EPSCreate(topOpt->comm, &eps);
    EPSSetType(eps, EPSKRYLOVSCHUR);
    EPSSetProblemType(eps, EPS_GHEP);
    EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL);
    EPSSetOperators(eps, Ks, topOpt->K);

    /// For using spectrum slicing eigenvalue method
    /*EPSSetType(eps, EPSCISS);
    EPSSetWhichEigenpairs(eps, EPS_ALL);
    RG rg;
    EPSGetRG(eps, &rg);
    RGSetType(rg, RGINTERVAL);
    RGIntervalSetEndpoints(rg, 0.026, 0.4, 0, 0);*/

    EPSSetDimensions(eps, nevals, PETSC_DEFAULT, PETSC_DEFAULT);
    EPSSetTolerances(eps, 1e-8, PETSC_DEFAULT);
    EPSSetFromOptions(eps);

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

    /// Check if this routine has been called before and if we need to create eigenvectors
    if (topOpt->bucklingShape.size() == 0)
    {
      topOpt->bucklingShape.setZero(topOpt->node.size(), nevals);
      topOpt->v.setZero(topOpt->node.size(), nevals);
      for (int i = 0; i < nevals; i++)
        VecPlaceArray(phi[i], topOpt->bucklingShape.data() +
                      i*topOpt->bucklingShape.rows());
    }
    else
    {
      for (int i = 0; i < nevals; i++)
        VecPlaceArray(phi[i], topOpt->bucklingShape.data() +
                      i*topOpt->bucklingShape.rows());
      EPSSetInitialSpace(eps, nevals, phi);
    }

    /// Perform the solve and look at the results
    EPSSolve(eps);
    PetscBool verbose = PETSC_FALSE;
    PetscInt requested = nevals;
    EPSGetConverged(eps, &nevals);
    nevals = std::min(nevals, requested);
    if (requested != nevals)
      PetscPrintf(topOpt->comm, "Only %li of %li eigenvalues converged\n", nevals, requested);

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
      VecGhostUpdateBegin(phi[i], INSERT_VALUES, SCATTER_FORWARD);
      VecGhostUpdateEnd(phi[i], INSERT_VALUES, SCATTER_FORWARD);
      VecResetArray( phi[i] );
    }

    /// Destroy EPS objects
    VecDestroyVecs(topOpt->bucklingShape.cols(), &phi);
    ST eps_st;
    EPSGetST(eps, &eps_st);
    KSP empty;
    KSPCreate(topOpt->comm, &empty);
    STSetKSP(eps_st, empty);
    EPSDestroy(&eps);

    /// Dot product of eigenvectors expanded to triplet form
    /// to match unassembled stiffness matrices
    Eigen::MatrixXd phim( (DE*DE)*topOpt->gElem.rows(), nevals );
    for (long el = 0; el < topOpt->gElem.rows(); el++)
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
              topOpt->bucklingShape.block(eDof(j),0,1,nevals).cwiseProduct(
                              topOpt->bucklingShape.block(eDof(i),0,1,nevals));
        }
      }
    }

    /// Stress Stiffness partial with respect to u (Es is factored out so this step is only needed once)
    if (topOpt->dKsdu.size() == 0)
    {
      short dDE = DE*DE;
      topOpt->dKsdu.setZero(dDE , DE);
      // Loop over dof of a single element
      for (int dof = 0; dof < DE; dof++)
      {
        Eigen::Map< Eigen::MatrixXd > dksdu( topOpt->dKsdu.data() + dDE*dof, DE, DE );
        Eigen::VectorXd du = Eigen::VectorXd::Zero(DE);
        du(dof) = 1;
        // Loop through quadrature points
        for (int qp = 0; qp < 4; qp++)
          dksdu += topOpt->W[qp]*topOpt->GT[qp]*sigtos(topOpt->d*topOpt->B[qp]*du)
                *  topOpt->G[qp]*topOpt->detJ;
      }
    }

    /// Construct adjoint vectors to be solved
    Eigen::MatrixXd dKsdU = Eigen::MatrixXd::Zero(topOpt->node.size(), nevals);
    const PetscScalar *p_Es;
    VecGetArrayRead(topOpt->Es, &p_Es);
    for (PetscInt el = 0; el < topOpt->element.rows(); el++)
    {
      Eigen::MatrixXd dKs = p_Es[el] * topOpt->dKsdu;
      for (int nd = 0; nd < NE; nd++)
      {
        if (topOpt->element(el,nd) < topOpt->nLocNode)
          dKsdU.block(DN*topOpt->element(el,nd), 0, DN, nevals) +=
            dKs.block(0, DN*nd, DE*DE, DN).transpose() *
            phim.block(el*DE*DE, 0, DE*DE, nevals);
      }
    }
    VecRestoreArrayRead(topOpt->Es, &p_Es);

    /// Solve the adjoint problem
    Vec dKsdU_vec;
    VecDuplicate(topOpt->U, &dKsdU_vec);
    Vec v_vec;
    VecDuplicate(topOpt->U, &v_vec);

    for (short i = 0; i < nevals; i++)
    {
      VecPlaceArray( dKsdU_vec, dKsdU.data() + i*dKsdU.rows() );
      VecPlaceArray( v_vec, topOpt->v.data() + i*dKsdU.rows() );
      MatZeroRowsColumns(topOpt->K, topOpt->fixedDof.size(), topOpt->fixedDof.data(),
                         1.0, v_vec, dKsdU_vec);
      KSPSolve( topOpt->KUF, dKsdU_vec, v_vec );
      VecGhostUpdateBegin(v_vec, INSERT_VALUES, SCATTER_FORWARD);
      VecGhostUpdateEnd(v_vec, INSERT_VALUES, SCATTER_FORWARD);
      VecResetArray( dKsdU_vec );
      VecResetArray( v_vec );
    }
    VecDestroy( &dKsdU_vec );
    VecDestroy( &v_vec );

    Eigen::MatrixXd vm( (DE*DE)*topOpt->nLocElem, nevals );
    Eigen::VectorXd Um( (DE*DE)*topOpt->nLocElem );
    const PetscScalar *p_U;
    VecGetArrayRead(topOpt->U, &p_U);
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
          vm.row((DE*DE)*el + DE*i + j) = topOpt->v.block(eDof(j),0,1,nevals);
          Um((DE*DE)*el + DE*i + j) = p_U[eDof[i]];
        }
      }
    }
    VecRestoreArrayRead(topOpt->U, &p_U);

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
    Eigen::MatrixXd df(dKdy.rows(),nevals);
    for (short i = 0; i < nevals; i++)
      df.col(i) = phim.block(0,i,dKdy.rows(),1).cwiseProduct(dKsdy-lambda[i]*dKdy)
                + vm.col(i).cwiseProduct(dKdy.cwiseProduct(Um));
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

    return;
  }

  void StressFnc( TopOpt *topOpt, Mat &Ks, Eigen::VectorXd &dKsdy )
  {
    // Mesh characteristics
    const short NE = topOpt->element.cols(), DN = topOpt->numDims, DE = NE*DN;
    // Track construction of Ks, dKs
    long ksmarker = 0, dksmarker = 0;
    dKsdy.resize( topOpt->nLocElem*(long)pow(DE,2) );

    // Get pointers to Petsc vectors
    const PetscScalar *p_Es, *p_dEsdy, *p_U;
    VecGetArrayRead(topOpt->Es, &p_Es);
    VecGetArrayRead(topOpt->dEsdy, &p_dEsdy);
    VecGetArrayRead(topOpt->U, &p_U);

    Eigen::MatrixXd ks = Eigen::MatrixXd::Zero(DE, DE);
    Eigen::Map< Eigen::VectorXd > ksVec(ks.data(), ks.size());
    /// Loop over elements
    for (long el = 0; el < topOpt->element.rows(); el++)
    {
      ks.setZero();
      Eigen::VectorXd u(DE);
      bool islocal[DE];

      /// Get fem solution for this element
      for (short n = 0; n < NE; n++)
      {
        for (short d = 0; d < DN; d++)
        {
          u(d + n*DN) = p_U[DN*topOpt->element(el, n) + d];
          islocal[topOpt->numDims*n + d] =
              (topOpt->element(el, n) < topOpt->nLocNode);
        }
      }

      /// Loop over quadrature points
      for (short qp = 0; qp < 4; qp++)
      {
        ks += topOpt->W[qp] * topOpt->GT[qp]
            * sigtos(topOpt->d * topOpt->B[qp] * u)
            * topOpt->G[qp] * topOpt->detJ;
      }

      /// Fill in dKsdy for local elements
      if (el < topOpt->nLocElem)
      {
        dKsdy.segment(dksmarker, ksVec.size()) = -p_dEsdy[el]*ksVec;
        dksmarker += ksVec.size();
      }

      /// Loop over indices to fill in KS
      for (int i = 0; i < DE; i++) // Looping over rows
      {
        if (islocal[i]) // If node is local to this process
        {
          for (int j = 0; j < DE; j++) // Looping over cols
          {
            PetscScalar v = -p_Es[el] * ks(i,j);
            MatSetValue(Ks, topOpt->i[ksmarker], topOpt->j[ksmarker], v, ADD_VALUES);
            ksmarker++;
          }
        }
      }

    }

    MatAssemblyBegin(Ks, MAT_FINAL_ASSEMBLY);
    VecRestoreArrayRead(topOpt->Es, &p_Es);
    VecRestoreArrayRead(topOpt->dEsdy, &p_dEsdy);
    VecRestoreArrayRead(topOpt->U, &p_U);
    MatAssemblyEnd(Ks, MAT_FINAL_ASSEMBLY);

    return;
  }

  Eigen::MatrixXd sigtos(Eigen::VectorXd sigma)
  {
    switch (sigma.size())
    {
      case 1: //1-D
      {
        return sigma;
        break;
      }
      case 3:
      {
        Eigen::MatrixXd s = Eigen::MatrixXd::Zero(4 , 4);
        s(0,0) = sigma(0);
        s(1,1) = sigma(1);
        s(0,1) = sigma(2);
        s(1,0) = sigma(2);
        s.block(2, 2, 2, 2) = s.block(0, 0, 2, 2);
        return s;
        break;
      }
      case 6:
      {
        Eigen::MatrixXd s = Eigen::MatrixXd::Zero(9 , 9);
        // Normal stresses
        s(0,0) = sigma(0);
        s(1,1) = sigma(1);
        s(2,2) = sigma(2);
        // xy shear
        s(0,1) = sigma(3);
        s(1,0) = sigma(3);
        // xz shear
        s(0,2) = sigma(5);
        s(2,0) = sigma(5);
        // yz shear
        s(1,2) = sigma(4);
        s(2,1) = sigma(4);
        s.block(3, 3, 3, 3) = s.block(0, 0, 3, 3);
        s.block(6, 6, 3, 3) = s.block(0, 0, 3, 3);
        return s;
        break;
      }
      default:
        std::cout << "INVALID SIZE OF STRESS VECTOR\n";
        break;
    }

    return sigma;
  }
}
