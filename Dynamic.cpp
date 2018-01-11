#include <iostream>
#include <fstream>
#include <numeric>
#include <math.h>
#include "Functions.h"
#include "TopOpt.h"
#include "EigLab.h"
#include "EigenPeetz.h"
#include <unsupported/Eigen/KroneckerProduct>
#include <sstream>

using namespace std;
typedef Eigen::Matrix<PetscScalar, -1, -1> MatrixPS;
typedef Eigen::Matrix<PetscScalar, -1, 1>  VectorPS;

namespace Functions
{
  int DiagMassFnc( TopOpt *topOpt, Mat &M, Eigen::VectorXd &dMdy );

  int Dynamic( TopOpt *topOpt, double *lambda, double *grad, PetscInt &nevals )
  {
    PetscErrorCode ierr = 0;
    short NE = topOpt->element.cols(), DN = topOpt->numDims, DE = NE*DN;

    /// Assemble Mass matrix and get sensitivity information
    Eigen::VectorXd dMdy;
    Mat M;
    ierr = DiagMassFnc( topOpt, M, dMdy ); CHKERRQ(ierr);

    /// Remove fixed and spring dof from M (and K if necessary)
    ierr = MatZeroRowsColumns(M, topOpt->fixedDof.size(),
                       topOpt->fixedDof.data(), 0.0, NULL, NULL); CHKERRQ(ierr);
    if (topOpt->nSpringDof > 0)
    {
      ierr = MatZeroRowsColumns(M, topOpt->springDof.size(),
                         topOpt->springDof.data(), 0.0, NULL, NULL); CHKERRQ(ierr);
      ierr = MatZeroRowsColumns(topOpt->K, topOpt->springDof.size(),
                         topOpt->springDof.data(), 10000.0, NULL, NULL); CHKERRQ(ierr);
      ierr = KSPSetOperators(topOpt->KUF, topOpt->K, topOpt->K); CHKERRQ(ierr);
    }
    ierr = KSPSetUp(topOpt->KUF); CHKERRQ(ierr);

    // Create JDMG instance
    JDMG jdmg(topOpt->comm);
    jdmg.Set_Verbose(topOpt->verbose);
    jdmg.Set_File(topOpt->output);
    // Get restrictors from FEM problem
    PC pcmg; PCType pctype;
    ierr = KSPGetPC(topOpt->KUF, &pcmg); CHKERRQ(ierr);
    ierr = PCGetType(pcmg, &pctype); CHKERRQ(ierr);
    /*if (!strcmp(pctype,PCMG))
    {*/
      ierr = jdmg.Set_Hierarchy(topOpt->PR); CHKERRQ(ierr);
    /*}
    else if (!strcmp(pctype,PCGAMG))
    {
      ierr = jdmg.PCMG_Extract(pcmg); CHKERRQ(ierr);
    }
    else
       SETERRQ1(topOpt->comm, PETSC_ERR_ARG_WRONG, "Preconditioner of type %s was provided, but must be one of mg or gamg", pctype);*/
    // Set Operators
    jdmg.Set_Operators(M, topOpt->K);
    // Set target eigenvalues
    Nev_Type target_type = UNIQUE_LAST_NEV;
    jdmg.Set_Target(LR, nevals, target_type);
    jdmg.Set_MaxIt(100*(PetscInt)log(topOpt->nElem));
    jdmg.Set_Cycle(FMGCycle);
    // Compute the eigenvalues
    ierr = PetscLogEventBegin(topOpt->JDCompEvent, 0, 0, 0, 0); CHKERRQ(ierr);
    ierr = jdmg.Compute(); CHKERRQ(ierr);
    ierr = PetscLogEventEnd(topOpt->JDCompEvent, 0, 0, 0, 0); CHKERRQ(ierr);

    // Get the results
    PetscInt nev_conv = 0;
    jdmg.Get_nev_conv(nev_conv);
    (target_type == TOTAL_NEV) ? : nev_conv--;
    jdmg.Get_EigenValues(lambda);

    // Return if sensitivities aren't needed
    if (grad == NULL)
      return 0;

    Vec *phi, phi_copy;
    jdmg.Get_EigenVectors(&phi);
    ierr = VecDuplicate(topOpt->U, &phi_copy); CHKERRQ(ierr);
    for (int i = 0; i < nev_conv; i++)
    {
      ierr = VecPlaceArray(phi_copy, topOpt->dynamicShape.data() +
                      i*topOpt->dynamicShape.rows()); CHKERRQ(ierr);
      ierr = VecCopy(phi[i], phi_copy);
      ierr = VecGhostUpdateBegin(phi_copy, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecGhostUpdateEnd(phi_copy, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecResetArray(phi_copy);
    }
    ierr = VecDestroy(&phi_copy); CHKERRQ(ierr);

    /// Dot product of eigenvectors expanded to triplet form
    /// to match unassembled stiffness matrices
    Eigen::MatrixXd phim( (DE*DE)*topOpt->nLocElem, nev_conv );
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
            topOpt->dynamicShape.block(eDof(j),0,1,nev_conv).cwiseProduct(
                            topOpt->dynamicShape.block(eDof(i),0,1,nev_conv));
        }
      }
    }

    /// Construct sensitivity of material stiffness matrix
    const PetscScalar *p_dEdy;
    ierr = VecGetArrayRead(topOpt->dEdy, &p_dEdy); CHKERRQ(ierr);
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
    ierr = VecRestoreArrayRead(topOpt->dEdy, &p_dEdy); CHKERRQ(ierr);

    /// Construct sensitivity
    Eigen::MatrixXd df = Eigen::MatrixXd::Zero((DE*DE)*topOpt->nLocElem,nevals);
    nevals = 0;
    for (short j = 0; j < nev_conv; j++)
    {
      if (abs(1.0-lambda[nevals]/lambda[j]) > 1e-5)
        nevals++;
      df.col(nevals) += phim.col(j).cwiseProduct(dMdy-lambda[j]*dKdy);
      lambda[nevals] = lambda[j];
    }
    df.conservativeResize(df.rows(), ++nevals);

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
    ierr = VecDestroy( &PETSc_grad ); ierr = VecDestroy( &dlamdy ); CHKERRQ(ierr);
    ierr = MatDestroy(&M); CHKERRQ(ierr);

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
}
