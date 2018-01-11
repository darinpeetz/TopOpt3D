#include <iostream>
#include <fstream>
#include <numeric>
#include "Functions.h"
#include "TopOpt.h"
#include "EigLab.h"
#include "EigenPeetz.h"
#include <unsupported/Eigen/KroneckerProduct>

using namespace std;

namespace Functions
{
  int StressFnc( TopOpt *topOpt, Mat &Ks, Eigen::VectorXd &dKsdy );
  Eigen::MatrixXd sigtos(Eigen::VectorXd sigma);

  int Buckling( TopOpt *topOpt, double *lambda, double *grad, PetscInt &nevals )
  {
    PetscErrorCode ierr = 0;
    short NE = topOpt->element.cols(), DN = topOpt->numDims, DE = NE*DN;

    /// Assemble stress stiffness matrix and get sensitivity information
    Eigen::VectorXd dKsdy;
    Mat Ks;
    ierr = MatDuplicate( topOpt->K, MAT_SHARE_NONZERO_PATTERN, &Ks ); CHKERRQ(ierr);
    ierr = StressFnc( topOpt, Ks, dKsdy ); CHKERRQ(ierr);

    /// Remove fixed and spring dof from M (and K if necessary)
    ierr = MatZeroRowsColumns(Ks, topOpt->fixedDof.size(), topOpt->fixedDof.data(),
                       0.0, NULL, NULL); CHKERRQ(ierr);
    if (topOpt->nSpringDof > 0)
    {
      ierr = MatZeroRowsColumns(Ks, topOpt->springDof.size(),
                        topOpt->springDof.data(), 0.0, NULL, NULL); CHKERRQ(ierr);
      ierr = MatZeroRowsColumns(topOpt->K, topOpt->springDof.size(),
                        topOpt->springDof.data(), 1.0, NULL, NULL); CHKERRQ(ierr);
      ierr = KSPSetOperators(topOpt->KUF, topOpt->K, topOpt->K); CHKERRQ(ierr);
      ierr = KSPSetUp(topOpt->KUF); CHKERRQ(ierr);
    }

    /// Create JDMG instance
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
    jdmg.Set_Operators(Ks, topOpt->K);
    // Set target eigenvalues
    Nev_Type target_type = UNIQUE_LAST_NEV;
    jdmg.Set_Target(LR, nevals, UNIQUE_LAST_NEV);
    jdmg.Set_MaxIt((nevals+1)*50*(PetscInt)log(topOpt->nElem));
    jdmg.Set_Cycle(FMGCycle);
    jdmg.Set_Tol(1e-6);
    // Compute the eigenvalues
    ierr = PetscLogEventBegin(topOpt->JDCompEvent, 0, 0, 0, 0); CHKERRQ(ierr);
    ierr = jdmg.Compute(); CHKERRQ(ierr);
    ierr = PetscLogEventEnd(topOpt->JDCompEvent, 0, 0, 0, 0); CHKERRQ(ierr);

    // Get the results
    PetscInt nev_conv = 0;
    jdmg.Get_nev_conv(nev_conv);
    nev_conv = (target_type == TOTAL_NEV) ? nev_conv : min(nevals, nev_conv);
    jdmg.Get_EigenValues(lambda);

    // Return if sensitivities aren't needed
    if (grad == NULL)
      return 0;

    Vec *phi, phi_copy;
    jdmg.Get_EigenVectors(&phi);

    ierr = VecDuplicate(topOpt->U, &phi_copy); CHKERRQ(ierr);
    for (int i = 0; i < nev_conv; i++)
    {
      ierr = VecPlaceArray(phi_copy, topOpt->bucklingShape.data() +
                      i*topOpt->bucklingShape.rows()); CHKERRQ(ierr);
      ierr = VecCopy(phi[i], phi_copy);
      ierr = VecGhostUpdateBegin(phi_copy, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecGhostUpdateEnd(phi_copy, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecResetArray(phi_copy);
    }
    ierr = VecDestroy(&phi_copy); CHKERRQ(ierr);

    /// Dot product of eigenvectors expanded to triplet form
    /// to match unassembled stiffness matrices
    Eigen::MatrixXd phim( (DE*DE)*topOpt->gElem.rows(), nev_conv );
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
              topOpt->bucklingShape.block(eDof(j),0,1,nev_conv).cwiseProduct(
                              topOpt->bucklingShape.block(eDof(i),0,1,nev_conv));
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
    Eigen::MatrixXd dKsdU = Eigen::MatrixXd::Zero(topOpt->node.size(), nev_conv);
    Eigen::MatrixXd v = dKsdU;
    const PetscScalar *p_Es;
    ierr = VecGetArrayRead(topOpt->Es, &p_Es); CHKERRQ(ierr);
    for (PetscInt el = 0; el < topOpt->element.rows(); el++)
    {
      Eigen::MatrixXd dKs = p_Es[el] * topOpt->dKsdu;
      for (int nd = 0; nd < NE; nd++)
      {
        if (topOpt->element(el,nd) < topOpt->nLocNode)
          dKsdU.block(DN*topOpt->element(el,nd), 0, DN, nev_conv) +=
            dKs.block(0, DN*nd, DE*DE, DN).transpose() *
            phim.block(el*DE*DE, 0, DE*DE, nev_conv);
      }
    }
    ierr = VecRestoreArrayRead(topOpt->Es, &p_Es); CHKERRQ(ierr);

    PC smooth_pc;
    KSP smooth_ksp;
    PetscInt nlevels;
    /// Switch to weighted Jacobi smoothing 
    ierr = PCMGGetLevels(pcmg, &nlevels); CHKERRQ(ierr);
    for (int i = 1; i < nlevels; i++)
    {
      ierr = PCMGGetSmoother(pcmg, i, &smooth_ksp); CHKERRQ(ierr);
      ierr = KSPSetType(smooth_ksp, KSPRICHARDSON); CHKERRQ(ierr);
      ierr = KSPRichardsonSetScale(smooth_ksp, 5.0/10.0); CHKERRQ(ierr);
      ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
      ierr = PCSetType(smooth_pc, PCJACOBI); CHKERRQ(ierr);
    }

    /// Solve the adjoint problem
    Vec dKsdU_vec;
    ierr = VecDuplicate(topOpt->U, &dKsdU_vec); CHKERRQ(ierr);
    Vec v_vec;
    ierr = VecDuplicate(topOpt->U, &v_vec); CHKERRQ(ierr);

    for (short i = 0; i < nev_conv; i++)
    {
      ierr = VecPlaceArray( dKsdU_vec, dKsdU.data() + i*dKsdU.rows() ); CHKERRQ(ierr);
      ierr = VecPlaceArray( v_vec, v.data() + i*dKsdU.rows() ); CHKERRQ(ierr);
      ierr = VecSet(v_vec, 0.0); CHKERRQ(ierr);
      ierr = KSPSolve( topOpt->KUF, dKsdU_vec, v_vec ); CHKERRQ(ierr);
      PetscInt its;
      ierr = KSPGetIterationNumber(topOpt->KUF, &its); CHKERRQ(ierr);
      KSPConvergedReason reason;
      ierr = KSPGetConvergedReason(topOpt->KUF, &reason); CHKERRQ(ierr);
      if (topOpt->verbose >= 1)
      {
        ierr = PetscFPrintf(topOpt->comm, topOpt->output, "Solve for adjoint equation #%i converged in %i iterations with reason: %i\n",
                            i, its, reason); CHKERRQ(ierr);
      }
      ierr = VecGhostUpdateBegin(v_vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecGhostUpdateEnd(v_vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecResetArray( dKsdU_vec ); CHKERRQ(ierr);
      ierr = VecResetArray( v_vec ); CHKERRQ(ierr);
    }
    ierr = VecDestroy( &dKsdU_vec ); CHKERRQ(ierr);
    ierr = VecDestroy( &v_vec ); CHKERRQ(ierr);

    Eigen::MatrixXd vm( (DE*DE)*topOpt->nLocElem, nev_conv );
    Eigen::VectorXd Um( (DE*DE)*topOpt->nLocElem );
    const PetscScalar *p_U;
    ierr = VecGetArrayRead(topOpt->U, &p_U); CHKERRQ(ierr);
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
          vm.row((DE*DE)*el + DE*i + j) = v.block(eDof(j),0,1,nev_conv);
          Um((DE*DE)*el + DE*i + j) = p_U[eDof[i]];
        }
      }
    }
    ierr = VecRestoreArrayRead(topOpt->U, &p_U); CHKERRQ(ierr);

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
    Eigen::MatrixXd df = Eigen::MatrixXd::Zero(dKdy.rows(),nevals);
    nevals = 0;
    for (short j = 0; j < nev_conv; j++)
    {
      if (abs(1.0-lambda[nevals]/lambda[j]) > 1e-5)
        nevals++;   
      df.col(nevals) += phim.block(0,j,dKdy.rows(),1).cwiseProduct(dKsdy-lambda[j]*dKdy)
              + vm.col(j).cwiseProduct(dKdy.cwiseProduct(Um));
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
    ierr = MatDestroy(&Ks); CHKERRQ(ierr);

    if (nev_conv < nevals)
    {
      PetscFPrintf(topOpt->comm, topOpt->output, "***************************************************\n");
      PetscFPrintf(topOpt->comm, topOpt->output, "Warning, nev_conv < nevals\n");
      PetscFPrintf(topOpt->comm, topOpt->output, "***************************************************\n");
    }

    return 0;
  }

  int StressFnc( TopOpt *topOpt, Mat &Ks, Eigen::VectorXd &dKsdy )
  {
    PetscErrorCode ierr = 0;
    // Mesh characteristics
    const short NE = topOpt->element.cols(), DN = topOpt->numDims, DE = NE*DN;
    // Track construction of Ks, dKs
    long dksmarker = 0;
    dKsdy.resize( topOpt->nLocElem*(long)pow(DE,2) );

    // Get pointers to Petsc vectors
    const PetscScalar *p_Es, *p_dEsdy, *p_U;
    ierr = VecGetArrayRead(topOpt->Es, &p_Es); CHKERRQ(ierr);
    ierr = VecGetArrayRead(topOpt->dEsdy, &p_dEsdy); CHKERRQ(ierr);
    ierr = VecGetArrayRead(topOpt->U, &p_U); CHKERRQ(ierr);

    Eigen::MatrixXd ks = Eigen::MatrixXd::Zero(DE, DE);
    Eigen::Map< Eigen::VectorXd > ksVec(ks.data(), ks.size());
    /// Loop over elements
    for (long el = 0; el < topOpt->element.rows(); el++)
    {
      ks.setZero();
      Eigen::VectorXd u(DE);

      /// Get fem solution for this element
      for (short n = 0; n < NE; n++)
      {
        for (short d = 0; d < DN; d++)
        {
          u(d + n*DN) = p_U[DN*topOpt->element(el, n) + d];
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

      /// Loop over nodes to fill in KS
      // First get list of global node numbers for this element
      std::vector<PetscInt> cols(NE);
      for (int nd = 0; nd < NE; nd++) // Looping over rows
        cols[nd] = topOpt->gNode(topOpt->element(el,nd));
      // Now construct
      ks *= -p_Es[el];
      for (int nd = 0; nd < NE; nd++) // Looping over rows
      {
        PetscInt node = topOpt->element(el,nd);
        if (node < topOpt->nLocNode) // If node is local to this process
        {
          ierr = MatSetValuesBlocked(Ks, 1, topOpt->gNode.data()+node,
            NE, cols.data(), ks.data() + DE*DN*nd, ADD_VALUES);
          CHKERRQ(ierr);
        }
      }

    }

    ierr = MatAssemblyBegin(Ks, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(topOpt->Es, &p_Es); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(topOpt->dEsdy, &p_dEsdy); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(topOpt->U, &p_U); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Ks, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    return 0;
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
