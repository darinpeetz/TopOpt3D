#include <numeric>
#include "Functions.h"
#include "TopOpt.h"
#include "EigLab.h"
#include <unsupported/Eigen/KroneckerProduct>

/********************************************************************
 * Compute principal buckling modes and their sensitivities
 * 
 * @param topOpt: The topology optimization object
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode Stability::Function(TopOpt *topOpt)
{
  PetscErrorCode ierr = 0;
  short NE = topOpt->element.cols(), DN = topOpt->numDims, DE = NE*DN;
  PC pc;

  /// Assemble stress stiffness matrix and get sensitivity information
  if (Ks == NULL) {
    dKsdy.resize(topOpt->nLocElem*(long)std::pow(DE,2));
    ierr = MatDuplicate(topOpt->K, MAT_SHARE_NONZERO_PATTERN, &Ks); CHKERRQ(ierr);
  }
  ierr = StressFnc(topOpt); CHKERRQ(ierr);

  /// Remove fixed dof from Ks
  ierr = MatZeroRowsColumns(Ks, topOpt->fixedDof.size(), topOpt->fixedDof.data(),
                            0.0, NULL, NULL); CHKERRQ(ierr);
  if (topOpt->nEigFixDof > 0) { // Fix additional parts of matrices if requested
    ierr = MatZeroRowsColumns(Ks, topOpt->eigenFixedDof.size(),
            topOpt->eigenFixedDof.data(), 0.0, NULL, NULL); CHKERRQ(ierr);
    ierr = MatZeroRowsColumns(topOpt->K, topOpt->eigenFixedDof.size(),
            topOpt->eigenFixedDof.data(), 1e8, NULL, NULL); CHKERRQ(ierr);
    ierr = MatSetNullSpace(topOpt->K, NULL); CHKERRQ(ierr);
    ierr = KSPSetOperators(topOpt->KUF, topOpt->K, topOpt->K); CHKERRQ(ierr);
    ierr = KSPSetUp(topOpt->KUF); CHKERRQ(ierr);

    // If we're using extra fixed dof for stability, make sure coarse grid is
    // LU and not expensive pseudo-inverse
    KSP smooth_ksp, *sub_ksp; PC smooth_pc, sub_pc; PetscInt blocks, first;
    ierr = KSPGetPC(topOpt->KUF, &pc); CHKERRQ(ierr);
    ierr = PCMGGetCoarseSolve(pc, &smooth_ksp); CHKERRQ(ierr);
    ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
    ierr = PCSetType(smooth_pc, PCBJACOBI); CHKERRQ(ierr);
    ierr = PCSetUp(smooth_pc); CHKERRQ(ierr);
    ierr = PCBJacobiGetSubKSP(smooth_pc, &blocks, &first, &sub_ksp); CHKERRQ(ierr);
    ierr = KSPGetPC(sub_ksp[0], &sub_pc); CHKERRQ(ierr);
    ierr = PCSetType(sub_pc, PCLU); CHKERRQ(ierr);
    ierr = PCFactorSetShiftType(sub_pc, MAT_SHIFT_INBLOCKS); CHKERRQ(ierr);
    ierr = KSPSetTolerances(sub_ksp[0], PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT, 1); CHKERRQ(ierr);
    ierr = KSPSetType(sub_ksp[0], KSPPREONLY); CHKERRQ(ierr);
    ierr = PCSetUp(sub_pc); CHKERRQ(ierr);
  }

  // Set ouptput parameters for lopgmres
  lopgmres.Set_Verbose(topOpt->verbose);
  lopgmres.Set_File(topOpt->output);

  // Set the preconditioner
  ierr = KSPGetPC(topOpt->KUF, &pc); CHKERRQ(ierr);
  ierr = lopgmres.Set_PC(pc); CHKERRQ(ierr);

  // Set Operators
  lopgmres.Set_Operators(Ks, topOpt->K);
  // Set target eigenvalues
  Nev_Type target_type = TOTAL_NEV;
  lopgmres.Set_Target(LR, nvals, target_type);
  lopgmres.Set_MaxIt(500);//150*(PetscInt)std::log(topOpt->nElem));
  lopgmres.Set_Tol(std::pow(10,std::log10(2*topOpt->nNode)/2-9));
  // Compute the eigenvalues
  double tEigStart = MPI_Wtime();
  ierr = lopgmres.Compute(); CHKERRQ(ierr);
  double tEigEnd = MPI_Wtime();
  PetscInt itEig = lopgmres.Get_It();

  // Get the results
  PetscInt nev_conv = lopgmres.Get_nev_conv();
  if (nev_conv == 0) {
    char name_suffix[30];
    sprintf(name_suffix, "_eigen_failure");
    ierr = topOpt->PrintVals(name_suffix); CHKERRQ(ierr);
    SETERRQ(topOpt->comm, PETSC_ERR_CONV_FAILED, "Eigensolver found 0 eigenvalues\n");
  } 
  ArrayXPS lambda(nev_conv);
  lopgmres.Get_Eigenvalues(lambda.data());

  // Aggregate eigenvalues
  PetscScalar p = 8; //TODO: make this an option to set
  values[0] = std::pow((PetscScalar)lambda.pow(p).sum(), 1/p);

  // Return if sensitivities aren't needed
  if (calc_gradient == PETSC_FALSE)
    return 0;

  // Make sure we have enough room for all eigenvectors
  Vec *phi, phi_copy;
  lopgmres.Get_Eigenvectors(&phi);
  topOpt->bucklingShape.resize(topOpt->bucklingShape.rows(), nev_conv);
  ierr = VecDuplicate(topOpt->U, &phi_copy); CHKERRQ(ierr);
  for (int i = 0; i < nev_conv; i++) {
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
  MatrixXPS phim((DE*DE)*topOpt->gElem.rows(), nev_conv);
  for (long el = 0; el < topOpt->gElem.rows(); el++) {
    ArrayXPI eDof(DE);
    for (int i = 0; i < NE; i++) {
      for (int j = 0; j < DN; j++)
        eDof(i*DN + j) = DN*topOpt->element(el, i) + j;
    }

    for (int i = 0; i < DE; i++) {
      for (int j = 0; j < DE; j++) {
        phim.row((DE*DE)*el + DE*i + j) =
          topOpt->bucklingShape.block(eDof(j),0,1,nev_conv).cwiseProduct(
                  topOpt->bucklingShape.block(eDof(i),0,1,nev_conv));
      }
    }
  }

  /// Stress Stiffness partial with respect to u
  /// (Es is factored out so this step is only needed once)
  if (this->dKsdu.size() == 0) {
    this->dKsdu.resize(DE);
    // Loop over dof of a single element
    for (int dof = 0; dof < DE; dof++) {
      this->dKsdu[dof].setZero(DE, DE);
      Eigen::VectorXd du = Eigen::VectorXd::Zero(DE);
      du(dof) = 1;
      // Loop through quadrature points
      for (int qp = 0; qp < pow(2, topOpt->numDims); qp++)
        this->dKsdu[dof] += topOpt->W[qp]*topOpt->GT[qp] *
                            sigtos(topOpt->d*topOpt->B[qp]*du) *
                            topOpt->G[qp]*topOpt->detJ;
    }
  }

  /// Construct adjoint vectors to be solved
  if (topOpt->verbose >= 2) {
    ierr = PetscFPrintf(topOpt->comm, topOpt->output, "Preparing to solve buckling"
                        " adjoint equations\n"); CHKERRQ(ierr);
  }
  MatrixXPS dKsdU = MatrixXPS::Zero(topOpt->node.size(), nev_conv);
  MatrixXPS phi_loc(DE, nev_conv);
  v.resize(dKsdU.rows(), dKsdU.cols());
  const PetscScalar *p_Es;
  ierr = VecGetArrayRead(topOpt->Es, &p_Es); CHKERRQ(ierr);
  for (PetscInt el = 0; el < topOpt->element.rows(); el++) {
    // Get the local parts of the eigenvectors in phi_loc
    for (int j = 0; j < NE; j++) {
      phi_loc.block(DN*j, 0, DN, nev_conv) =
            topOpt->bucklingShape.block(DN*topOpt->element(el, j), 0, DN, nev_conv);
    }
    // Take inner product of phi^T*dKsdu*phi for each dof in the element
    for (int nd = 0; nd < NE; nd++) {
      if (topOpt->element(el,nd) < topOpt->nLocNode) {
        for (int dof = 0; dof < topOpt->numDims; dof++) {
          MatrixXPS dKs = p_Es[el] * this->dKsdu[DN*nd + dof];
          dKsdU.row(DN*topOpt->element(el,nd)+dof) +=
            phi_loc.cwiseProduct(dKs*phi_loc).colwise().sum();
        }
      }
    }
  }
  ierr = VecRestoreArrayRead(topOpt->Es, &p_Es); CHKERRQ(ierr);

  /// Solve the adjoint problem
  // Vectors to be used with the solver
  Vec dKsdU_vec;
  ierr = VecDuplicate(topOpt->U, &dKsdU_vec); CHKERRQ(ierr);
  Vec v_vec;
  ierr = VecDuplicate(topOpt->U, &v_vec); CHKERRQ(ierr);

  // Zero out fixed dof in the rhs
  for (unsigned int i = 0; i < topOpt->fixedDof.size(); i++)
    dKsdU.row(topOpt->fixedDof[i]-topOpt->numDims*topOpt->nddist[topOpt->myid]).setZero();
  for (unsigned int i = 0; i < topOpt->eigenFixedDof.size(); i++)
    dKsdU.row(topOpt->eigenFixedDof[i]-topOpt->numDims*topOpt->nddist[topOpt->myid]).setZero();

  // Solving each adjoint problem
  double tAdjoint = 0;
  PetscInt itAdjoint = 0;
  for (short i = 0; i < nev_conv; i++) {
    ierr = VecPlaceArray(dKsdU_vec, dKsdU.data() + i*dKsdU.rows()); CHKERRQ(ierr);
    ierr = VecPlaceArray(v_vec, v.data() + i*dKsdU.rows()); CHKERRQ(ierr);
    ierr = VecSet(v_vec, 0.0); CHKERRQ(ierr);
    double t0 = MPI_Wtime();
    ierr = KSPSolve(topOpt->KUF, dKsdU_vec, v_vec); CHKERRQ(ierr);
    double t1 = MPI_Wtime();
    PetscInt its;
    ierr = KSPGetIterationNumber(topOpt->KUF, &its); CHKERRQ(ierr);
    tAdjoint += t1-t0;
    itAdjoint += its;
    KSPConvergedReason reason;
    ierr = KSPGetConvergedReason(topOpt->KUF, &reason); CHKERRQ(ierr);
    if (topOpt->verbose >= 1 and reason > 0) {
      ierr = PetscFPrintf(topOpt->comm, topOpt->output, "Solve for adjoint "
                    "equation #%i converged in %i iterations with reason: %i\n",
                    i, its, reason); CHKERRQ(ierr);
    }
    else if (topOpt->verbose >= 1) {
      ierr = PetscFPrintf(topOpt->comm, topOpt->output, "Solve for adjoint "
                    "equation #%i failed with reason %i", i, reason); CHKERRQ(ierr);
    }
    ierr = VecGhostUpdateBegin(v_vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(v_vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecResetArray(dKsdU_vec); CHKERRQ(ierr);
    ierr = VecResetArray(v_vec); CHKERRQ(ierr);
  }
  ierr = VecDestroy(&dKsdU_vec); CHKERRQ(ierr);
  ierr = VecDestroy(&v_vec); CHKERRQ(ierr);

  // dlamdrhof
  VectorXPS U_loc(DE);
  MatrixXPS v_loc(DE, nev_conv);
  const PetscScalar *p_dEdz, *p_U;
  ierr = VecGetArrayRead(topOpt->dEdz, &p_dEdz); CHKERRQ(ierr);
  ierr = VecGetArrayRead(topOpt->U, &p_U); CHKERRQ(ierr);
  gradients.setZero();
  for (long el = 0; el < topOpt->nLocElem; el++) {
    // Get local parts of U, phi, and v
    for (int j = 0; j < NE; j++) {
      for (int i = 0; i < DN; i++) {
        PetscInt dof = DN * topOpt->element(el, j) + i;
        U_loc(DN*j+i) = p_U[dof];
        phi_loc.row(DN*j+i) = topOpt->bucklingShape.block(dof, 0, 1, nev_conv);
        v_loc.row(DN*j+i) = v.block(dof, 0, 1, nev_conv);
      }
    }

    // Material stiffness sensitivity
    MatrixXPS dKdy;
    if (topOpt->regular) {
      dKdy = p_dEdz[el] * topOpt->ke[0];
    } else {
      dKdy = p_dEdz[el] * topOpt->ke[el];
    }

    // Throw it all together to get the sensitivity
    Eigen::Map< MatrixXPS > dKs(dKsdy.data() + DE*DE*el, DE, DE);
    VectorXPS pdKp = (phi_loc.cwiseProduct(dKdy*phi_loc)).colwise().sum();
    VectorXPS pdKsp = (phi_loc.cwiseProduct(dKs*phi_loc)).colwise().sum();
    gradients(el) = (lambda.pow(p-1) * (pdKsp - lambda.matrix().cwiseProduct(pdKp) +
                                        v_loc.transpose() * (dKdy*U_loc)).array()).sum();
  }
  ierr = VecRestoreArrayRead(topOpt->dEdz, &p_dEdz); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(topOpt->U, &p_U); CHKERRQ(ierr);
  // Last part of p-norm aggregation
  gradients *= std::pow((PetscScalar)values(0), 1-p);

  /// dlamdrhof*drhofdrho
  if (topOpt->verbose >= 3) {
    ierr = PetscFPrintf(topOpt->comm, topOpt->output,
                        "Assembling filter chain rule to sensitivity\n"); CHKERRQ(ierr);
  }
  Vec dlamdy;
  ierr = VecDuplicate(topOpt->dEdz, &dlamdy); CHKERRQ(ierr);
  ierr = VecPlaceArray(dlamdy, gradients.data()); CHKERRQ(ierr);
  ierr = topOpt->Chain_Filter(NULL, dlamdy); CHKERRQ(ierr);
  ierr = VecResetArray(dlamdy); CHKERRQ(ierr);
  ierr = VecDestroy(&dlamdy); CHKERRQ(ierr);

  if (topOpt->verbose >= 2) {
    ierr = PetscFPrintf(topOpt->comm, topOpt->output, "%1.16g seconds "
                        "and %i iterations for eigenvalues and %1.16g "
                        "seconds and %i iterations for adjoint problems\n",
                        tEigEnd-tEigStart, itEig, tAdjoint, itAdjoint); CHKERRQ(ierr);
  }

  return 0;
}

/********************************************************************
 * Creates the stress stiffness matrix
 * 
 * @param topOpt: The topology optimization object
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode Stability::StressFnc(TopOpt *topOpt)
{
  PetscErrorCode ierr = 0;
  // Mesh characteristics
  const short NE = topOpt->element.cols(), DN = topOpt->numDims, DE = NE*DN;

  // Make sure Ks is zeroed out
  ierr = MatZeroEntries(Ks); CHKERRQ(ierr);

  // Track construction of Ks, dKs
  long dksmarker = 0;

  // Get pointers to Petsc vectors
  const PetscScalar *p_Es, *p_dEsdz, *p_U;
  ierr = VecGetArrayRead(topOpt->Es, &p_Es); CHKERRQ(ierr);
  ierr = VecGetArrayRead(topOpt->dEsdz, &p_dEsdz); CHKERRQ(ierr);
  ierr = VecGetArrayRead(topOpt->U, &p_U); CHKERRQ(ierr);

  MatrixXPS ks = MatrixXPS::Zero(DE, DE);
  Eigen::Map< Eigen::VectorXd > ksVec(ks.data(), ks.size());
  /// Loop over elements
  for (long el = 0; el < topOpt->element.rows(); el++) {
    ks.setZero();
    Eigen::VectorXd u(DE);

    /// Get fem solution for this element
    for (short n = 0; n < NE; n++) {
      for (short d = 0; d < DN; d++) {
        u(d + n*DN) = p_U[DN*topOpt->element(el, n) + d];
      }
    }

    /// Loop over quadrature points
    for (short qp = 0; qp < pow(2, topOpt->numDims); qp++) {
      ks += topOpt->W[qp] * topOpt->GT[qp]
        * sigtos(topOpt->d * topOpt->B[qp] * u)
        * topOpt->G[qp] * topOpt->detJ;
    }

    /// Fill in dKsdy for local elements
    if (el < topOpt->nLocElem) {
      dKsdy.segment(dksmarker, ksVec.size()) = -p_dEsdz[el]*ksVec;
      dksmarker += ksVec.size();
    }

    /// Loop over nodes to fill in KS
    // First get list of global node numbers for this element
    std::vector<PetscInt> cols(NE);
    for (int nd = 0; nd < NE; nd++) // Looping over rows
    cols[nd] = topOpt->gNode(topOpt->element(el,nd));
    // Now construct
    ks *= -p_Es[el];
    for (int nd = 0; nd < NE; nd++) { // Looping over rows
      PetscInt node = topOpt->element(el,nd);
      if (node < topOpt->nLocNode) { // If node is local to this process
        ierr = MatSetValuesBlocked(Ks, 1, topOpt->gNode.data()+node,
        NE, cols.data(), ks.data() + DE*DN*nd, ADD_VALUES);
        CHKERRQ(ierr);
      }
    }
  }

  ierr = MatAssemblyBegin(Ks, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(topOpt->Es, &p_Es); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(topOpt->dEsdz, &p_dEsdz); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(topOpt->U, &p_U); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Ks, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  return 0;
}

/********************************************************************
 * Converts stress in vector form to matrix form
 * 
 * @param sigma: Stress vector
 * 
 * @return s: Stress matrix
 * 
 *******************************************************************/
MatrixXPS Stability::sigtos(VectorXPS sigma)
{
  switch (sigma.size()) {
    case 1: { //1-D
      return sigma;
      break;
    }
    case 3: {
      MatrixXPS s = MatrixXPS::Zero(4 , 4);
      s(0,0) = sigma(0);
      s(1,1) = sigma(1);
      s(0,1) = sigma(2);
      s(1,0) = sigma(2);
      s.block(2, 2, 2, 2) = s.block(0, 0, 2, 2);
      return s;
      break;
    }
    case 6: {
      MatrixXPS s = MatrixXPS::Zero(9 , 9);
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
