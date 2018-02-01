#include <numeric>
#include "Functions.h"
#include "TopOpt.h"
#include "EigLab.h"
#include "JDMG.h"
#include "LOPGMRES.h"
#include <unsupported/Eigen/KroneckerProduct>

using namespace std;

PetscErrorCode Stability::Function( TopOpt *topOpt)
{
  PetscErrorCode ierr = 0;
  short NE = topOpt->element.cols(), DN = topOpt->numDims, DE = NE*DN;

  /// Assemble stress stiffness matrix and get sensitivity information
  if (dKsdy.size() == 0)
  {
    dKsdy.resize( topOpt->nLocElem*(long)pow(DE,2) );
    ierr = MatDuplicate( topOpt->K, MAT_SHARE_NONZERO_PATTERN, &Ks ); CHKERRQ(ierr);
  }
  ierr = StressFnc( topOpt ); CHKERRQ(ierr);

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

Mat Ks2;
ierr = MatDuplicate(Ks, MAT_SHARE_NONZERO_PATTERN, &Ks2); CHKERRQ(ierr);
ierr = MatCopy(Ks, Ks2, SAME_NONZERO_PATTERN); CHKERRQ(ierr);
ierr = MatScale(Ks2, -1); CHKERRQ(ierr);

PetscReal tol = pow(10,log10(2*topOpt->nNode)/2-5);
PetscInt MaxIt = (nvals+1)*50*(PetscInt)log(topOpt->nElem);
VectorXPS l_JDMG, l_LOBPCG, l_LOPGMRES;

FILE *timings;
ierr = PetscFOpen(topOpt->comm, "Timings", "a", &timings); CHKERRQ(ierr);
double t0 = MPI_Wtime();
  LOPGMRES lopgmres(topOpt->comm);
  lopgmres.Set_Verbose(topOpt->verbose);
  lopgmres.Set_File(topOpt->output);
  // Get restrictors from FEM problem
  PC pcmg; PCType pctype;
  ierr = KSPGetPC(topOpt->KUF, &pcmg); CHKERRQ(ierr);
  ierr = PCGetType(pcmg, &pctype); CHKERRQ(ierr);
  /*if (!strcmp(pctype,PCMG))
  {*/
    ierr = lopgmres.Set_Hierarchy(topOpt->PR); CHKERRQ(ierr);
  /*}
  else if (!strcmp(pctype,PCGAMG))
  {
    ierr = lopgmres.PCMG_Extract(pcmg); CHKERRQ(ierr);
  }
  else
     SETERRQ1(topOpt->comm, PETSC_ERR_ARG_WRONG, "Preconditioner of type %s was provided, but must be one of mg or gamg", pctype);*/
  // Set Operators
  lopgmres.Set_Operators(Ks, topOpt->K);
  // Set target eigenvalues
  Nev_Type target_type = UNIQUE_LAST_NEV;
  lopgmres.Set_Target(LR, nvals, target_type);
  lopgmres.Set_MaxIt(MaxIt);
  lopgmres.Set_Cycle(FMGCycle);
  lopgmres.Set_Tol(tol);
  // Compute the eigenvalues
  ierr = lopgmres.Compute(); CHKERRQ(ierr);
  PetscInt nev_conv = lopgmres.Get_nev_conv();
  PetscInt its = lopgmres.Get_Iterations();
  l_LOPGMRES.resize(nev_conv);
  lopgmres.Get_Eigenvalues(l_LOPGMRES.data());

ierr = PetscFPrintf(topOpt->comm, timings, "LOPGMRES took %1.6g seconds and %i iterations to find %i eigenvalues\n", MPI_Wtime()-t0, its, nev_conv); CHKERRQ(ierr);
t0 = MPI_Wtime();

{
  EPS eps; ST st; KSP ksp; PC pc, JUNK;
  ierr = EPSCreate(topOpt->comm, &eps); CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps, EPS_GHEP); CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL); CHKERRQ(ierr);
  ierr = EPSSetTolerances(eps, tol, MaxIt); CHKERRQ(ierr);
  // Setting up krylov ST structure
  ierr = EPSGetST(eps, &st); CHKERRQ(ierr);
  ierr = STGetKSP(st, &ksp); CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, PCMG); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, topOpt->K, topOpt->K); CHKERRQ(ierr);
  PetscInt nlevels = topOpt->PR.size()+1;
  ierr = PCMGSetLevels(pc, nlevels, NULL); CHKERRQ(ierr);
  ierr = PCMGSetType(pc, PC_MG_FULL); CHKERRQ(ierr);
  ierr = PCMGSetGalerkin(pc, PETSC_TRUE); CHKERRQ(ierr);
  for (int i = 1; i < nlevels; i++) {
  ierr = PCMGSetInterpolation(pc, i, topOpt->PR[nlevels-i-1]); CHKERRQ(ierr); }
  // Use direct solve on coarse level
  ierr = PCSetUp(pc); CHKERRQ(ierr);
  KSP smooth_ksp, *sub_ksp; PC smooth_pc, sub_pc; PetscInt blocks, first;
  ierr = PCMGGetCoarseSolve(pc, &smooth_ksp); CHKERRQ(ierr);
  ierr = KSPSetType(smooth_ksp, KSPPREONLY); CHKERRQ(ierr);
  ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
  ierr = PCSetType(smooth_pc, PCBJACOBI); CHKERRQ(ierr);
  ierr = PCSetUp(smooth_pc); CHKERRQ(ierr);
  ierr = KSPSetUp(smooth_ksp); CHKERRQ(ierr);
  ierr = PCBJacobiGetSubKSP(smooth_pc, &blocks, &first, &sub_ksp); CHKERRQ(ierr);
  ierr = KSPGetPC(sub_ksp[0], &sub_pc); CHKERRQ(ierr);
  ierr = PCSetType(sub_pc, PCLU); CHKERRQ(ierr);
  ierr = PCFactorSetShiftType(sub_pc, MAT_SHIFT_INBLOCKS); CHKERRQ(ierr);
  ierr = KSPSetTolerances(sub_ksp[0], PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1); CHKERRQ(ierr);
  ierr = KSPSetType(sub_ksp[0], KSPPREONLY); CHKERRQ(ierr);
  ierr = PCSetUp(pc); CHKERRQ(ierr);
  // Use Jacobi smoothing
  for (int i = 1; i < nlevels; i++)
  {
    ierr = PCMGGetSmoother(pc, i, &smooth_ksp); CHKERRQ(ierr);
    ierr = KSPSetType(smooth_ksp, KSPRICHARDSON); CHKERRQ(ierr);
    ierr = KSPRichardsonSetScale(smooth_ksp, 5.0/10.0); CHKERRQ(ierr);
    ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
    ierr = PCSetType(smooth_pc, PCJACOBI); CHKERRQ(ierr);
  }

  ierr = EPSSetDimensions(eps, nev_conv, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = EPSSetType(eps, EPSLOBPCG); CHKERRQ(ierr);
  ierr = EPSSetOperators(eps, Ks2, topOpt->K); CHKERRQ(ierr);
  ierr = EPSSetOptionsPrefix(eps, "LOBPCG_"); CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
  ierr = EPSSolve(eps); CHKERRQ(ierr);
  ierr = EPSGetConverged(eps, &nev_conv); CHKERRQ(ierr);

  l_LOBPCG.resize(nev_conv);
  for (int i = 0; i < nev_conv; i++){
  ierr = EPSGetEigenvalue(eps, i, l_LOBPCG.data()+i, NULL); CHKERRQ(ierr); }

  ierr = EPSGetIterationNumber(eps, &its); CHKERRQ(ierr);
  ierr = PCCreate(topOpt->comm, &JUNK); CHKERRQ(ierr);
  ierr = KSPSetPC(ksp, JUNK); CHKERRQ(ierr);
  ierr = PCDestroy(&JUNK); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(topOpt->K, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(topOpt->K, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = EPSDestroy(&eps); CHKERRQ(ierr);
}

ierr = PetscFPrintf(topOpt->comm, timings, "LOBPCG took %1.6g seconds and %i iterations to find %i eigenvalues\n", MPI_Wtime()-t0, its, nev_conv); CHKERRQ(ierr);
t0 = MPI_Wtime();

  /// Create JDMG instance
  JDMG jdmg(topOpt->comm);
  jdmg.Set_Verbose(topOpt->verbose);
  jdmg.Set_File(topOpt->output);
  // Get restrictors from FEM problem
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
  jdmg.Set_Target(LR, nvals, target_type);
  jdmg.Set_MaxIt(MaxIt);
  jdmg.Set_Cycle(FMGCycle);
  jdmg.Set_Tol(tol);
  // Compute the eigenvalues
  ierr = jdmg.Compute(); CHKERRQ(ierr);

  // Get the results
  its = jdmg.Get_Iterations();
  nev_conv = jdmg.Get_nev_conv();
  VectorXPS lambda(nev_conv);
  l_JDMG.resize(nev_conv);
  jdmg.Get_Eigenvalues(l_JDMG.data());
  lambda = l_JDMG;

ierr = PetscFPrintf(topOpt->comm, timings, "JDMG took %1.6g seconds and %i iterations to find %i eigenvalues\n\n", MPI_Wtime()-t0, its, nev_conv); CHKERRQ(ierr);

l_LOBPCG *= -1;
if ( (l_LOBPCG.size() != l_LOPGMRES.size()) || (l_LOBPCG.size() != l_JDMG.size()) || (l_LOBPCG-l_LOPGMRES).norm()/l_LOPGMRES.norm() > 1e-3 || (l_LOBPCG-l_JDMG).norm()/l_JDMG.norm() > 1e-3 )
{
  ierr = PetscFPrintf(topOpt->comm, timings, "LOPGMRES:\tLOBPCG:\tJDMG:\n"); CHKERRQ(ierr);
  for (int ii = 0; ii < max(max(l_LOPGMRES.size(), l_LOBPCG.size()), l_JDMG.size()); ii++)
  {
  if (ii < l_LOPGMRES.size())
    PetscFPrintf(topOpt->comm, timings, "%17.16g\t", l_LOPGMRES(ii));
  else
    PetscFPrintf(topOpt->comm, timings, "0\t");
  if (ii < l_LOBPCG.size())
    PetscFPrintf(topOpt->comm, timings, "%17.16g\t", l_LOBPCG(ii));
  else
    PetscFPrintf(topOpt->comm, timings, "0\t");
  if (ii < l_JDMG.size())
    PetscFPrintf(topOpt->comm, timings, "%17.16g\n", l_JDMG(ii));
  else
    PetscFPrintf(topOpt->comm, timings, "0\n");
  }
  PetscFPrintf(topOpt->comm, timings, "\n");
  ierr = PetscFClose(topOpt->comm, timings); CHKERRQ(ierr);
  MPI_Barrier(topOpt->comm);
  SETERRQ(topOpt->comm, PETSC_ERR_PLIB, "Different eigenvalues generated by some methods");
}
ierr = PetscFClose(topOpt->comm, timings); CHKERRQ(ierr);

  nev_conv -= (target_type == TOTAL_NEV) ? 0 : 1;

  for (short j = 0; j < nvals-1; j++)
    values(j) = lambda[j];
  for (short j = nvals-1; j < nev_conv; j++)
    values(nvals-1) = lambda[j];
  values(nvals-1) /= nev_conv-nvals+1;

  // Return if sensitivities aren't needed
  if (calc_gradient == PETSC_FALSE)
    return 0;

  Vec *phi, phi_copy;
  jdmg.Get_Eigenvectors(&phi);

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
  MatrixXPS phim( (DE*DE)*topOpt->gElem.rows(), nev_conv );
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
  if (this->dKsdu.size() == 0)
  {
    short dDE = DE*DE;
    this->dKsdu.setZero(dDE , DE);
    // Loop over dof of a single element
    for (int dof = 0; dof < DE; dof++)
    {
    Eigen::Map< MatrixXPS > dksdu( this->dKsdu.data() + dDE*dof, DE, DE );
    Eigen::VectorXd du = Eigen::VectorXd::Zero(DE);
    du(dof) = 1;
    // Loop through quadrature points
    for (int qp = 0; qp < 4; qp++)
      dksdu += topOpt->W[qp]*topOpt->GT[qp]*sigtos(topOpt->d*topOpt->B[qp]*du)
        *  topOpt->G[qp]*topOpt->detJ;
    }
  }

  /// Construct adjoint vectors to be solved
  MatrixXPS dKsdU = MatrixXPS::Zero(topOpt->node.size(), nev_conv);
  v.resize(dKsdU.rows(), dKsdU.cols());
  const PetscScalar *p_Es;
  ierr = VecGetArrayRead(topOpt->Es, &p_Es); CHKERRQ(ierr);
  for (PetscInt el = 0; el < topOpt->element.rows(); el++)
  {
    MatrixXPS dKs = p_Es[el] * this->dKsdu;
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

  MatrixXPS vm( (DE*DE)*topOpt->nLocElem, nev_conv );
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
  MatrixXPS dKdy;
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
  MatrixXPS df = MatrixXPS::Zero(dKdy.rows(),nvals);
  for (short j = 0; j < nvals-1; j++)
  {
    df.col(j) += phim.block(0,j,dKdy.rows(),1).cwiseProduct(dKsdy-lambda[j]*dKdy)
        + vm.col(j).cwiseProduct(dKdy.cwiseProduct(Um));
  }
  for (short j = nvals-1; j < nev_conv; j++)
  {
    df.col(nvals-1) += phim.block(0,j,dKdy.rows(),1).cwiseProduct(dKsdy-lambda[j]*dKdy)
        + vm.col(j).cwiseProduct(dKdy.cwiseProduct(Um));
  }
  df.col(nvals-1) /= nev_conv-nvals+1;
  
  for (long el = 0; el < topOpt->nLocElem; el++)
    gradients.row(el) = df.block(el*(DE*DE), 0, (DE*DE), nvals).colwise().sum();

  /// dCdrhof*drhofdrho
  Vec dlamdy;
  ierr = VecDuplicate( topOpt->dEdy, &dlamdy ); CHKERRQ(ierr);
  for (short i = 0; i < nvals; i++)
  {
    ierr = VecPlaceArray( dlamdy, gradients.data()+i*gradients.rows() ); CHKERRQ(ierr);
    ierr = Chain_Filter( topOpt->P, dlamdy ); CHKERRQ(ierr);
    ierr = VecResetArray( dlamdy ); CHKERRQ(ierr);
  }
  ierr = VecDestroy( &dlamdy ); CHKERRQ(ierr);

  if (nev_conv < nvals)
  {
    PetscFPrintf(topOpt->comm, topOpt->output, "***************************************************\n");
    PetscFPrintf(topOpt->comm, topOpt->output, "Warning, nev_conv < nevals\n");
    PetscFPrintf(topOpt->comm, topOpt->output, "***************************************************\n");
  }

  return 0;
}

PetscErrorCode Stability::StressFnc( TopOpt *topOpt )
{
  PetscErrorCode ierr = 0;
  // Mesh characteristics
  const short NE = topOpt->element.cols(), DN = topOpt->numDims, DE = NE*DN;
  // Track construction of Ks, dKs
  long dksmarker = 0;

  // Get pointers to Petsc vectors
  const PetscScalar *p_Es, *p_dEsdy, *p_U;
  ierr = VecGetArrayRead(topOpt->Es, &p_Es); CHKERRQ(ierr);
  ierr = VecGetArrayRead(topOpt->dEsdy, &p_dEsdy); CHKERRQ(ierr);
  ierr = VecGetArrayRead(topOpt->U, &p_U); CHKERRQ(ierr);

  MatrixXPS ks = MatrixXPS::Zero(DE, DE);
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

MatrixXPS Stability::sigtos(VectorXPS sigma)
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
    MatrixXPS s = MatrixXPS::Zero(4 , 4);
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
