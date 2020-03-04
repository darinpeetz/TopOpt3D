#include <numeric>
#include <math.h>
#include "Functions.h"
#include "TopOpt.h"
#include "EigLab.h"
#include "JDMG.h"
#include "LOPGMRES.h"
#include <unsupported/Eigen/KroneckerProduct>

using namespace std;

PetscErrorCode Frequency::Function( TopOpt *topOpt )
{
  PetscErrorCode ierr = 0;
  short NE = topOpt->element.cols(), DN = topOpt->numDims, DE = NE*DN;

  if (topOpt->verbose >= 3)
  {
    ierr = PetscFPrintf(topOpt->comm, topOpt->output, "Performing dynamic analysis\n"); CHKERRQ(ierr);
  }

  /// Assemble Mass matrix and get sensitivity information
  if (dMdy.size() == 0)
  {
    dMdy.resize( topOpt->nLocElem*(long)pow(DE,2) );
    // Initialize M
    ierr = MatCreate(topOpt->comm, &M); CHKERRQ(ierr);
    ierr = MatSetSizes(M, topOpt->numDims*topOpt->nLocNode, topOpt->numDims*topOpt->nLocNode,
          topOpt->numDims*topOpt->nNode, topOpt->numDims*topOpt->nNode); CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(M,"M_"); CHKERRQ(ierr);
    ierr = MatSetFromOptions(M); CHKERRQ(ierr);
    ArrayXPI onDiag = ArrayXPI::Ones(topOpt->nLocNode);
    ArrayXPI offDiag = ArrayXPI::Zero(topOpt->nLocNode);
    ierr = MatXAIJSetPreallocation(M, topOpt->numDims, onDiag.data(), offDiag.data(), 0, 0); CHKERRQ(ierr);
  }  

  ierr = DiagMassFnc( topOpt ); CHKERRQ(ierr);
  /// Remove fixed and spring dof from M (and K if necessary)
  ierr = MatZeroRowsColumns(M, topOpt->fixedDof.size(),
             topOpt->fixedDof.data(), 1e-8, NULL, NULL); CHKERRQ(ierr);
  if (topOpt->nEigFixDof > 0) { // Fix additional parts of matrices if requested
    ierr = MatZeroRowsColumns(M, topOpt->eigenFixedDof.size(),
            topOpt->eigenFixedDof.data(), 0.0, NULL, NULL); CHKERRQ(ierr);
    ierr = MatZeroRowsColumns(topOpt->K, topOpt->eigenFixedDof.size(),
            topOpt->eigenFixedDof.data(), 1.0, NULL, NULL); CHKERRQ(ierr);
    ierr = MatSetNullSpace(topOpt->K, NULL); CHKERRQ(ierr);
    ierr = KSPSetOperators(topOpt->KUF, topOpt->K, topOpt->K); CHKERRQ(ierr);
    ierr = KSPSetUp(topOpt->KUF); CHKERRQ(ierr);
  }

  // Create LOPGMRES instance
  LOPGMRES lopgmres(topOpt->comm);
  lopgmres.Set_Verbose(topOpt->verbose);
  lopgmres.Set_File(topOpt->output);
  
  // Get restrictors from FEM problem
  PC pc;
  ierr = KSPGetPC(topOpt->KUF, &pc); CHKERRQ(ierr);
  ierr = lopgmres.Set_PC(pc); CHKERRQ(ierr);

  // Set Operators
  lopgmres.Set_Operators(M, topOpt->K);
  // Set target eigenvalues
  Nev_Type target_type = UNIQUE_LAST_NEV;
  lopgmres.Set_Target(LR, nvals, target_type);
  lopgmres.Set_Tol(std::pow(10, std::log10(2*topOpt->nNode)/2-9));
  lopgmres.Set_MaxIt(3*(nvals+1)*50*(PetscInt)std::log(topOpt->nElem));
  ierr = lopgmres.Compute(); CHKERRQ(ierr);

  // Get the results
  PetscInt nev_conv = lopgmres.Get_nev_conv();
  if (nev_conv == 0)
  {
    char name_suffix[30];
    sprintf(name_suffix, "_eigen_failure");
    ierr = topOpt->PrintVals(name_suffix); CHKERRQ(ierr);
    SETERRQ(topOpt->comm, PETSC_ERR_CONV_FAILED, "Eigensolver found 0 eigenvalues\n");
  }
  VectorXPS lambda(nev_conv);
  lopgmres.Get_Eigenvalues(lambda.data());

  // Number of converged eigenvalues to use for optimization
  nev_conv -= (target_type == TOTAL_NEV) ? 0 : 1;

  // Mertge any duplicate eigenvalues
  for (short j = 0; j < nvals-1; j++)
    values(j) = lambda[j];
  for (short j = nvals-1; j < nev_conv; j++)
    values(nvals-1) = lambda[j];
  values(nvals-1) /= nev_conv-nvals+1;

  // Return if sensitivities aren't needed
  if (calc_gradient == PETSC_FALSE)
    return 0;

  // Make sure we have enough room for all eigenvectors
  Vec *phi, phi_copy;
  lopgmres.Get_Eigenvectors(&phi);
  topOpt->dynamicShape.resize(topOpt->dynamicShape.rows(), nev_conv);
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
  MatrixXPS phim( (DE*DE)*topOpt->nLocElem, nev_conv );
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
  const PetscScalar *p_dEdz;
  ierr = VecGetArrayRead(topOpt->dEdz, &p_dEdz); CHKERRQ(ierr);
  Eigen::Map< const Eigen::VectorXd > dEdz(p_dEdz, topOpt->nLocElem);
  MatrixXPS dKdy;
  if (topOpt->regular)
  {
    Eigen::Map< Eigen::VectorXd > ke(topOpt->ke[0].data(), DE*DE);
    dKdy = Eigen::kroneckerProduct(dEdz, ke);
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
    dKdy.block(ind, 0, ke.size(), 1) = dEdz(el)*ke;
    }
  }
  ierr = VecRestoreArrayRead(topOpt->dEdz, &p_dEdz); CHKERRQ(ierr);

  /// Construct sensitivity
  MatrixXPS df = MatrixXPS::Zero((DE*DE)*topOpt->nLocElem,nvals);
  for (short j = 0; j < nvals-1; j++)
    df.col(j) += phim.col(j).cwiseProduct(dMdy-lambda[j]*dKdy);
  for (short j = nvals-1; j < nev_conv; j++)
    df.col(nvals-1) += phim.col(j).cwiseProduct(dMdy-lambda[j]*dKdy);
  df.col(nvals-1) /= nev_conv-nvals+1;

  for (long el = 0; el < topOpt->nLocElem; el++)
    gradients.row(el) = df.block(el*(DE*DE), 0, (DE*DE), nvals).colwise().sum();

  /// dCdrhof*drhofdrho
  Vec dlamdy;
  ierr = VecDuplicate( topOpt->dEdz, &dlamdy ); CHKERRQ(ierr);
  for (short i = 0; i < nvals; i++)
  {
    ierr = VecPlaceArray( dlamdy, gradients.data()+i*gradients.rows() ); CHKERRQ(ierr);
    ierr = topOpt->Chain_Filter( NULL, dlamdy ); CHKERRQ(ierr);
    ierr = VecResetArray(dlamdy); CHKERRQ(ierr);
  }
  ierr = VecDestroy( &dlamdy ); CHKERRQ(ierr);

  return 0;
}

/********************************************************************/
/**                Creates the diagonal mass matrix                **/
/********************************************************************/
PetscErrorCode Frequency::DiagMassFnc( TopOpt *topOpt )
{
  PetscErrorCode ierr = 0;

  // Mesh characteristics
  const short NE = topOpt->element.cols(), DN = topOpt->numDims, DE = NE*DN;

  // Make sure M is zeroed out
  ierr = MatZeroEntries(M); CHKERRQ(ierr);

  // Track construction of Ks, dKs
  long dMmarker = 0;

  // Get pointers to Petsc vectors
  const PetscScalar *p_V, *p_dVdrho;
  ierr = VecGetArrayRead(topOpt->V, &p_V); CHKERRQ(ierr);
  ierr = VecGetArrayRead(topOpt->dVdrho, &p_dVdrho); CHKERRQ(ierr);

  MatrixPS mMat = 1.0/pow(2,topOpt->numDims)/topOpt->numDims*
      topOpt->elemSize(0)*topOpt->density*MatrixXPS::Identity(DE, DE);
  Eigen::Map< VectorXPS > mVec(mMat.data(), mMat.size());
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
    dMdy.segment(dMmarker, mVec.size()) = p_dVdrho[el] * mVec;
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
  ierr = VecRestoreArrayRead(topOpt->dVdrho, &p_dVdrho); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatDiagonalSet(M, topOpt->MLump, ADD_VALUES); CHKERRQ(ierr);

  return 0;
}
