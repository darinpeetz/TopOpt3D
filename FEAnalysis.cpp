#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "TopOpt.h"
#include "EigLab.h"

using namespace std;
typedef Eigen::Map< Eigen::RowVectorXd, Eigen::Unaligned,
                    Eigen::InnerStride<-1> > Bmap;
typedef Eigen::Array<PetscScalar, -1, 1> ArrayXPS;

int TopOpt::Initialize ( ) // Set up the stiffness matrix and solver context
{
  PetscErrorCode ierr;
  // Number of nodes and number of dof per element
  Eigen::Array<short, -1, 1> NE(element.rows()), DE(element.rows());
  if (regular)
  {
    NE.setConstant( pow(2, numDims) );
    DE.setConstant( NE(0)*numDims );
    ke.resize(1);
  }
  else
  {
    DE = numDims*NE;
    ke.resize(element.rows());
    // TODO: Assign number of nodes and dof per element for irregular elements
  }

  // Fixing dofs
  fixedDof.resize( supports.cast<short>().sum() );
  int ind = 0;
  for (PetscInt i = 0; i < suppNode.rows(); i++)
  {
    for (short j = 0; j < numDims; j++)
    {
      if (supports(i,j))
        fixedDof(ind++) = numDims*gNode(suppNode(i))+j;
    }
  }
  fixedDof.conservativeResize(ind);
  nFixDof = ind;
  MPI_Allreduce(MPI_IN_PLACE, &nFixDof, 1, MPI_PETSCINT, MPI_SUM, comm);

  // Get free dof
  ArrayXPI AllDof = ArrayXPI::LinSpaced(numDims*nLocNode, numDims*nddist(myid),
                                        numDims*(nddist(myid+1))-1);
  freeDof = AllDof;
  PetscInt *it = set_difference(AllDof.data(), AllDof.data()+AllDof.size(),
              fixedDof.data(), fixedDof.data()+fixedDof.size(), freeDof.data());
  freeDof.conservativeResize(it - freeDof.data());
  nFreeDof = it - freeDof.data();
  MPI_Allreduce(MPI_IN_PLACE, &nFreeDof, 1, MPI_PETSCINT, MPI_SUM, comm);

  // Get dof with no springs or supports
  springlessDof = freeDof;
  springDof.resize(springs.size());
  ind = 0;
  for (PetscInt i = 0; i < springNode.rows(); i++)
  {
    for (short j = 0; j < numDims; j++)
    {
      if (springs(i,j) != 0)
        springDof(ind++) = numDims*gNode(springNode(i))+j;
    }
  }
  springDof.conservativeResize(ind);
  it = set_difference(freeDof.data(), freeDof.data()+freeDof.size(),
      springDof.data(), springDof.data()+springDof.size(), springlessDof.data());
  springlessDof.conservativeResize(it - springlessDof.data());
  nSpringDof = ind;
  MPI_Allreduce(MPI_IN_PLACE, &nSpringDof, 1, MPI_PETSCINT, MPI_SUM, comm);

  // Get Preallocations
  PetscInt nnz = 0; // Total nonzeros from this process
  std::vector<std::vector<PetscInt> > connectivity(nLocNode);
  PetscInt *onDiag  = new PetscInt[nLocNode];
  PetscInt *offDiag = new PetscInt[nLocNode];
  // Loop over all elements (including nonlocal) to determine connectivity of
  // locally-owned nodes
  for (int el = 0; el < element.rows(); el++)
  {
    for (int nd = 0; nd < NE(el); nd++)
    {
      PetscInt node = element(el,nd);
      if (node < nLocNode)
      {
        connectivity[node].insert(connectivity[node].end(),
          element.data()+NE(el)*el, element.data()+NE(el)*(el+1));
      }
    }
  }

  // Remove duplicates from connectivity, and fill in onDiag and offDiag
  for (int nd = 0; nd < nLocNode; nd++)
  {
    // Remove duplicates
    sort(connectivity[nd].begin(), connectivity[nd].end());
    vector<PetscInt>::iterator it = unique(connectivity[nd].begin(),
              connectivity[nd].end());
    int nCols = distance(connectivity[nd].begin(), it);

    // Determine row preallocation
    PetscInt &on = onDiag[nd];
    for (on = 0; on < nCols; on++)
    {
      if (connectivity[nd][on] >= nLocNode)
        break;
    }
    offDiag[nd] = nCols - on;
    nnz += numDims*nCols;
  }

  // Initialize K matrix
  ierr = MatCreate(comm, &this->K); CHKERRQ(ierr);
  ierr = MatSetSizes(this->K, numDims*nLocNode, numDims*nLocNode,
              numDims*nNode, numDims*nNode); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(this->K, "K_"); CHKERRQ(ierr);
  //ierr = MatSetType(this->K, MATMPIAIJ); CHKERRQ(ierr);
  ierr = MatSetFromOptions(this->K); CHKERRQ(ierr);
  //ierr = MatSetBlockSize(this->K, this->numDims); CHKERRQ(ierr);
  ierr = MatXAIJSetPreallocation(this->K, this->numDims, onDiag, offDiag, 0, 0); CHKERRQ(ierr);
  delete[] onDiag; delete[] offDiag;

  // Allocate space for the sparse matrix assembly values
  this->i.clear(); this->i.reserve(nnz);
  this->j.clear(); this->j.reserve(nnz);
  this->k.clear(); this->k.reserve(nnz);
  this->e.clear(); this->e.reserve(nnz);

  // Assemble element stiffness matrices for each element
  if (regular)
    this->ke[0] = LocalK(0);
  else
  {
    this->ke.resize(this->nLocElem);
    for (long el = 0; el < element.rows(); el++)
      this->ke[el] = LocalK(el);
  }

  // Create load vector
  PetscScalar *p_F;
  ierr = VecGetArray(this->F, &p_F); CHKERRQ(ierr);
  for (long i = 0 ; i < loads.rows(); i++)
  {
    for (short j = 0; j < numDims; j++)
      p_F[numDims*loadNode(i)+j] += loads(i,j);
  }
  ierr = VecRestoreArray(this->F, &p_F); CHKERRQ(ierr);
  // Start ghosting force vector
  ierr = VecGhostUpdateBegin(this->F, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  // Construct secondary K corresponding to any springs
  ierr = VecCreateMPI(comm, numDims*nLocNode, numDims*nNode, &spKVec); CHKERRQ(ierr);
  ierr = VecSet(spKVec, 0.0); CHKERRQ(ierr);
  for (int i = 0; i < springNode.rows(); i++)
  {
    ArrayXPI where = ArrayXPI::LinSpaced(numDims, numDims*gNode(springNode(i)),
                                         numDims*(gNode(springNode(i))+1)-1 );
    ierr = VecSetValues(spKVec, numDims, where.data(),
                      springs.data()+numDims*i, ADD_VALUES); CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(spKVec); CHKERRQ(ierr);

  // Construct M vector for lumped masses
  ierr = VecCreateMPI(comm, numDims*nLocNode,
               numDims*nNode, &MLump); CHKERRQ(ierr);
  ierr = VecSet(MLump, 0.0); CHKERRQ(ierr);
  for (int i = 0; i < massNode.rows(); i++)
  {
    ArrayXPI where = ArrayXPI::LinSpaced(numDims, numDims*gNode(massNode(i)),
                                         numDims*(gNode(massNode(i))+1)-1 );
    ierr = VecSetValues(MLump, numDims, where.data(),
                      masses.data()+numDims*i, ADD_VALUES); CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(MLump); CHKERRQ(ierr);

  // Matrix of spring stiffnesses
  /*ierr = MatCreate(Comm, &spK); CHKERRQ(ierr);
  ierr = MatSetSizes(spK, numDims*nLocNode, numDims*nLocNode,
                        numDims*nNode, numDims*nNode); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(spK,"spK_"); CHKERRQ(ierr);
  ierr = MatSetFromOptions(spK); CHKERRQ(ierr);
  ArrayXPI spOnDiag = ArrayXPI::Ones(topOpt->numDims*nLocNode);
  ArrayXPI spOffDiag = ArrayXPI::Zero(topOpt->numDims*nLocNode);
  ierr = MatXAIJSetPreallocation(spK, 1, spOnDiag.data(), spOffDiag.data(), 0, 0); CHKERRQ(ierr);*/

  // Create solver context
  ierr = KSPCreate(comm, &KUF); CHKERRQ(ierr);
  ierr = KSPSetType(KUF, KSPGMRES); CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(this->KUF, PETSC_FALSE); CHKERRQ(ierr);
  ierr = KSPSetTolerances(KUF, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 1e3); CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(KUF, "kuf_"); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(KUF); CHKERRQ(ierr);
  // Set Preconditioner
  PC pc; PCType pctype;
  ierr = KSPGetPC(KUF, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, PCGAMG); CHKERRQ(ierr); // SWITCH FOR GAMG AND MG!!
  ierr = PCSetOptionsPrefix(pc, "kuf_"); CHKERRQ(ierr);
  ierr = PCSetFromOptions(pc); CHKERRQ(ierr);
  // Set up geometric multigrid hierarchy if desired
  ierr = PCGetType(pc, &pctype); CHKERRQ(ierr);
  if (!strcmp(pctype, PCMG))
  {
    PetscInt nlevels = this->PR.size()+1;
    ierr = PCMGSetLevels(pc, nlevels, NULL); CHKERRQ(ierr);
    ierr = PCMGSetType(pc, PC_MG_MULTIPLICATIVE); CHKERRQ(ierr);
    ierr = PCMGSetGalerkin(pc, PETSC_TRUE); CHKERRQ(ierr);
    for (int i = 1; i < nlevels; i++) {
      ierr = PCMGSetInterpolation(pc, i, this->PR[nlevels-i-1]); CHKERRQ(ierr); }
  }

  ierr = VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr); // Finish ghosting force vector
  ierr = VecAssemblyEnd(spKVec); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(MLump); CHKERRQ(ierr);

  return 0;
}
/*****************************************************/
/**              Solve the FEM problem              **/
/*****************************************************/
int TopOpt::FESolve( )
{
  PetscErrorCode ierr;
  // Grab element stiffnesses;
  const PetscScalar *p_E;
  ierr = VecGetArrayRead(this->E, &p_E); CHKERRQ(ierr);

  // Reassemble K
  ierr = MatZeroEntries(this->K); CHKERRQ(ierr);
  PetscInt node;
  PetscInt el = -1;
  Eigen::MatrixXd ke = p_E[0]*this->ke[0];
  std::vector<PetscInt> cols(element.cols());
  for (long nd = 0; nd < element.size(); nd++)
  {
    node = *(element.data() + nd);
    if (node < this->nLocNode)
    {
      if (el != nd/this->element.cols())
      {
        el = nd/this->element.cols();
        for (int j = 0; j < this->element.cols(); j++)
          cols[j] = this->gNode(this->element(el,j));
        if (!regular)
          ke = p_E[el]*this->ke[el];
        else
          ke = p_E[el]*this->ke[0];
      }

      ierr = MatSetValuesBlocked(this->K, 1, this->gNode.data()+node,
        this->element.cols(), cols.data(), ke.data() +
        ke.rows()*this->numDims*(nd % this->element.cols()), ADD_VALUES);
      CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(this->K, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(this->E, &p_E); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(this->K, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  // Apply Spring B.C.'s
  ierr = MatDiagonalSet(this->K, this->spKVec, ADD_VALUES); CHKERRQ(ierr);
  // Apply Dirichlet B.C.'s
  ierr = MatZeroRowsColumns(this->K, fixedDof.size(), fixedDof.data(), 1.0, U, F); CHKERRQ(ierr);
  // Set operators
  ierr = KSPSetOperators(this->KUF, this->K, this->K); CHKERRQ(ierr);

  // Solve
  PC pc; KSPType ksptype; PCType pctype;
  ierr = KSPGetType(KUF, &ksptype); CHKERRQ(ierr);
  ierr = KSPGetPC(KUF, &pc); CHKERRQ(ierr);
  ierr = PCGetType(pc, &pctype); CHKERRQ(ierr);
  if (this->verbose >= 2)
  {
    ierr = PetscPrintf(comm, "Solving governing PDE with %s solver using %s preconditioning\n",
                       ksptype, pctype); CHKERRQ(ierr);
  }
  // Some extra work to do for multigrid preconditioners
  if (!strcmp(pctype,PCGAMG))
  {
    ierr = PCSetCoordinates(pc, this->numDims, this->nLocNode, this->node.data()); CHKERRQ(ierr);
  }
  if (!strcmp(pctype,PCGAMG) || !strcmp(pctype,PCMG))
  {
    ierr = PCSetUp(pc); CHKERRQ(ierr);
    PetscInt levels;
    KSP smooth_ksp; PC smooth_pc; KSPType smooth_ksp_type; PCType smooth_pc_type;
    ierr = PCMGGetLevels(pc, &levels); CHKERRQ(ierr);
    if (!strcmp(pctype,PCMG))
    {
      KSP *sub_ksp; PC sub_pc; PetscInt blocks, first;
      ierr = PCMGGetCoarseSolve(pc, &smooth_ksp); CHKERRQ(ierr);
      ierr = KSPSetType(smooth_ksp, KSPPREONLY); CHKERRQ(ierr);
      ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
      ierr = PCSetType(smooth_pc, PCBJACOBI); CHKERRQ(ierr);
      ierr = PCSetUp(smooth_pc); CHKERRQ(ierr);
      ierr = KSPSetUp(smooth_ksp); CHKERRQ(ierr);
      ierr = PCBJacobiGetSubKSP(smooth_pc, &blocks, &first, &sub_ksp); CHKERRQ(ierr);
      if (blocks != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"blocks on this process, %D, is not one",blocks);
      ierr = KSPGetPC(sub_ksp[0], &sub_pc); CHKERRQ(ierr);
      ierr = PCSetType(sub_pc, PCLU); CHKERRQ(ierr);
      ierr = PCFactorSetShiftType(sub_pc, MAT_SHIFT_INBLOCKS); CHKERRQ(ierr);
      ierr = KSPSetTolerances(sub_ksp[0], PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1); CHKERRQ(ierr);
      ierr = KSPSetType(sub_ksp[0], KSPPREONLY); CHKERRQ(ierr);
    }
    ierr = PCSetUp(pc); CHKERRQ(ierr);

    // Verify that the requested smoothers are being used
    ierr = PCMGGetSmoother(pc, levels-1, &smooth_ksp); CHKERRQ(ierr);
    ierr = KSPGetType(smooth_ksp, &smooth_ksp_type); CHKERRQ(ierr);
    ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
    ierr = PCGetType(smooth_pc, &smooth_pc_type); CHKERRQ(ierr);
    if (strcmp(this->smoother.c_str(), smooth_ksp_type))
    {
      if (!strcmp(this->smoother.c_str(), KSPRICHARDSON))
      {
        for (int i = 1; i < levels; i++)
        {
          ierr = PCMGGetSmoother(pc, i, &smooth_ksp); CHKERRQ(ierr);
          ierr = KSPSetType(smooth_ksp, KSPRICHARDSON); CHKERRQ(ierr);
          ierr = KSPRichardsonSetScale(smooth_ksp, 5.0/10.0); CHKERRQ(ierr);
          ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
          ierr = PCSetType(smooth_pc, PCJACOBI); CHKERRQ(ierr);
        }
      }
      else if (!strcmp(this->smoother.c_str(), KSPCHEBYSHEV))
      {
        for (int i = 1; i < levels; i++)
        {
          ierr = PCMGGetSmoother(pc, i, &smooth_ksp); CHKERRQ(ierr);
          ierr = KSPSetType(smooth_ksp, KSPCHEBYSHEV); CHKERRQ(ierr);
          ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
          ierr = PCSetType(smooth_pc, PCSOR); CHKERRQ(ierr);
        }
      }
      ierr = PCMGGetSmoother(pc, levels-1, &smooth_ksp); CHKERRQ(ierr);
      ierr = KSPGetType(smooth_ksp, &smooth_ksp_type); CHKERRQ(ierr);
      ierr = KSPGetPC(smooth_ksp, &smooth_pc); CHKERRQ(ierr);
      ierr = PCGetType(smooth_pc, &smooth_pc_type); CHKERRQ(ierr);
    }
    if (this->verbose >= 2)
    {
      PetscPrintf(comm, "Multigrid preconditioning is using %s smoothing with %s preconditioning\n",
                  smooth_ksp_type, smooth_pc_type); CHKERRQ(ierr);
    }
  }
  ierr = KSPSolve( this->KUF, this->F, this->U ); CHKERRQ(ierr);

  KSPConvergedReason reason;
  ierr = KSPGetConvergedReason(this->KUF, &reason); CHKERRQ(ierr);
  if (this->verbose >= 1)
  {
    if (reason < 0)
    {
      PetscPrintf(comm, "Solve for displacements failed, reason: %i\n", reason);
    }
    else
    {
      PetscInt its;
      ierr = KSPGetIterationNumber(this->KUF, &its); CHKERRQ(ierr);
      ierr = PetscPrintf(comm, "Solve for displacements converged in %i iterations with reason: %i\n",
                         its, reason); CHKERRQ(ierr);
    }
  }

  ierr = VecGhostUpdateBegin(this->U, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(this->U, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  return 0;
}
/*****************************************************/
/**            Element Stiffness Matrix             **/
/*****************************************************/
Eigen::MatrixXd TopOpt::LocalK ( PetscInt el )
{
  // Nodes per element - this currently only works for rectangular elements
  int NE = pow(2, numDims);
  Eigen::MatrixXd Ke = Eigen::MatrixXd::Zero( numDims * NE , numDims * NE );
  Eigen::MatrixXd dNdxi;
  Eigen::MatrixXd coords( NE , numDims );
  Eigen::ArrayXXd GP = GaussPoints();
  for (int q = 0 ; q < GP.cols() ; q++)
  {
      W[q] = 1;
      dNdxi = dN(GP.data() + q*numDims);
      for (int i = 0 ; i < NE ; i++)
          coords.block(i, 0, 1, numDims) = node.block(element(el, i), 0, 1, numDims);
      Eigen::MatrixXd J = dNdxi * coords;
      Eigen::MatrixXd InvJ = J.inverse();
      detJ = J.determinant();
      Eigen::MatrixXd dNdx = InvJ*dNdxi;
      AssignB(dNdx, B[q]);
      AssignG(dNdx, G[q], GT[q]);
      Ke += W[q] * B[q].transpose() * d * B[q] * detJ;
  }

  return Ke;
}
/*****************************************************/
/**             Material Interpolation              **/
/*****************************************************/
int TopOpt::MatIntFnc( const Eigen::VectorXd &y )
{
  PetscErrorCode ierr;
  double eps = 1e-4; // Minimum stiffness
  double *p_x, *p_rho, *p_V, *p_E, *p_Es, /**p_dVdy,*/ *p_dEdy, *p_dEsdy; // Pointers

  // Feed in the raw density values
  PetscInt low, high;
  ierr = VecGetOwnershipRange(x, &low, &high); CHKERRQ(ierr);
  ierr = VecGetArray(x, &p_x); CHKERRQ(ierr);
  for (long i = 0; i < (high-low); i++)
      p_x[i] = y(i);
  ierr = VecRestoreArray(x, &p_x); CHKERRQ(ierr);

  // Apply the filter
  ierr = MatMult(P, x, this->rho); CHKERRQ(ierr);
  ierr = VecGetArray(this->rho, &p_rho); CHKERRQ(ierr);
  Eigen::Map< ArrayXPS > rho(p_rho, high-low);

  // Give the filtered values to PETSc interpolation vectors
  ierr = VecGetArray(this->V, &p_V); CHKERRQ(ierr);
  Eigen::Map< ArrayXPS > V(p_V, high-low);

  ierr = VecGetArray(this->E, &p_E); CHKERRQ(ierr);
  Eigen::Map< ArrayXPS > E(p_E, high-low);

  ierr = VecGetArray(this->Es, &p_Es); CHKERRQ(ierr);
  Eigen::Map< ArrayXPS > Es(p_Es, high-low);

  /*ierr = VecGetArray(this->dVdy, &p_dVdy); CHKERRQ(ierr);
  Eigen::Map< ArrayXPS > dVdy(p_dVdy, high-low);*/

  ierr = VecGetArray(this->dEdy, &p_dEdy); CHKERRQ(ierr);
  Eigen::Map< ArrayXPS > dEdy(p_dEdy, high-low);

  ierr = VecGetArray(this->dEsdy, &p_dEsdy); CHKERRQ(ierr);
  Eigen::Map< ArrayXPS > dEsdy(p_dEsdy, high-low);

  // Volume Interpolations
  V = rho;
  ierr = VecRestoreArray(this->V, &p_V); CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(this->V, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  //ierr = VecRestoreArray(this->dVdy, &p_dVdy); CHKERRQ(ierr);
  ierr = VecSet(this->dVdy, 1.0); CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(this->V, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  //ierr = VecGhostUpdateBegin(fem->dVdy, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

// Stiffness Interpolations
  dEsdy = ArrayXPS::Ones(high-low);
  double dummyPenal = this->penal;
  while (1.0 <= --dummyPenal)
      dEsdy = dEsdy.cwiseProduct(rho);
  Es = dEsdy.cwiseProduct(rho); //dEsdy = z^round(penal-1), Es = z^round(penal)

  // Square Roots
  short frac = dummyPenal*32768;
  short maxshrt = 16384;
  short nsqrt = 6; // Maximum number of square roots to take to approximate penal
  for (short i = 0; i < nsqrt; i++)
  {
      rho = rho.cwiseSqrt();
      if (frac & maxshrt)
      {
          Es = Es.cwiseProduct(rho);
          dEsdy = dEsdy.cwiseProduct(rho);
      }
      frac<<=1;
      if (!frac)
          break;
  }
  // Es = z^penal, dEsdy = z^(penal-1)
  // Let go of filtered density
  ierr =VecRestoreArray(this->rho, &p_rho); CHKERRQ(ierr);
  //ierr = VecGhostUpdateEnd(this->dVdy, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  // Finalizing Values and returning PETSc vectors
  dEsdy *= this->penal;
  dEdy  = (1-eps)*dEsdy;
  // Return dEdy
  ierr = VecRestoreArray(this->dEdy, &p_dEdy); CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(this->dEdy, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  // Return dEsdy
  ierr = VecRestoreArray(this->dEsdy, &p_dEsdy); CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(this->dEdy, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(this->dEsdy, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  E = (1-eps)*Es;
  // Return Es
  ierr = VecRestoreArray(this->Es, &p_Es); CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(this->dEsdy, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(this->Es, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  E += eps;
  // Return E
  ierr = VecRestoreArray(this->E, &p_E); CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(this->Es, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(this->E, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  ierr = VecGhostUpdateEnd(this->E, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  return 0;
}
/*****************************************************/
/**        Rectangular Element Guass Points         **/
/*****************************************************/
Eigen::ArrayXXd TopOpt::GaussPoints( )
{
  Eigen::ArrayXXd GP(numDims, (int)pow(2, numDims));
  switch (numDims)
  {
    case 1:
      GP(0,0) = -1;   GP(0,1) = 1;
      break;
    case 2:
      GP.col(0) << -1, -1;
      GP.col(1) <<  1, -1;
      GP.col(2) <<  1,  1;
      GP.col(3) << -1,  1;
      break;
    case 3:
      GP.col(0) << -1, -1, -1;
      GP.col(1) <<  1, -1, -1;
      GP.col(2) <<  1,  1, -1;
      GP.col(3) << -1,  1, -1;
      GP.col(4) << -1, -1,  1;
      GP.col(5) <<  1, -1,  1;
      GP.col(6) <<  1,  1,  1;
      GP.col(7) << -1,  1,  1;
      break;
  }
  GP *= 1/sqrt(3);
  return GP;
}
/*****************************************************/
/**       Rectangular Element Shape Functions       **/
/*****************************************************/
Eigen::MatrixXd TopOpt::dN(double *gaussPoint)
{
  // Shape function derivatives in parent coordinates
  Eigen::MatrixXd dNdxi(numDims, (int)pow(2, numDims));
  double xi, eta, zeta;
  switch (numDims)
  {
    case 1:
      dNdxi(0,0) = -1.0/2;
      dNdxi(0,1) =  1.0/2;
      break;
    case 2:
      xi = gaussPoint[0]; eta = gaussPoint[1];
      dNdxi(0,0) = -1.0/4 * (1-eta) ; dNdxi(1,0) = -1.0/4 * (1-xi);
      dNdxi(0,1) =  1.0/4 * (1-eta) ; dNdxi(1,1) = -1.0/4 * (1+xi);
      dNdxi(0,2) =  1.0/4 * (1+eta) ; dNdxi(1,2) =  1.0/4 * (1+xi);
      dNdxi(0,3) = -1.0/4 * (1+eta) ; dNdxi(1,3) =  1.0/4 * (1-xi);
      break;
    case 3:
      xi = gaussPoint[0]; eta = gaussPoint[1]; zeta = gaussPoint[2];
      // N1 = 1/8*(1-xi)*(1-eta)*(1-zeta)
      dNdxi(0,0) = -1.0/8*(1-eta)*(1-zeta);
      dNdxi(1,0) = -1.0/8*(1-xi)*(1-zeta);
      dNdxi(2,0) = -1.0/8*(1-xi)*(1-eta);
      // N2 = 1/8*(1+xi)*(1-eta)*(1-zeta)
      dNdxi(0,1) =  1.0/8*(1-eta)*(1-zeta);
      dNdxi(1,1) = -1.0/8*(1+xi)*(1-zeta);
      dNdxi(2,1) = -1.0/8*(1+xi)*(1-eta);
      // N3 = 1/8*(1+xi)*(1+eta)*(1-zeta)
      dNdxi(0,2) =  1.0/8*(1+eta)*(1-zeta);
      dNdxi(1,2) =  1.0/8*(1+xi)*(1-zeta);
      dNdxi(2,2) = -1.0/8*(1+xi)*(1+eta);
      // N4 = 1/8*(1-xi)*(1+eta)*(1-zeta)
      dNdxi(0,3) = -1.0/8*(1+eta)*(1-zeta);
      dNdxi(1,3) =  1.0/8*(1-xi)*(1-zeta);
      dNdxi(2,3) = -1.0/8*(1-xi)*(1+eta);
      // N5 = 1/8*(1-xi)*(1-eta)*(1+zeta)
      dNdxi(0,4) = -1.0/8*(1-eta)*(1+zeta);
      dNdxi(1,4) = -1.0/8*(1-xi)*(1+zeta);
      dNdxi(2,4) =  1.0/8*(1-xi)*(1-eta);
      // N6 = 1/8*(1+xi)*(1-eta)*(1+zeta)
      dNdxi(0,5) =  1.0/8*(1-eta)*(1+zeta);
      dNdxi(1,5) = -1.0/8*(1+xi)*(1+zeta);
      dNdxi(2,5) =  1.0/8*(1+xi)*(1-eta);
      // N7 = 1/8*(1+xi)*(1+eta)*(1+zeta)
      dNdxi(0,6) =  1.0/8*(1+eta)*(1+zeta);
      dNdxi(1,6) =  1.0/8*(1+xi)*(1+zeta);
      dNdxi(2,6) =  1.0/8*(1+xi)*(1+eta);
      // N8 = 1/8*(1-xi)*(1+eta)*(1+zeta)
      dNdxi(0,7) = -1.0/8*(1+eta)*(1+zeta);
      dNdxi(1,7) =  1.0/8*(1-xi)*(1+zeta);
      dNdxi(2,7) =  1.0/8*(1-xi)*(1+eta);
      break;
  }
  return dNdxi;
}
/*****************************************************/
/**    Construct B matrix for given Gauss Point     **/
/*****************************************************/
void TopOpt::AssignB(Eigen::MatrixXd &dNdx, Eigen::MatrixXd &B)
{
  switch (numDims)
  {
    case 1:
    {
      B.setZero(1, numDims*dNdx.cols());
      B = dNdx;
      break;
    }
    case 2:
    {
      B.setZero(3, numDims*dNdx.cols());
      Eigen::InnerStride<-1> skip(numDims*B.rows());
      Bmap rowInsert(NULL, 0, skip );
      new (&rowInsert)Bmap(B.data(), dNdx.cols(), skip);
      rowInsert = dNdx.row(0);
      new (&rowInsert)Bmap(B.data()+B.rows()+1, dNdx.cols(), skip);
      rowInsert = dNdx.row(1);
      new (&rowInsert)Bmap(B.data()+2, dNdx.cols(), skip);
      rowInsert = dNdx.row(1);
      new (&rowInsert)Bmap(B.data()+B.rows()+2, dNdx.cols(), skip);
      rowInsert = dNdx.row(0);
      break;
    }
    case 3:
    {
      B.setZero(6, numDims*dNdx.cols());
      Eigen::InnerStride<-1> skip(numDims*B.rows());
      Bmap rowInsert(NULL, 0, skip );
      new (&rowInsert)Bmap(B.data(), dNdx.cols(), skip);
      rowInsert = dNdx.row(0);
      new (&rowInsert)Bmap(B.data()+B.rows()+1, dNdx.cols(), skip);
      rowInsert = dNdx.row(1);
      new (&rowInsert)Bmap(B.data()+2*B.rows()+2, dNdx.cols(), skip);
      rowInsert = dNdx.row(2);
      new (&rowInsert)Bmap(B.data()+3, dNdx.cols(), skip);
      rowInsert = dNdx.row(1);
      new (&rowInsert)Bmap(B.data()+B.rows()+3, dNdx.cols(), skip);
      rowInsert = dNdx.row(0);
      new (&rowInsert)Bmap(B.data()+B.rows()+4, dNdx.cols(), skip);
      rowInsert = dNdx.row(2);
      new (&rowInsert)Bmap(B.data()+2*B.rows()+4, dNdx.cols(), skip);
      rowInsert = dNdx.row(1);
      new (&rowInsert)Bmap(B.data()+5, dNdx.cols(), skip);
      rowInsert = dNdx.row(2);
      new (&rowInsert)Bmap(B.data()+2*B.rows()+5, dNdx.cols(), skip);
      rowInsert = dNdx.row(0);
      break;
    }
  }
  return;
}
/*****************************************************/
/**    Construct G matrix for given Gauss Point     **/
/*****************************************************/
void TopOpt::AssignG(Eigen::MatrixXd &dNdx, Eigen::MatrixXd &G,
                     Eigen::MatrixXd &GT)
{
  int numDimsSquare = pow(numDims,2);
  G.setZero(numDimsSquare, numDims*dNdx.cols());
  Eigen::InnerStride<-1> skip(numDims*G.rows());
  Bmap rowInsert(NULL, 0, skip);
  /// Outer iteration
  for (short i = 0; i < numDims; i++)
  {
    /// Inner iteration
    for (short j = 0; j < numDims; j++)
    {
      new (&rowInsert)Bmap(G.data()+(numDimsSquare+numDims)*i + j,
                           dNdx.cols(),skip);
      rowInsert = dNdx.row(j);
    }
  }
  GT = G.transpose();

  return;
}
