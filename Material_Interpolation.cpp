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

/********************************************************************
 *  Apply filters to design variables and interpolate to material parameters
 * 
 * @param design: The design variables
 * 
 * @return ierr: PetscErrorCode
 *******************************************************************/
PetscErrorCode TopOpt::MatIntFnc( const Eigen::VectorXd &design )
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
  {
    ierr = PetscFPrintf(comm, output, "Interpolating design variables to material parameters\n"); CHKERRQ(ierr);
  }

  PetscScalar eps = 0; // Minimum stiffness
  PetscScalar *p_x, *p_y; // Pointers

  // Apply the filter to design variables
  ierr = VecGetArray(x, &p_x); CHKERRQ(ierr);
  copy(design.data(), design.data()+design.size(), p_x);
  ierr = VecRestoreArray(x, &p_x); CHKERRQ(ierr);
  ierr = MatMult(P, x, this->rho); CHKERRQ(ierr);

  // Maximum length scale filtering
  Vec z; // Used as a temporary vector initially
  ierr = VecDuplicate(this->y, &z); CHKERRQ(ierr);
  if (this->vdMin > 0) { // vdMin <= 0 means no max length
    ierr = VecSet(z, 1); CHKERRQ(ierr); // z=1
    ierr = VecPointwiseMin(this->rho, this->rho, z); CHKERRQ(ierr);// Numerically rho can exceed 1, which causes problems
    ierr = VecAXPY(z, -1, this->rho); CHKERRQ(ierr); // z=1-rho
    ierr = VecPow(z, this->vdPenal); CHKERRQ(ierr); // z=(1-rho)^q
    ierr = MatMultAdd(this->R, z, this->REdge, this->y); CHKERRQ(ierr); // y=R*z
    ierr = VecScale(this->y, 1/this->vdMin); CHKERRQ(ierr); //y=R*z/vdmin
  }
  else {
    ierr = VecSet(this->y, 1); CHKERRQ(ierr);
  }
  ierr = VecSet(z, 1); CHKERRQ(ierr); // z=1
  ierr = VecPointwiseMin(this->y, this->y, z); CHKERRQ(ierr); //y=min(R*z/vdmin, 1)

  ierr = VecSet(this->rhoq, 1); CHKERRQ(ierr);
  ierr = VecAXPY(this->rhoq, -1, this->rho); CHKERRQ(ierr);
  ierr = VecPow(this->rhoq, this->vdPenal-1); CHKERRQ(ierr);
  ierr = VecScale(this->rhoq, this->vdPenal / this->vdMin); CHKERRQ(ierr);

  // Setting the actual value of z
  ierr = VecPointwiseMult(z, this->rho, this->y); CHKERRQ(ierr);

  // Volume Interpolations
  ierr = VecCopy(this->rho, this->V); CHKERRQ(ierr);
  ierr = VecSet(this->dVdrho, 1); CHKERRQ(ierr);

  ierr = VecCopy(z, this->dEsdz); CHKERRQ(ierr);
  ierr = VecPow(this->dEsdz, this->penal-1); CHKERRQ(ierr);
  ierr = VecPointwiseMult(this->Es, this->dEsdz, z); CHKERRQ(ierr);
  ierr = VecScale(this->dEsdz, this->penal); CHKERRQ(ierr);

  ierr = VecCopy(this->dEsdz, this->dEdz); CHKERRQ(ierr);
  ierr = VecScale(this->dEdz, 1-eps); CHKERRQ(ierr);
  ierr = VecSet(this->E, eps); CHKERRQ(ierr);
  ierr = VecAXPY(this->E, 1-eps, this->Es); CHKERRQ(ierr);

  switch (interpolation) {
    case SIMP:
      break;
    case SIMP_CUT: {
      Vec tempVec;
      VecDuplicate(this->x, &tempVec);
      ierr = VecGetArray(z, &p_x); CHKERRQ(ierr);
      ierr = VecGetArray(tempVec, &p_y); CHKERRQ(ierr);
      Eigen::Map< ArrayXPS > ZZ(p_x, this->nLocElem);
      Eigen::Map< ArrayXPS > cut(p_y, this->nLocElem);

      cut = (ZZ > interp_param[0]).cast<PetscScalar>();
      
      ierr = VecRestoreArray(z, &p_x); CHKERRQ(ierr);
      ierr = VecRestoreArray(tempVec, &p_y); CHKERRQ(ierr);

      ierr = VecPointwiseMult(this->Es, this->Es, tempVec); CHKERRQ(ierr);
      ierr = VecPointwiseMult(this->dEsdz, this->dEsdz, tempVec); CHKERRQ(ierr);
      ierr = VecDestroy(&tempVec); CHKERRQ(ierr);
      break;
    }
    case SIMP_LOGISTIC: {
      ierr = VecGetArray(z, &p_x); CHKERRQ(ierr);
      Eigen::Map< ArrayXPS > ZZ(p_x, this->nLocElem);
      ArrayXPS denom = 1 + (interp_param[0]*(interp_param[1]-ZZ)).exp();
      ierr = VecRestoreArray(z, &p_x); CHKERRQ(ierr);

      ierr = VecGetArray(this->Es, &p_x); CHKERRQ(ierr);
      Eigen::Map< ArrayXPS > Es(p_x, this->nLocElem); CHKERRQ(ierr);
      Es /= denom;

      ierr = VecGetArray(this->dEsdz, &p_y); CHKERRQ(ierr);
      Eigen::Map< ArrayXPS > dEsdz(p_y, this->nLocElem); CHKERRQ(ierr);
      dEsdz = (dEsdz + (Es*interp_param[0])*(1-1/denom))/denom;

      ierr = VecRestoreArray(this->Es, &p_x); CHKERRQ(ierr);
      ierr = VecRestoreArray(this->dEsdz, &p_y); CHKERRQ(ierr);
      break;
    }
  }

  ierr = VecDestroy(&z); CHKERRQ(ierr);

  // Ghost updates
  ierr = VecGhostUpdateBegin(this->V, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(this->V, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  
  ierr = VecGhostUpdateBegin(this->dVdrho, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(this->dVdrho, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  ierr = VecGhostUpdateBegin(this->dEdz, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(this->dEdz, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  
  ierr = VecGhostUpdateBegin(this->dEsdz, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(this->dEsdz, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  
  ierr = VecGhostUpdateBegin(this->Es, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(this->Es, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  
  ierr = VecGhostUpdateBegin(this->E, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(this->E, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  
  return 0;
}

/********************************************************************
 * Apply chain rule from filtering to sensitivities
 * 
 * @param dfdV: Sensitivity with respect to element volumes
 * @param dfdE: Sensitivity with respect to element sensitivities
 * 
 * @return ierr: PetscErrorCode
 *******************************************************************/
PetscErrorCode TopOpt::Chain_Filter(Vec dfdV, Vec dfdE)
{
  PetscErrorCode ierr = 0;

  Vec temp1, temp2;
  ierr = VecDuplicate(this->x, &temp1); CHKERRQ(ierr);
  ierr = VecDuplicate(this->x, &temp2); CHKERRQ(ierr);

  // Derivatives with respect to volume
  if (dfdV)
  {
    ierr = MatMultTranspose(this->P, dfdV, temp1); CHKERRQ(ierr);
    ierr = VecCopy(temp1, dfdV); CHKERRQ(ierr);
  }

  // Derivatives with respect to stiffness
  if (dfdE)
  {
    PetscScalar *p_Vec;
    ierr = VecPointwiseMult(temp2, dfdE, this->rho); CHKERRQ(ierr);
    ierr = MatMultTranspose(this->R, temp2, temp1); CHKERRQ(ierr);

    ierr = VecCopy(this->y, temp2); CHKERRQ(ierr);
    ierr = VecGetArray(temp2, &p_Vec); CHKERRQ(ierr);
    Eigen::Map< ArrayXPS > y(p_Vec, this->nLocElem); CHKERRQ(ierr);
    y = (y < 1.).cast<PetscScalar>();
    ierr = VecRestoreArray(temp2, &p_Vec); CHKERRQ(ierr);

    ierr = VecPointwiseMult(temp1, temp1, temp2); CHKERRQ(ierr);
    ierr = VecPointwiseMult(temp1, this->rhoq, temp1); CHKERRQ(ierr);
    ierr = VecPointwiseMult(dfdE, dfdE, this->y); CHKERRQ(ierr);
    ierr = VecAYPX(temp1, -1, dfdE); CHKERRQ(ierr);
    ierr = MatMultTranspose(this->P, temp1, dfdE); CHKERRQ(ierr);
  }

  ierr = VecDestroy(&temp1); CHKERRQ(ierr);
  ierr = VecDestroy(&temp2); CHKERRQ(ierr);

  return ierr;
}