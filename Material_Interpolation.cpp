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

/*****************************************************/
/**             Material Interpolation              **/
/*****************************************************/
int TopOpt::MatIntFnc( const Eigen::VectorXd &y )
{
  PetscErrorCode ierr = 0;
  if (this->verbose >= 3)
  {
    ierr = PetscFPrintf(comm, output, "Interpolating design variables to material parameters\n"); CHKERRQ(ierr);
  }

  double eps = 1e-10; // Minimum stiffness
  double *p_x, *p_rho, *p_V, *p_E, *p_Es, *p_dVdy, *p_dEdy, *p_dEsdy; // Pointers

  // Apply the filter to design variables
  ierr = VecGetArray(x, &p_x); CHKERRQ(ierr);
  copy(y.data(), y.data()+y.size(), p_x);
  ierr = VecRestoreArray(x, &p_x); CHKERRQ(ierr);
  ierr = MatMult(P, x, this->rho); CHKERRQ(ierr);
  ierr = VecGetArray(this->rho, &p_rho); CHKERRQ(ierr);
  Eigen::Map< ArrayXPS > rho(p_rho, nLocElem);

  // Give the filtered values to PETSc interpolation vectors
  ierr = VecGetArray(this->V, &p_V); CHKERRQ(ierr);
  Eigen::Map< ArrayXPS > V(p_V, nLocElem);

  ierr = VecGetArray(this->E, &p_E); CHKERRQ(ierr);
  Eigen::Map< ArrayXPS > E(p_E, nLocElem);

  ierr = VecGetArray(this->Es, &p_Es); CHKERRQ(ierr);
  Eigen::Map< ArrayXPS > Es(p_Es, nLocElem);

  ierr = VecGetArray(this->dVdy, &p_dVdy); CHKERRQ(ierr);
  Eigen::Map< ArrayXPS > dVdy(p_dVdy, nLocElem);

  ierr = VecGetArray(this->dEdy, &p_dEdy); CHKERRQ(ierr);
  Eigen::Map< ArrayXPS > dEdy(p_dEdy, nLocElem);

  ierr = VecGetArray(this->dEsdy, &p_dEsdy); CHKERRQ(ierr);
  Eigen::Map< ArrayXPS > dEsdy(p_dEsdy, nLocElem);

  switch (interpolation) {
    case SIMP: case SIMP_CUT: case SIMP_SMOOTH: case SIMP_LOGISTIC: {
      // Volume Interpolations
      V = rho;
      dVdy.setOnes();

      // Stiffness Interpolations
      dEsdy = ArrayXPS::Ones(nLocElem);
      double dummyPenal = this->penal;
      while (1.0 <= --dummyPenal)
        dEsdy = dEsdy.cwiseProduct(rho);
      Es = dEsdy.cwiseProduct(rho);
      // At this point dEsdy = z^round(penal-1), Es = z^round(penal)

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
      // At this point Es = z^penal, dEsdy = z^(penal-1)
      // Let go of filtered density
      ierr = VecRestoreArray(this->rho, &p_rho); CHKERRQ(ierr);

      // Finalizing Values and returning PETSc vectors
      dEsdy *= this->penal;
      E = (1-eps)*Es + eps;
      dEdy  = (1-eps)*dEsdy;

      switch (interpolation) {
        case SIMP:
          break;
        case SIMP_CUT: {
          Eigen::Array<bool, -1, 1> cut = V > interp_param[0];
          Es *= cut.cast<PetscScalar>();
          dEsdy *= cut.cast<PetscScalar>();
          break;
        }
        case SIMP_LOGISTIC: {
          ArrayXPS denom = 1 + (interp_param[0]*(interp_param[1]-V)).exp();
          Es /= denom;
          dEsdy = (dEsdy + (Es*interp_param[0])*(1-1/denom))/denom;
          break;
        }
        case SIMP_SMOOTH: {
          ArrayXPS shift = (V-interp_param[0])/(interp_param[1]-interp_param[0]);
          Eigen::Array<bool, -1, 1> cut = (shift >= 0 && shift <= 1);
          ArrayXPS adjust = (6*shift.pow(5) - 15*shift.pow(4) + 10*shift.pow(3)) 
                      * cut.cast<PetscScalar>() + (shift>1).cast<PetscScalar>();
          ArrayXPS dadjust = (30*shift.pow(4) - 60*shift.pow(3) + 30*shift.pow(2))
                    / (interp_param[1]-interp_param[0])*cut.cast<PetscScalar>();
          dEsdy = dEsdy*adjust + Es*dadjust;
          Es = Es*adjust;
          break;
        }
      }

    // Return V
    ierr = VecRestoreArray(this->V, &p_V); CHKERRQ(ierr);
    ierr = VecGhostUpdateBegin(this->V, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(this->V, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    // Return dVdy
    ierr = VecRestoreArray(this->dVdy, &p_dVdy); CHKERRQ(ierr);
    ierr = VecGhostUpdateBegin(this->dVdy, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(this->dVdy, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    // Return dEdy
    ierr = VecRestoreArray(this->dEdy, &p_dEdy); CHKERRQ(ierr);
    ierr = VecGhostUpdateBegin(this->dEdy, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

    // Return dEsdy
    ierr = VecRestoreArray(this->dEsdy, &p_dEsdy); CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(this->dEdy, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecGhostUpdateBegin(this->dEsdy, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

    // Return Es
    ierr = VecRestoreArray(this->Es, &p_Es); CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(this->dEsdy, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecGhostUpdateBegin(this->Es, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

    // Return E
    ierr = VecRestoreArray(this->E, &p_E); CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(this->Es, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecGhostUpdateBegin(this->E, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(this->E, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    break;
  }
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Shouldn't be able to reach this line");
  }
  
  return 0;
}
