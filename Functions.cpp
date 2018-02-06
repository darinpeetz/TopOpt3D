#include "Functions.h"
#include "TopOpt.h"

using namespace std;
const char *Function_Base::name[] = {"Compliance", "Perimeter", "Volume", "Stability", "Frequency"};

/******************************************************************************/
/**                       Function_Base constructor                          **/
/******************************************************************************/
Function_Base::Function_Base(std::vector<PetscScalar> &values, PetscScalar min_val, PetscScalar max_val, PetscBool objective, PetscBool calc_gradient)
{
  nvals = values.size();
  this->values = VectorXPS::Zero(nvals);
  weights = VectorXPS::Zero(nvals);
  copy(values.begin(), values.end(), weights.data());
  this->min_val = min_val;
  this->max_val = max_val;
  this->objective = objective;
  this->calc_gradient = calc_gradient;
}

/******************************************************************************/
/**                  Get value and gradient of a function                    **/
/******************************************************************************/
PetscErrorCode Function_Base::Compute(TopOpt *topOpt)
{
  PetscErrorCode ierr = 0;
  if (topOpt->verbose >= 2)
  {
    ierr = PetscFPrintf(topOpt->comm, topOpt->output, "Calling %s function\n",
                        name[this->func_type]); CHKERRQ(ierr);
  }
  ierr = Function(topOpt); CHKERRQ(ierr);

  // Calculate combined function value and gradient
  if (topOpt->verbose >= 3)
  {
    ierr = PetscFPrintf(topOpt->comm, topOpt->output,
                        "Tabulating results of %s function\n",
                         name[this->func_type]); CHKERRQ(ierr);
  }
  value = 0; gradient.setZero();
  for (PetscInt i = 0; i < weights.size(); i++)
  {
    value += (objective==PETSC_TRUE?weights(i):1)*(values(i)-(objective==PETSC_TRUE?min_val:weights(i)));
    gradient += (objective==PETSC_TRUE?weights(i):1)*gradients.col(i);
  }
  value /= max_val-min_val;
  gradient /= max_val-min_val;

  return ierr;
}

/******************************************************************************/
/**                 Assemble function values and gradients                   **/
/******************************************************************************/
PetscErrorCode Function_Base::Function_Call(TopOpt *topOpt, double &f, Eigen::VectorXd &dfdx, Eigen::VectorXd &g, Eigen::MatrixXd &dgdx)
{
  PetscErrorCode ierr = 0;

  int constraint = 0;
  f = 0; dfdx.setZero();
  for (unsigned int ii = 0; ii < topOpt->function_list.size(); ii++)
  {
    ierr = topOpt->function_list[ii]->Compute(topOpt); CHKERRQ(ierr);
    if (topOpt->function_list[ii]->objective == PETSC_TRUE)
    {
      f += topOpt->function_list[ii]->Get_Value();
      dfdx += topOpt->function_list[ii]->Get_Gradient();
    }
    else
    {
      g(constraint) = topOpt->function_list[ii]->Get_Value();
      dgdx.col(constraint) = topOpt->function_list[ii]->Get_Gradient();
      constraint++;
    }
  }

  return ierr;
}
 
/******************************************************************************/
/**                  Print out values at the end of a run                    **/
/******************************************************************************/
PetscErrorCode Function_Base::Normalization(TopOpt *topOpt)
{
  PetscErrorCode ierr = 0;

  std::vector<PetscScalar> values; values.push_back(1);
  PetscScalar min = 0, max = 1;
  PetscBool objective = PETSC_TRUE, gradient = PETSC_FALSE;
  Function_Base *function;

  function = new Compliance(values, min, max, objective, gradient);
  ierr = function->Function(topOpt); CHKERRQ(ierr);
  ierr = PetscFPrintf(topOpt->comm, topOpt->output, "\tCompliance:\t%1.16g\n",
             function->values(0)); CHKERRQ(ierr);
  delete function;
  function = new Perimeter(values, min, max, objective, gradient);
  ierr = function->Function(topOpt); CHKERRQ(ierr);
  ierr = PetscFPrintf(topOpt->comm, topOpt->output, "\tPerimeter:\t%1.16g\n",
             function->values(0)); CHKERRQ(ierr);
  delete function;
  function = new Volume(values, min, max, objective, gradient);
  ierr = function->Function(topOpt); CHKERRQ(ierr);
  ierr = PetscFPrintf(topOpt->comm, topOpt->output, "\tVolume:\t%1.16g\n",
             function->values(0)); CHKERRQ(ierr);
  delete function;
  function = new Stability(values, min, max, objective, gradient);
  ierr = function->Function(topOpt); CHKERRQ(ierr);
  ierr = PetscFPrintf(topOpt->comm, topOpt->output, "\tStability:\t%1.16g\n",
             function->values(0)); CHKERRQ(ierr);
  delete function;
  function = new Frequency(values, min, max, objective, gradient);
  ierr = function->Function(topOpt); CHKERRQ(ierr);
  ierr = PetscFPrintf(topOpt->comm, topOpt->output, "\tFrequency:\t%1.16g\n",
             function->values(0)); CHKERRQ(ierr);
  delete function;

  return ierr;
}
 
/******************************************************************************/
/**                       Apply filter for chain rule                        **/
/******************************************************************************/
PetscErrorCode Function_Base::Chain_Filter(Mat P, Vec x)
{
  PetscErrorCode ierr = 0;

  Vec temp;
  ierr = VecDuplicate(x, &temp); CHKERRQ(ierr);
  ierr = MatMultTranspose(P, x, temp); CHKERRQ(ierr);
  ierr = VecCopy(temp, x); CHKERRQ(ierr);
  ierr = VecDestroy(&temp); CHKERRQ(ierr);

  return 0;
}
