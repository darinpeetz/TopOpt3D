#ifndef FUNCTIONS_H_INCLUDED
#define FUNCTIONS_H_INCLUDED

#include <Eigen/Eigen>
#include <slepceps.h>

typedef Eigen::Matrix<PetscScalar, -1, -1> MatrixXPS;
typedef Eigen::Matrix<PetscScalar, -1, 1>  VectorXPS;

class TopOpt;

enum FUNCTION_TYPE {COMPLIANCE, VOLUME, STABILITY, FREQUENCY};

class Function_Base
{
public:
  Function_Base(std::vector<PetscScalar> &values, PetscScalar min_val,
                PetscScalar max_val, PetscBool objective, PetscBool calc_gradient);
  virtual ~Function_Base() {}

  // Compute weighted and normalized values
  PetscErrorCode Compute(TopOpt *topOpt);
  // Get value and gradient
  PetscScalar Get_Value() {return value;}
  VectorXPS &Get_Gradient() {return gradient;}

  // Boolean indicating constraint or objective
  PetscBool objective;
  // Gradient calculation needed
  PetscBool calc_gradient;

  // Initialize gradient arrays
  PetscErrorCode Initialize_Arrays(PetscInt nElem) {
    gradients.resize(nElem, nvals); gradient.resize(nElem); return 0;
  }
  // Assemble all function values and gradients
  static PetscErrorCode Function_Call(TopOpt *topOpt, double &f, VectorXPS &g,
                                      VectorXPS &dgdx, MatrixXPS &dfdx);
  // Print all values at the end of a run if desired
  static PetscErrorCode Normalization(TopOpt *topOpt);
  // Names of functions
  static const char *name[];
  FUNCTION_TYPE func_type;

protected:
  // Number of values to compute
  PetscInt nvals;
  // Weight of each function value
  VectorXPS weights;
  // Function normalization factors
  PetscScalar min_val, max_val;
  // Internal function values and combined function value
  VectorXPS values;
  PetscScalar value;
  // Internal gradients and combined gradient
  MatrixXPS gradients;
  VectorXPS gradient;

  // Compute internal function values
  virtual PetscErrorCode Function(TopOpt* topOpt) = 0;
};

class Compliance : public Function_Base
{
public:
  Compliance(std::vector<PetscScalar> &values, PetscScalar min_val,
             PetscScalar max_val, PetscBool objective,
             PetscBool calc_gradient=PETSC_TRUE) : 
             Function_Base(values, min_val, max_val, objective,
                           calc_gradient) {func_type = COMPLIANCE;}
  ~Compliance() {}

protected:
  PetscErrorCode Function(TopOpt *topOpt);
};

class Volume : public Function_Base
{
public:
  Volume(std::vector<PetscScalar> &values, PetscScalar min_val,
         PetscScalar max_val, PetscBool objective,
         PetscBool calc_gradient=PETSC_TRUE) :
         Function_Base(values, min_val, max_val, objective,
                       calc_gradient) {func_type = VOLUME;}
  ~Volume() {}

protected:
  PetscErrorCode Function(TopOpt *topOpt);
};

class Stability : public Function_Base
{
public:
  Stability(std::vector<PetscScalar> &values, PetscScalar min_val,
            PetscScalar max_val, PetscBool objective,
            PetscBool calc_gradient=PETSC_TRUE) :
            Function_Base(values, min_val, max_val, objective,
                          calc_gradient) {Ks = NULL; func_type = STABILITY;}
  ~Stability() {MatDestroy(&Ks);}

protected:
  // Stress Stiffness matrix
  Mat Ks;
  // Stress stiffness partial sensitivity
  VectorXPS dKsdy;
  // Element stress stiffness sensitivity wrt local displacement
  MatrixXPS dKsdu;
  // Adjoint vector
  MatrixXPS v;
  // Internal functions
  PetscErrorCode StressFnc(TopOpt *topOpt);
  MatrixXPS sigtos(VectorXPS sigma);
  PetscErrorCode Function(TopOpt *topOpt);
};

class Frequency : public Function_Base
{
public:
  Frequency(std::vector<PetscScalar> &values, PetscScalar min_val,
            PetscScalar max_val, PetscBool objective,
            PetscBool calc_gradient=PETSC_TRUE) :
            Function_Base(values, min_val, max_val, objective,
                          calc_gradient) {M = NULL; func_type = FREQUENCY;}
  ~Frequency() {MatDestroy(&M);}

protected:
  // Mass matrix
  Mat M;
  // Mass matrix partial sensitivity
  VectorXPS dMdy;
  // Internal functions
  PetscErrorCode DiagMassFnc(TopOpt *topOpt);
  PetscErrorCode Function(TopOpt *topOpt);
};

#endif // FUNCTIONS_H_INCLUDED
