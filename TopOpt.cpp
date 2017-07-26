#include <sstream>
#include "TopOpt.h"

using namespace std;

void TopOpt::PrepLog(){
  PetscLogEventRegister("Optimization Update", 0, &UpdateEvent);
  PetscLogEventRegister("Functions", 0, &funcEvent);
  PetscLogEventRegister("FE Analysis", 0, &FEEvent);
  PetscLogEventRegister("JD Compute", 0, &JDCompEvent);
}
