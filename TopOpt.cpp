#include <sstream>
#include "TopOpt.h"

using namespace std;

void TopOpt::PrepLog(){
  PetscLogEventRegister("Functions", 0, &funcEvent);
  PetscLogEventRegister("FE Analysis", 0, &FEEvent);
  PetscLogEventRegister("JD Compute", 0, &JDCompEvent);
}
