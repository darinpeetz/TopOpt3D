#include "TopOpt.h"

using namespace std;

void TopOpt::PrepLog(){
  PetscLogEventRegister("Optimization Update", 0, &UpdateEvent);
  PetscLogEventRegister("Functions", 0, &funcEvent);
  PetscLogEventRegister("FE Analysis", 0, &FEEvent);
}

void TopOpt::Clear()
{ 
  delete[] B; delete[] G; delete[] GT; delete[] W;
  VecDestroy(&F); VecDestroy(&U); /*MatDestroy(&spK);*/ VecDestroy(&spKVec);
  MatDestroy(&K); VecDestroy(&MLump); KSPDestroy(&KUF);
  /*KSPDestroy(&dynamicKSP); KSPDestroy(&bucklingKSP);*/ MatDestroy(&P);
  VecDestroy(&V); VecDestroy(&dVdy); VecDestroy(&E); VecDestroy(&dEdy);
  VecDestroy(&Es); VecDestroy(&dEsdy); VecDestroy(&x); VecDestroy(&rho);
  for (unsigned int i = 0; i < function_list.size(); i++)
    delete function_list[i];
}
