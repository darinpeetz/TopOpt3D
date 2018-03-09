#include <Eigen/Eigen>
#include <iostream>
#include <fstream>
#include "EigLab.h"
#include "Functions.h"
#include "Domain.h"
#include "TopOpt.h"
#include <cmath>

using namespace std;
typedef Eigen::Matrix<double, -1, -1, Eigen::RowMajor> MatrixXdRM;

/****************************************************************/
/**         Find next numerical value in string of text        **/
/****************************************************************/
void Find_Next_Digit(const char *line, unsigned short &offset, int length)
{
  offset = max((unsigned short)0,offset);
  while (!isdigit(line[offset]) && line[offset] != '+' &&
        line[offset] != '-' && offset < length)
    offset++;
  return;
}

/****************************************************************/
/**     Get numerical values from a line in the input file     **/
/****************************************************************/
vector<double> Get_Values(string line)
{
  unsigned short offset = 0;
  char *next;
  vector<double> vals;
  while (true)
  {
    Find_Next_Digit(line.c_str(), offset, line.length());
    if (offset == line.length())
      break;
    vals.push_back(strtod(line.c_str()+offset, &next));
    offset = next - line.c_str();
  }

  return vals;
}

/****************************************************************/
/**  Set the boundary conditions using definitions from below  **/
/****************************************************************/
PetscErrorCode TopOpt::Set_BC(Eigen::ArrayXd center, Eigen::ArrayXd radius,
          Eigen::ArrayXXd limits, Eigen::ArrayXd values, BCTYPE TYPE)
{
  PetscErrorCode ierr = 0;
  Eigen::ArrayXd distances = Eigen::ArrayXd::Zero(nLocNode);
  Eigen::Array<bool, -1, 1> valid = Eigen::Array<bool, -1, 1>::Ones(nLocNode);

  // Get distances from center point and check limits in each direction
  for (int i = 0; i < numDims; i++)
  {
    distances += ((node.block(0,i,nLocNode,1).array() - center(i))/radius(i)).square();
    valid = valid && (node.block(0,i,nLocNode,1).array() >= limits(i,0)) &&
            (node.block(0,i,nLocNode,1).array() <= limits(i,1));
  }
  valid = valid && (distances <= 1);
  ArrayXPI newNode = EigLab::find(valid, 1).col(0);

  switch (TYPE)
  {
    case SUPPORT:
      supports.conservativeResize(suppNode.rows()+newNode.rows(), numDims);
      for (int i = 0; i < numDims; i++)
        supports.block(suppNode.rows(), i, newNode.rows(), 1).setConstant(values(i));
      suppNode.conservativeResize(suppNode.rows()+newNode.rows());
      suppNode.segment(suppNode.rows()-newNode.rows(), newNode.rows()) = newNode;
      break;
    case LOAD:
      loads.conservativeResize(loadNode.rows()+newNode.rows(), numDims);
      for (int i = 0; i < numDims; i++)
        loads.block(loadNode.rows(), i, newNode.rows(), 1).setConstant(values(i));
      loadNode.conservativeResize(loadNode.rows()+newNode.rows());
      loadNode.segment(loadNode.rows()-newNode.rows(), newNode.rows()) = newNode;
      break;
    case MASS:
      masses.conservativeResize(massNode.rows()+newNode.rows(), numDims);
      for (int i = 0; i < numDims; i++)
        masses.block(massNode.rows(), i, newNode.rows(), 1).setConstant(values(i));
      massNode.conservativeResize(massNode.rows()+newNode.rows());
      massNode.segment(massNode.rows()-newNode.rows(), newNode.rows()) = newNode;
      break;
    case SPRING:
      springs.conservativeResize(springNode.rows()+newNode.rows(), numDims);
      for (int i = 0; i < numDims; i++)
        springs.block(springNode.rows(), i, newNode.rows(), 1).setConstant(values(i));
      springNode.conservativeResize(springNode.rows()+newNode.rows());
      springNode.segment(springNode.rows()-newNode.rows(), newNode.rows()) = newNode;
      break;
    default:
      PetscPrintf(comm, "Bad BC type given in input file, valid types are");
      PetscPrintf(comm, " Support, Load, Mass, and Spring\n");
      break;
  }
  return ierr;
}

/****************************************************************/
/**       Set various parameters/options from input file       **/
/****************************************************************/
PetscErrorCode TopOpt::Def_Param(MMA *optmma, Eigen::VectorXd &Dimensions,
               ArrayXPI &Nel, double &R, bool &Normalization,
               bool &Reorder_Mesh, PetscInt &mg_levels, PetscInt &min_size)
{
  PetscErrorCode ierr = 0;

  // Variables needed for parsing input file
  ifstream file(filename.c_str());
  if (!file.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Could not find specified input file");
  string line;
  file >> line;
  bool active_section = false;
  unsigned short offset = 0;
  char *next;

  // Parse the file
  while (!file.eof())
  {
    if (!active_section)
    {
      active_section = !line.compare("[Params]");
      getline(file, line);
      file >> line;
    }
    else
    {
      // Convert to all uppercase to avoid captilization errors
      for (string::size_type i = 0; i < line.length(); ++i)
        line[i] = toupper(line[i]);

      // Check which option is being set
      if (!line.compare(0,9,"[/PARAMS]"))
        return ierr;
      else if (!line.compare(0,10,"DIMENSIONS"))
      {
        getline(file, line);
        offset = 0;
        vector<double> temp;
        while (true)
        {
          Find_Next_Digit(line.c_str(), offset, line.length());
          if (offset == line.length())
            break;
          temp.push_back(strtod(line.c_str()+offset, &next));
          offset = next - line.c_str();
        }
        Dimensions = Eigen::Map<Eigen::VectorXd>(temp.data(), temp.size());
      }
      else if (!line.compare(0,3,"NEL"))
      {
        getline(file, line);
        offset = 0;
        vector<PetscInt> temp;
        while (true)
        {
          Find_Next_Digit(line.c_str(), offset, line.length());
          if (offset == line.length())
            break;
          temp.push_back(strtol(line.c_str()+offset, &next, 0));
          offset = next - line.c_str();
        }
        Nel = Eigen::Map<ArrayXPI>(temp.data(), temp.size());
      }
      else if (!line.compare(0,3,"NU0"))
      {
        file >> line;
        Nu0 = strtod(line.c_str(), NULL);
      }
      else if (!line.compare(0,2,"E0"))
      {
        file >> line;
        E0 = strtod(line.c_str(), NULL);
      }
      else if (!line.compare(0,7,"DENSITY"))
      {
        file >> line;
        density = strtod(line.c_str(), NULL);
      }
      else if (!line.compare(0,7,"PENALTY"))
      {
        file >> line;
        pmin = strtod(line.c_str(), NULL);
        file >> line;
        pstep = strtod(line.c_str(), NULL);
        file >> line;
        pmax = strtod(line.c_str(), NULL);
      }
      else if (!line.compare(0,5,"MATER") || !line.compare(0,6,"INTERP"))
      {
        file >> line;
        if (!line.compare(0,4,"SIMP")) {
          string params;
          getline(file, params);
          offset = 0;
          while (true)
          {
            Find_Next_Digit(params.c_str(), offset, params.length());
            if (offset == params.length())
              break;
            interp_param.push_back(strtod(params.c_str()+offset, &next));
            offset = next - params.c_str();
          }
          if (!line.compare(5,3,"CUT")) {
            if (interp_param.size() != 1) {
              SETERRQ(comm, PETSC_ERR_ARG_WRONG, "SIMP_CUT needs 1 parameter");}
            interpolation = SIMP_CUT; }
          else if (!line.compare(5,8,"LOGISTIC")) {
            if (interp_param.size() != 2) {
            SETERRQ(comm, PETSC_ERR_ARG_WRONG, "SIMP_LOGISTIC needs 2 parameters");}
            interpolation = SIMP_LOGISTIC; }
          else if (!line.compare(5,6,"SMOOTH")) {
            if (interp_param.size() != 2) {
              SETERRQ(comm, PETSC_ERR_ARG_WRONG, "SIMP_SMOOTH needs 2 parameters");}
            interpolation = SIMP_SMOOTH; }
          else {
            interpolation = SIMP; }
        }
      }
      else if (!line.compare(0,7,"RFACTOR"))
      {
        file >> line;
        R = strtod(line.c_str(), NULL)*(Dimensions(1)-Dimensions(0))/Nel(0);
      }
      else if (!line.compare(0,1,"R"))
      {
        file >> line;
        R = strtod(line.c_str(), NULL);
      }
      else if (!line.compare(0,14,"MIN_ITERATIONS"))
      {
        file >> line;
        optmma->Set_Iter_Limit_Min(strtol(line.c_str(), NULL, 0));
      }
      else if (!line.compare(0,14,"MAX_ITERATIONS"))
      {
        file >> line;
        optmma->Set_Iter_Limit_Max(strtol(line.c_str(), NULL, 0));
      }
      else if (!line.compare(0,9,"KKT_LIMIT"))
      {
        file >> line;
        optmma->Set_KKT_Limit(strtod(line.c_str(), NULL));
      }
      else if (!line.compare(0,12,"CHANGE_LIMIT"))
      {
        file >> line;
        optmma->Set_Change_Limit(strtod(line.c_str(), NULL));
      }
      else if (!line.compare(0,10,"STEP_LIMIT"))
      {
        file >> line;
        optmma->Set_Step_Limit(strtod(line.c_str(), NULL));
      }
      else if (!line.compare(0,13,"NORMALIZATION"))
      {
        Normalization = true;
      }
      else if (!line.compare(0,18,"NO_MESH_REORDERING"))
      {
        Reorder_Mesh = false;
      }
      else if (!line.compare(0,9,"MG_LEVELS"))
      {
        file >> line;
        mg_levels = strtol(line.c_str(), NULL, 0);
      }
      else if (!line.compare(0,15,"MG_MIN_MAT_SIZE"))
      {
        file >> line;
        min_size = strtol(line.c_str(), NULL, 0);
      }
      else if (!line.compare(0,14,"MG_COARSE_SIZE"))
      {
        file >> line;
        int c_size = strtol(line.c_str(), NULL, 0);
        double temp = log2(Nel.size());
        for (int i = 0; i < Nel.size(); i++)
          temp += log2(Nel(i)+1);
        temp -= log2(c_size);
        temp /= Nel.size();
        mg_levels = ceil(temp)+1;
      }
      else if (!line.compare(0,8,"SMOOTHER"))
      {
        file >> this->smoother;
      }
      else if (!line.compare(0,7,"VERBOSE"))
      {
        file >> line;
        this->verbose = strtol(line.c_str(), NULL, 0);
      }
      else if (!line.compare(0,6,"FOLDER") || !line.compare(0,7,"RESTART"))
      {
        file.ignore();
        getline(file, this->folder);
      }
      else if (!line.compare(0,5,"PRINT_EVERY"))
      {
        file >> line;
        print_every = strtol(line.c_str(), NULL, 0);
      }

      file >> line;
    }
  }

  return ierr;
}

/****************************************************************/
/**                Get options from command line               **/
/****************************************************************/
PetscErrorCode TopOpt::Get_CL_Options()
{
  PetscErrorCode ierr = 0;
  ierr = PetscOptionsGetInt(NULL, NULL, "-Verbose", &verbose, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-Print_Every", &print_every, NULL);
         CHKERRQ(ierr);
  return ierr;
}

/****************************************************************/
/**                Set the optimization functions              **/
/****************************************************************/
PetscErrorCode TopOpt::Set_Funcs()
{
  PetscErrorCode ierr = 0;
  // Variables needed for parsing input file
  ifstream file(filename.c_str());
  if (!file.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Could not find specified input file");
  string line;
  file >> line;
  bool active_section = false;

  // Parse the file
  while (!file.eof())
  {
    if (!active_section)
    {
      active_section = !line.compare("[Functions]");
      getline(file, line);
      file >> line;
    }
    else
    {
      FUNCTION_TYPE func;
      if (!line.compare("[/Functions]"))
        return ierr;
      else if (!line.compare(0,10,"Compliance"))
        func = COMPLIANCE;
      else if (!line.compare(0,9,"Perimeter"))
        func = PERIMETER;
      else if (!line.compare(0,6,"Volume"))
        func = VOLUME;
      else if (!line.compare(0,9,"Stability"))
        func = STABILITY;
      else if (!line.compare(0,7,"Dynamic") || !line.compare(0,9,"Frequency"))
        func = FREQUENCY;
      else
        SETERRQ1(comm, PETSC_ERR_SUP, "Unknown function type specified: %s", line.c_str());
      file >> line;

      // Function details
      double min = 0, max = 0;
      vector<PetscScalar> values;
      PetscBool objective = PETSC_TRUE;

      while (true)
      {
        if (!line.compare(0,9,"Objective"))
        {
          objective = PETSC_TRUE;
          file >> line;
          continue;
        }
        else if (!line.compare(0,10,"Constraint"))
        {
          objective = PETSC_FALSE;
          file >> line;
          continue;
        }
        else if (!line.compare(0,6,"Values"))
        {
          file >> line;
          if (!isdigit(line.c_str()[0]))
            SETERRQ(comm, PETSC_ERR_ARG_NULL, "Need to specify function weight/constraint");
          while (isdigit(line.c_str()[0]) || !line.compare(0,1,"-"))
          {
            values.push_back(strtod(line.c_str(), NULL));
            file >> line;
          }
          continue;
        }
        else if (!line.compare(0,5,"Range"))
        {
          file >> line;
          min = strtod(line.c_str(), NULL);
          file >> line;
          max = strtod(line.c_str(), NULL);
          file >> line;
          continue;
        }
        else if (!line.compare(0,3,"Nev"))
        {
          // To prevent errors from old input files
          getline(file, line);
          continue;
        }  
        break;
      }

      switch(func){
        case COMPLIANCE :
          function_list.push_back(new Compliance(values, min, max, objective));
          needK = PETSC_TRUE; needU = PETSC_TRUE;
          break;
        case PERIMETER :
          function_list.push_back(new Perimeter(values, min, max, objective));
          break;
        case VOLUME :
          function_list.push_back(new Volume(values, min, max, objective));
          break;
        case STABILITY :
          function_list.push_back(new Stability(values, min, max, objective));
          needK = PETSC_TRUE; needU = PETSC_TRUE;
          bucklingShape.resize(1, values.size());
          break;
        case FREQUENCY :
          function_list.push_back(new Frequency(values, min, max, objective));
          needK = PETSC_TRUE;
          dynamicShape.resize(1, values.size());
          break;
      }
    }
  }
  return ierr;
}

/****************************************************************/
/**             Check if elements should be removed            **/
/****************************************************************/
PetscErrorCode TopOpt::Domain(Eigen::ArrayXXd &Points, const Eigen::VectorXd &Box,
            Eigen::Array<bool, -1, 1> &elemValidity)
{
  PetscErrorCode ierr = 0;
  elemValidity.setOnes(Points.rows());

  return ierr;
}

/****************************************************************/
/**                 Define boundary conditions                 **/
/****************************************************************/
PetscErrorCode TopOpt::Def_BC()
{
  PetscErrorCode ierr = 0;
  // Variables needed for parsing input file
  ifstream file(filename.c_str());
  if (!file.is_open())
    SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Could not find specified input file");
  string line;
  file >> line;
  bool active_section = false;

  // Parse the file
  BCTYPE TYPE;
  while (!file.eof())
  {
    if (!active_section)
    {
      active_section = !line.compare("[BC]");
      getline(file, line);
      file >> line;
    }
    else
    {
      Eigen::ArrayXd center, radius, values;
      Eigen::ArrayXXd limits;
      TYPE = OTHER;
      if (!line.compare("[/BC]"))
        return ierr;
      else if (!line.compare(0,7,"Support"))
        TYPE = SUPPORT;
      else if (!line.compare(0,4,"Load"))
        TYPE = LOAD;
      else if (!line.compare(0,4,"Mass"))
        TYPE = MASS;
      else if (!line.compare(0,6,"Spring"))
        TYPE = SPRING;
      getline(file, line);

      while (true)
      {
        file >> line;
        if (!line.compare(0,6,"Center"))
        {
          getline(file, line);
          vector<double> temp = Get_Values(line);
          if (temp.size() != (unsigned short)numDims)
            cout << "Centers for BC " << TYPE << " are specified incorrectly\n";
          center = Eigen::Map<Eigen::ArrayXd>(temp.data(), temp.size());
          continue;
        }
        if (!line.compare(0,6,"Radius"))
        {
          getline(file, line);
          vector<double> temp = Get_Values(line);
          if (temp.size() != (unsigned short)numDims)
            cout << "Radii for BC " << TYPE << " are specified incorrectly\n";
          radius = Eigen::Map<Eigen::ArrayXd>(temp.data(), temp.size());
          continue;
        }
        if (!line.compare(0,6,"Limits"))
        {
          getline(file, line);
          vector<double> temp = Get_Values(line);
          if (temp.size()/2 != (unsigned short)numDims)
            cout << "Limits for BC " << TYPE << " are specified incorrectly\n";
          limits = Eigen::Map<Eigen::ArrayXXd>(temp.data(), 2, temp.size()/2);
          limits.transposeInPlace();
          continue;
        }
        if (!line.compare(0,6,"Values"))
        {
          getline(file, line);
          vector<double> temp = Get_Values(line);
          if (temp.size() != (unsigned short)numDims)
            cout << "Values for BC " << TYPE << " are specified incorrectly\n";
          values = Eigen::Map<Eigen::ArrayXd>(temp.data(), temp.size());
          continue;
        }
        break;
      }

      ierr = Set_BC(center, radius, limits, values, TYPE); CHKERRQ(ierr);
    }
  }
  return 0;
}
