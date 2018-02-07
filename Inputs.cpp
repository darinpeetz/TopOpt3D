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

void Find_Next_Digit(const char *line, unsigned short &offset, int length)
{
  offset = max((unsigned short)0,offset);
  while (!isdigit(line[offset]) && line[offset] != '+' &&
        line[offset] != '-' && offset < length)
    offset++;
  return;
}

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

PetscErrorCode TopOpt::Def_Param(MMA *optmma, TopOpt *topOpt, Eigen::VectorXd &Dimensions,
                       ArrayXPI &Nel, double &R, bool &Normalization,
                       bool &Reorder_Mesh, PetscInt &mg_levels, PetscInt &min_size)
{
  PetscErrorCode ierr = 0;
  topOpt->smoother = "chebyshev";
  topOpt->verbose = 1;
  topOpt->folder = "";
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
      if (!line.compare(0,9,"[/Params]"))
      {
        ierr = PetscOptionsGetInt(NULL, NULL, "-Verbose", &verbose, NULL);
               CHKERRQ(ierr);
        return ierr;
      }
      else if (!line.compare(0,10,"Dimensions"))
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
      else if (!line.compare(0,3,"Nu0"))
      {
        file >> line;
        Nu0 = strtod(line.c_str(), NULL);
      }
      else if (!line.compare(0,2,"E0"))
      {
        file >> line;
        E0 = strtod(line.c_str(), NULL);
      }
      else if (!line.compare(0,7,"Density"))
      {
        file >> line;
        density = strtod(line.c_str(), NULL);
      }
      else if (!line.compare(0,7,"Penalty"))
      {
        file >> line;
        pmin = strtod(line.c_str(), NULL);
        file >> line;
        pstep = strtod(line.c_str(), NULL);
        file >> line;
        pmax = strtod(line.c_str(), NULL);
      }
      else if (!line.compare(0,7,"RFactor"))
      {
        file >> line;
        R = strtod(line.c_str(), NULL)*(Dimensions(1)-Dimensions(0))/Nel(0);
      }
      else if (!line.compare(0,1,"R"))
      {
        file >> line;
        R = strtod(line.c_str(), NULL);
      }
      else if (!line.compare(0,14,"Min_Iterations"))
      {
        file >> line;
        optmma->Set_Iter_Limit_Min(strtol(line.c_str(), NULL, 0));
      }
      else if (!line.compare(0,14,"Max_Iterations"))
      {
        file >> line;
        optmma->Set_Iter_Limit_Max(strtol(line.c_str(), NULL, 0));
      }
      else if (!line.compare(0,9,"KKT_Limit"))
      {
        file >> line;
        optmma->Set_KKT_Limit(strtod(line.c_str(), NULL));
      }
      else if (!line.compare(0,12,"Change_Limit"))
      {
        file >> line;
        optmma->Set_Change_Limit(strtod(line.c_str(), NULL));
      }
      else if (!line.compare(0,10,"Step_Limit"))
      {
        file >> line;
        optmma->Set_Step_Limit(strtod(line.c_str(), NULL));
      }
      else if (!line.compare(0,13,"Normalization"))
      {
        Normalization = true;
      }
      else if (!line.compare(0,18,"No_Mesh_Reordering"))
      {
        Reorder_Mesh = false;
      }
      else if (!line.compare(0,9,"MG_Levels"))
      {
        file >> line;
        mg_levels = strtol(line.c_str(), NULL, 0);
      }
      else if (!line.compare(0,15,"MG_Min_Mat_Size"))
      {
        file >> line;
        min_size = strtol(line.c_str(), NULL, 0);
      }
      else if (!line.compare(0,14,"MG_Coarse_Size"))
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
      else if (!line.compare(0,8,"Smoother"))
      {
        file >> topOpt->smoother;
      }
      else if (!line.compare(0,7,"Verbose"))
      {
        file >> line;
        topOpt->verbose = strtol(line.c_str(), NULL, 0);
      }
      else if (!line.compare(0,6,"Folder") || !line.compare(0,7,"Restart"))
      {
        file.ignore();
        getline(file, topOpt->folder);
      }

      file >> line;
    }
  }

  return ierr;
}

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

PetscErrorCode TopOpt::Domain(Eigen::ArrayXXd &Points, const Eigen::VectorXd &Box,
            Eigen::Array<bool, -1, 1> &elemValidity)
{
  PetscErrorCode ierr = 0;
  elemValidity.setOnes(Points.rows());

  return ierr;
}

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
