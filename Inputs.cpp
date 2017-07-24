#include "mpi.h"
#include <Eigen/Eigen>
#include <iostream>
#include <fstream>
#include "EigLab.h"
#include "Inputs.h"
#include "Functions.h"
#include "Domain.h"
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

void TopOpt::Set_BC(Eigen::ArrayXd center, Eigen::ArrayXd radius,
          Eigen::ArrayXXd limits, Eigen::ArrayXd values, BCTYPE TYPE)
{
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
  return;
}

void TopOpt::Def_Param(MMA *optmma, TopOpt *topOpt, Eigen::VectorXd &Dimensions,
                       ArrayXPI &Nel, double &R, bool &Normalization,
                       bool &Reorder_Mesh, int &mg_levels)
{
  topOpt->smoother = "chebyshev";
  topOpt->verbose = 1;
  // Variables needed for parsing input file
  ifstream file(filename.c_str());
  if (!file.is_open())
  {
    PetscPrintf(comm, "Could not find specified input file\n");
    MPI_Abort(comm, 404);
  }
  string line;
  file >> line;
  bool active_section = false;
  unsigned short offset = 0;
  char *next;

  // Parse the file
  while (!file.eof())
  {
    sleep(1);
    if (!active_section)
    {
      active_section = !line.compare("[Params]");
      getline(file, line);
      file >> line;
    }
    else
    {
      if (!line.compare(0,9,"[/Params]"))
        return;
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
      else if (!line.compare(0,8,"Smoother"))
      {
        file >> topOpt->smoother;
      }
      else if (!line.compare(0,7,"Verbose"))
      {
        file >> line;
        topOpt->verbose = strtol(line.c_str(), NULL, 0);
      }

      file >> line;
    }
  }

  return;
}

void TopOpt::Set_Funcs()
{
  // Assume no functions initially
  Comp = 0; Perim = 0; Vol = 0; Stab = 0; Dyn = 0;
  Stab_optnev = 0; Stab_nev = 0; Dyn_optnev = 0; Dyn_nev = 0;

  // Variables needed for parsing input file
  ifstream file(filename.c_str());
  if (!file.is_open())
  {
    PetscPrintf(comm, "Could not find specified input file\n");
    MPI_Abort(comm, 404);
  }
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
      // Aliases for function details
      double *min = NULL, *max = NULL;
      vector<double> *value = NULL;
      short *func = NULL, *nev = NULL, *optnev = NULL;

      if (!line.compare("[/Functions]"))
        return;
      else if (!line.compare(0,10,"Compliance"))
      {
        func = &Comp;
        value = &Comp_val;
        min = &Comp_min;
        max = &Comp_max;
      }
      else if (!line.compare(0,9,"Perimeter"))
      {
        func = &Perim;
        value = &Perim_val;
        min = &Perim_min;
        max = &Perim_max;
      }
      else if (!line.compare(0,6,"Volume"))
      {
        func = &Vol;
        value = &Vol_val;
        min = &Vol_min;
        max = &Vol_max;
      }
      else if (!line.compare(0,9,"Stability"))
      {
        func = &Stab;
        value = &Stab_val;
        min = &Stab_min;
        max = &Stab_max;
        optnev = &Stab_optnev;
        nev = &Stab_nev;
      }
      else if (!line.compare(0,7,"Dynamic") || !line.compare(0,9,"Frequency"))
      {
        func = &Dyn;
        value = &Dyn_val;
        min = &Dyn_min;
        max = &Dyn_max;
        optnev = &Dyn_optnev;
        nev = &Dyn_nev;
      }
      else
      {
        if (myid == 0)
          cout << "Unknown function type specified in Input file\n";
        MPI_Abort(comm, 404);
      }

      file >> line;
      while (true)
      {
        if (!line.compare(0,9,"Objective"))
        {
          *func = 1;
          file >> line;
          continue;
        }
        else if (!line.compare(0,10,"Constraint"))
        {
          *func = 2;
          file >> line;
          continue;
        }
        else if (!line.compare(0,6,"Values"))
        {
          file >> line;
          while (isdigit(line.c_str()[0]) || !line.compare(0,1,"-"))
          {
            value->push_back(strtod(line.c_str(), NULL));
            file >> line;
          }
          if (optnev != NULL)
            *optnev = value->size();
          continue;
        }
        else if (!line.compare(0,5,"Range"))
        {
          file >> line;
          *min = strtod(line.c_str(), NULL);
          file >> line;
          *max = strtod(line.c_str(), NULL);
          file >> line;
          continue;
        }
        else if (!line.compare(0,3,"Nev"))
        {
          file >> line;
          *nev = strtol(line.c_str(), NULL, 0);
          file >> line;
          continue;
        }
        break;
      }
    }
  }
  return;
}

void TopOpt::Domain(Eigen::ArrayXXd &Points, const Eigen::VectorXd &Box,
            Eigen::Array<bool, -1, 1> &elemValidity)
{
  elemValidity.setOnes(Points.rows());

  return;
}

void TopOpt::Def_BC()
{
  // Variables needed for parsing input file
  ifstream file(filename.c_str());
  if (!file.is_open())
  {
    PetscPrintf(comm, "Could not find specified input file\n");
    MPI_Abort(comm, 404);
  }
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
        return;
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

      Set_BC(center, radius, limits, values, TYPE);
    }
  }
  return;
}
