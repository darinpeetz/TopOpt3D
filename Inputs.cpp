#include <Eigen/Eigen>
#include <iostream>
#include <fstream>
#include "EigLab.h"
#include "Functions.h"
#include "Domain.h"
#include "TopOpt.h"
#include <cmath>

using namespace std;

/********************************************************************
 * Find next numerical value in string of text
 * 
 * @param line: the string of text
 * @param offset: the first character in the string to look at
 * @param length: length of the string of text
 * 
 *******************************************************************/
void Find_Next_Digit(const char *line, unsigned short &offset, PetscInt length)
{
  offset = max((unsigned short)0,offset);
  while (!isdigit(line[offset]) && line[offset] != '+' &&
        line[offset] != '-' && offset < length)
    offset++;
  return;
}

/********************************************************************
 * Get numerical values from a line in the input file
 * 
 * @param line: the string of text
 * 
 * @return vals: a vector of all the numeric values
 * 
 *******************************************************************/
vector<PetscScalar> Get_Values(string line)
{
  unsigned short offset = 0;
  char *next;
  vector<PetscScalar> vals;
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


/********************************************************************
 * Get all local node indices describing all the elements faces
 * 
 * @param numDims: Number of dimensions to the problem (1D, 2D, or 3D)
 * 
 * @return faces: Array with face indices across each row
 * 
 *******************************************************************/
ArrayXXPI Get_Faces(PetscInt numDims)
{
  switch (numDims) {
    case 1: {
      return ArrayXXPI::Zero(0, 0);
    }
    case 2: {
      ArrayXXPI faces = ArrayXXPI(4, 2);
      faces(0, 0) = 0; faces(0, 1) = 1;
      faces(1, 0) = 1; faces(1, 1) = 2;
      faces(2, 0) = 2; faces(2, 1) = 3;
      faces(3, 0) = 3; faces(3, 1) = 0;
      return faces;
    }
    case 3: {
      ArrayXXPI faces = ArrayXXPI(6, 4);
      faces(0, 0) = 0; faces(0, 1) = 1; faces(0, 2) = 2; faces(0, 3) = 3;
      faces(1, 0) = 4; faces(1, 1) = 5; faces(1, 2) = 6; faces(1, 3) = 7;
      faces(2, 0) = 0; faces(2, 1) = 1; faces(2, 2) = 5; faces(2, 3) = 4;
      faces(3, 0) = 3; faces(3, 1) = 2; faces(3, 2) = 6; faces(3, 3) = 7;
      faces(4, 0) = 0; faces(4, 1) = 4; faces(4, 2) = 7; faces(4, 3) = 3;
      faces(5, 0) = 1; faces(5, 1) = 5; faces(5, 2) = 6; faces(5, 3) = 2;
      return faces;
    }
    default: {
      std::cout << "Bad dimension of problem provided to Get_Faces\n";
      return ArrayXXPI::Zero(0, 0);
    }
  }
}

/********************************************************************
 * Set the boundary conditions using definitions from below
 * 
 * @param center: center of the region where BC are applied
 * @param radius: radius of the region where BC are applied
 * @param limits: max extents of the region where BC are applied
 * @param values: values of the BC to apply
 * @param TYPE: The type of BC being applied (fixed, load, spring, etc.)
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode TopOpt::Set_BC(ArrayXPS center, ArrayXPS radius,
          ArrayXXPS limits, ArrayXPS values, BCTYPE TYPE)
{
  PetscErrorCode ierr = 0;
  ArrayXPS distances = ArrayXPS::Zero(nLocNode);
  Eigen::Array<bool, -1, 1> valid = Eigen::Array<bool, -1, 1>::Ones(nLocNode);
  ArrayXPI newNode, newFace;
  ArrayXXPI faces = Get_Faces(this->numDims);

  if (TYPE != PRESSURE) {
    // Get distances from center point and check limits in each direction
    for (PetscInt i = 0; i < numDims; i++) {
      distances += ((node.block(0,i,nLocNode,1).array() - center(i))/radius(i)).square();
      valid = valid && (node.block(0,i,nLocNode,1).array() >= limits(i,0)) &&
              (node.block(0,i,nLocNode,1).array() <= limits(i,1));
    }
    valid = valid && (distances <= 1);
    newNode = EigLab::find(valid, 1).col(0);
  }
  else {
    // Get face centers
    ArrayXXPS centers = ArrayXXPS::Zero(faces.rows()*element.rows(), this->numDims);
    for (PetscInt el = 0; el < element.rows(); el++) {
      for (PetscInt face = 0; face < faces.rows(); face++) {
        for (PetscInt nd = 0; nd < faces.cols(); nd++) {
          centers.row(el*faces.rows() + face) += node.row(element(el, faces(face, nd))).array();
        }
      }
    }
    centers /= faces.rows();

    // Get distances from center point and check limits in each direction
    for (PetscInt i = 0; i < numDims; i++) {
      distances += ((centers - center(i))/radius(i)).square();
      valid = valid && (centers >= limits(i,0)) && (centers <= limits(i,1));
    }
    valid = valid && (distances <= 1);
    newFace = EigLab::find(valid, 1).col(0);
  }

  switch (TYPE) {
    case SUPPORT: {
      supports.conservativeResize(suppNode.rows()+newNode.rows(), numDims);
      for (PetscInt i = 0; i < numDims; i++)
        supports.block(suppNode.rows(), i, newNode.rows(), 1).setConstant(values(i));
      suppNode.conservativeResize(suppNode.rows()+newNode.rows());
      suppNode.segment(suppNode.rows()-newNode.rows(), newNode.rows()) = newNode;
      break;
    }
    case EIGEN: {
      eigenSupports.conservativeResize(eigenSuppNode.rows()+newNode.rows(), numDims);
      for (PetscInt i = 0; i < numDims; i++)
        eigenSupports.block(eigenSuppNode.rows(), i, newNode.rows(), 1).setConstant(values(i));
      eigenSuppNode.conservativeResize(eigenSuppNode.rows()+newNode.rows());
      eigenSuppNode.segment(eigenSuppNode.rows()-newNode.rows(), newNode.rows()) = newNode;
      break;
    }
    case LOAD: {
      loads.conservativeResize(loadNode.rows()+newNode.rows(), numDims);
      for (PetscInt i = 0; i < numDims; i++)
        loads.block(loadNode.rows(), i, newNode.rows(), 1) = values(i);
      loadNode.conservativeResize(loadNode.rows()+newNode.rows());
      loadNode.segment(loadNode.rows()-newNode.rows(), newNode.rows()) = newNode;
      break;
    }
    case PRESSURE: {
      // Start from end of current list
      PetscInt ind = loadNode.rows();
      // Allocate enough for every node identified to be local
      loads.conservativeResize(loadNode.rows()+faces.size(), numDims);
      loadNode.conservativeResize(loadNode.rows()+faces.size());
      // Loop over every face selected
      for (PetscInt face = 0; face < newFace.rows(); face++) {
        PetscInt el = newFace(face) / faces.rows();
        PetscInt elface = newFace(face) % faces.rows();
        // Loop over every node in that face
        for (PetscInt nd = 0; nd < faces.cols(); nd++) {
          PetscInt inode = element(el, faces(elface, nd));
          // If local node, at it to list 
          if (inode < nLocNode) {
            loadNode(ind) = inode;
            loads.row(ind) = values;
            ind++;
          }
        }
      }
      // Trim extra space from nodes that were not local
      loads.conservativeResize(ind, numDims);
      loadNode.conservativeResize(ind);
      break;
    }
    case MASS: {
      masses.conservativeResize(massNode.rows()+newNode.rows(), numDims);
      for (PetscInt i = 0; i < numDims; i++)
        masses.block(massNode.rows(), i, newNode.rows(), 1).setConstant(values(i));
      massNode.conservativeResize(massNode.rows()+newNode.rows());
      massNode.segment(massNode.rows()-newNode.rows(), newNode.rows()) = newNode;
      break;
    }
    case SPRING: {
      springs.conservativeResize(springNode.rows()+newNode.rows(), numDims);
      for (PetscInt i = 0; i < numDims; i++)
        springs.block(springNode.rows(), i, newNode.rows(), 1).setConstant(values(i));
      springNode.conservativeResize(springNode.rows()+newNode.rows());
      springNode.segment(springNode.rows()-newNode.rows(), newNode.rows()) = newNode;
      break;
    }
    default: {
      PetscPrintf(comm, "Bad BC type given in input file, valid types are");
      PetscPrintf(comm, " Support, Load, Mass, and Spring\n");
      break;
    }
  }
  return ierr;
}

/********************************************************************
 * Set various parameters/options from input file
 * 
 * @param optmma: instance of the MMA optimizer
 * @param Dimensions: physical dimensions of the optimization domain
 * @param Nel: number of elements in each dimensions
 * @param Rmin: minimum length scale radius
 * @param Rmax: maximum length scale radius
 * @param Normalization: flag to calculate all possible objective
 *                       values upon termination
 * @param Reorder_Mesh: flag indicating if mesh should be redistributed
 * @param mg_levels: Number of levels to use in the GMG hierarchy
 * @param min_size: minimum matrix size per processor
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode TopOpt::Def_Param(MMA *optmma, Eigen::VectorXd &Dimensions,
               ArrayXPI &Nel, PetscScalar &Rmin, PetscScalar &Rmax, bool &Normalization,
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
      else if (!line.compare(0,1,"%")||!line.compare(0,1,"#")||!line.compare(0,2,"//"))
        getline(file,line);
      else if (!line.compare(0,10,"DIMENSIONS"))
      {
        getline(file, line);
        offset = 0;
        vector<PetscScalar> temp;
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
        PetscScalar pmin = strtod(line.c_str(), NULL);
        file >> line;
        PetscScalar pstep = strtod(line.c_str(), NULL);
        file >> line;
        PetscScalar pmax = strtod(line.c_str(), NULL);
        this->penalties.reserve(PetscInt(std::max(pmax-pmin, 0.0)/pstep));
        for (PetscScalar p = pmin; p <= pmax+pstep/2; p += pstep)
          this->penalties.push_back(p);
      }
      else if (!line.compare(0,12,"VOID_PENALTY"))
      {
        file >> line;
        PetscScalar pmin = strtod(line.c_str(), NULL);
        file >> line;
        PetscScalar pstep = strtod(line.c_str(), NULL);
        file >> line;
        PetscScalar pmax = strtod(line.c_str(), NULL);
        if (pmax == pmin || pstep == 0)
          this->void_penalties.push_back(pmin);
        else
        {
          this->void_penalties.reserve(PetscInt(std::max(pmax-pmin, 0.0)/pstep));
          for (PetscScalar p = pmin; p <= pmax+pstep/2; p += pstep)
            this->void_penalties.push_back(p);
        }
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
          else {
            interpolation = SIMP; }
        }
      }
      else if (!line.compare(0,10,"RMINFACTOR"))
      {
        file >> line;
        Rmin = strtod(line.c_str(), NULL)*(Dimensions(1)-Dimensions(0))/Nel(0);
      }
      else if (!line.compare(0,4,"RMIN"))
      {
        file >> line;
        Rmin = strtod(line.c_str(), NULL);
      }
      else if (!line.compare(0,10,"RMAXFACTOR"))
      {
        file >> line;
        Rmax = strtod(line.c_str(), NULL)*(Dimensions(1)-Dimensions(0))/Nel(0);
      }
      else if (!line.compare(0,4,"RMAX"))
      {
        file >> line;
        Rmax = strtod(line.c_str(), NULL);
      }
      else if (!line.compare(0,12,"VOID_MINIMUM"))
      {
        file >> line;
        this->vdMin = strtod(line.c_str(), NULL);
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
        PetscInt c_size = strtol(line.c_str(), NULL, 0);
        PetscScalar temp = log2(Nel.size());
        for (PetscInt i = 0; i < Nel.size(); i++)
          temp += log2(Nel(i)+1);
        temp -= log2(c_size);
        temp /= Nel.size();
        mg_levels = ceil(temp)+1;
      }
      else if (!line.compare(0,8,"SMOOTHER"))
      {
        string smoother; file >> smoother;
        for (string::size_type i = 0; i < smoother.length(); ++i)
          smoother[i] = toupper(smoother[i]);
        if (!smoother.compare(0,4,"RICH") || !smoother.compare(0,4,"WJAC") ||
            !smoother.compare(0,3,"JAC"))
          this->smoother = KSPRICHARDSON;
        else if (!smoother.compare(0,5,"CHEBY"))
          this->smoother = KSPCHEBYSHEV;
        else {
          SETERRQ1(comm, PETSC_ERR_SUP, "Unknown smoother type \"%s\" specified",
                   smoother.c_str()); }
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
      else if (!line.compare(0,5,"PRINT"))
      {
        file >> line;
        print_every = strtol(line.c_str(), NULL, 0);
      }

      file >> line;
    }
  }

  return ierr;
}

/********************************************************************
 * Get options from command line
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode TopOpt::Get_CL_Options()
{
  PetscErrorCode ierr = 0;
  ierr = PetscOptionsGetInt(NULL, NULL, "-Verbose", &verbose, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-Print_Every", &print_every, NULL);
         CHKERRQ(ierr);
  return ierr;
}

/********************************************************************
 * Set the optimization functions
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
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
      if (!line.compare(0,1,"%")||!line.compare(0,1,"#")||!line.compare(0,2,"//"))
      {
        getline(file,line);
        file >> line;
        continue;
      }
      else if (!line.compare(0,10,"Compliance"))
        func = COMPLIANCE;
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
      PetscScalar min = 0, max = 0;
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
        // Normalize constraint values
        if (objective == PETSC_FALSE)
        {
          for (unsigned int i = 0; i < values.size(); i++)
            values[i] *= max-min;
        }
        break;
      }

      switch(func){
        case COMPLIANCE :
          function_list.push_back(new Compliance(values, min, max, objective));
          needK = PETSC_TRUE; needU = PETSC_TRUE;
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

/********************************************************************
 * Check if elements should be removed
 * 
 * @param Points: Centroids of each element
 * @param elemValidity: List of whether each element is in or out
 * @param key: What domain we're setting ("Domain", "Active", or "Passive")
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
PetscErrorCode TopOpt::Domain(MatrixXPS &Points, Eigen::Array<bool, -1, 1> &elemValidity,
                              std::string key)
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
  ArrayXXPS D = ArrayXXPS::Zero(Points.rows(), 0);
  char enter[30], exit[30];
  sprintf(enter, "[%s]", key.c_str());
  sprintf(exit, "[/%s]", key.c_str());
  while (!file.eof())
  {
    if (!active_section)
    {
      active_section = !line.compare(enter);
      getline(file, line);
      file >> line;
    }
    else
    {
      if (!line.compare(exit))
      {
        if (D.cols() > 0) // Only set if domain was specified
          elemValidity = (D.col(D.cols()-1) < 0.0);
        return ierr;
      }
      else if (!line.compare(0,1,"%")||!line.compare(0,1,"#")||!line.compare(0,2,"//"))
      {
        getline(file,line);
        file >> line;
        continue;
      }

      // Convert to all uppercase to avoid captilization errors
      line[0] = toupper(line[0]);

      if (line[0] == 'E') { // Ellipsoid
        ArrayXPS center, radius;
        getline(file, line);
        while (true) {
          file >> line;
          if (toupper(line[0]) == 'C' && toupper(line[1]) == 'E') { // Centroid
            getline(file, line);
            vector<PetscScalar> temp = Get_Values(line);
            if (temp.size() != (unsigned short)numDims)
              cout << "Centers for Ellipsoid are specified incorrectly\n";
            center = Eigen::Map<ArrayXPS>(temp.data(), temp.size());
            continue;
          }
          if (toupper(line[0]) == 'R') { // Radii
            getline(file, line);
            vector<PetscScalar> temp = Get_Values(line);
            if (temp.size() != (unsigned short)numDims)
              cout << "Radii for ellipsoid are specified incorrectly\n";
            radius = Eigen::Map<ArrayXPS>(temp.data(), temp.size());
            continue;
          }
          break;
        }
        D.conservativeResize(D.rows(), D.cols()+1);
        D.col(D.cols()-1) = Domain::Ellipsoid(Points, center, radius);
      }
      else if (line[0] == 'C') // Cylinder
      {
        ArrayXPS center;
        VectorXPS normal;
        PetscScalar r, h;
        getline(file, line);
        while (true)
        {
          file >> line;
          if (toupper(line[0]) == 'C' && toupper(line[1]) == 'E')
          {
            getline(file, line);
            vector<PetscScalar> temp = Get_Values(line);
            if (temp.size() < (unsigned short)numDims)
              cout << "Centers for Cylinder are specified incorrectly\n";
            center = Eigen::Map<ArrayXPS>(temp.data(), temp.size());
            continue;
          }
          if (toupper(line[0]) == 'N')
          {
            getline(file, line);
            vector<PetscScalar> temp = Get_Values(line);
            if (temp.size() < (unsigned short)numDims)
              cout << "Axis vectors for Cylinder are specified incorrectly\n";
            normal = Eigen::Map<VectorXPS>(temp.data(), temp.size());
            continue;
          }
          if (toupper(line[0]) == 'R')
          {
            getline(file, line);
            r = strtod(line.c_str(), NULL);
            continue;
          }
          if (toupper(line[0]) == 'H' && toupper(line[2]) == 'I')
          {
            getline(file, line);
            h = strtod(line.c_str(), NULL);
            continue;
          }
          break;
        }
        D.conservativeResize(D.rows(), D.cols()+1);
        D.col(D.cols()-1) = Domain::Cylinder(Points, center, normal, h, r);
      }
      else if (line[0] == 'H') // Hexahedron
      {
        ArrayXPS low, up;
        getline(file, line);
        while (true)
        {
          file >> line;
          if (toupper(line[0]) == 'L')
          {
            getline(file, line);
            vector<PetscScalar> temp = Get_Values(line);
            if (temp.size() < (unsigned short)numDims)
              cout << "Lower bounds for hexahedron are specified incorrectly\n";
            low = Eigen::Map<ArrayXPS>(temp.data(), temp.size());
            continue;
          }
          if (toupper(line[0]) == 'U' && toupper(line[1]) == 'P')
          {
            getline(file, line);
            vector<PetscScalar> temp = Get_Values(line);
            if (temp.size() < (unsigned short)numDims)
              cout << "Upper bounds for hexahedron are specified incorrectly\n";
            up = Eigen::Map<ArrayXPS>(temp.data(), temp.size());
            continue;
          }
          break;
        }
        D.conservativeResize(D.rows(), D.cols()+1);
        D.col(D.cols()-1) = Domain::Hexahedron(Points, low, up);
      }
      else if (line[0] == 'P') // Plane
      {
        ArrayXPS base;
        VectorXPS normal;
        getline(file, line);
        while (true)
        {
          file >> line;
          if (toupper(line[0]) == 'B')
          {
            getline(file, line);
            vector<PetscScalar> temp = Get_Values(line);
            if (temp.size() < (unsigned short)numDims)
              cout << "Base coordinates for plane are specified incorrectly\n";
            base = Eigen::Map<ArrayXPS>(temp.data(), temp.size());
            continue;
          }
          if (toupper(line[0]) == 'N')
          {
            getline(file, line);
            vector<PetscScalar> temp = Get_Values(line);
            if (temp.size() < (unsigned short)numDims)
              cout << "Normal coordinates for plane are specified incorrectly\n";
            normal = Eigen::Map<VectorXPS>(temp.data(), temp.size());
            continue;
          }
          break;
        }
        D.conservativeResize(D.rows(), D.cols()+1);
        D.col(D.cols()-1) = Domain::Plane(Points, base, normal);
      }
      else if (line[0] == 'U' || line[0] == 'I' || line[0] == 'D')
      { // Union, Intersection, or Difference
        char type = line[0];
        PetscInt d1, d2;
        file >> line;
        d1 = strtol(line.c_str(), NULL, 0);
        getline(file, line);
        d2 = strtol(line.c_str(), NULL, 0);
        file >> line;
        D.conservativeResize(D.rows(), D.cols()+1);

        if (type == 'U') // Union
          D.col(D.cols()-1) = Domain::Union(D.col(d1), D.col(d2));
        else if (type == 'I') // Intersection
          D.col(D.cols()-1) = Domain::Intersect(D.col(d1), D.col(d2));
        else if (type == 'D') // Difference
          D.col(D.cols()-1) = Domain::Difference(D.col(d1), D.col(d2));
      }
      else
      {
        SETERRQ1(comm, PETSC_ERR_SUP, "Unknown domain specifier, %s, specified",
                  line.c_str());
      }
    }
        
  }

  return ierr;
}

/********************************************************************
 * Define boundary conditions
 * 
 * @return ierr: PetscErrorCode
 * 
 *******************************************************************/
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
      ArrayXPS center, radius, values;
      ArrayXXPS limits;
      TYPE = OTHER;
      if (!line.compare("[/BC]"))
        return ierr;
      else if (!line.compare(0,1,"%")||!line.compare(0,1,"#")||!line.compare(0,2,"//"))
      {
        getline(file,line);
        file >> line;
        continue;
      }
      else if (!line.compare(0,7,"Support"))
        TYPE = SUPPORT;
      else if (!line.compare(0,4,"Load"))
        TYPE = LOAD;
      else if (!line.compare(0,5,"Press"))
        TYPE = PRESSURE;
      else if (!line.compare(0,4,"Mass"))
        TYPE = MASS;
      else if (!line.compare(0,6,"Spring"))
        TYPE = SPRING;
      else if (!line.compare(0,3,"Eig"))
        TYPE = EIGEN;
      getline(file, line);

      while (true)
      {
        file >> line;
        if (!line.compare(0,6,"Center"))
        {
          getline(file, line);
          vector<PetscScalar> temp = Get_Values(line);
          if (temp.size() != (unsigned short)numDims)
            cout << "Centers for BC " << TYPE << " are specified incorrectly\n";
          center = Eigen::Map<ArrayXPS>(temp.data(), temp.size());
          continue;
        }
        if (!line.compare(0,6,"Radius"))
        {
          getline(file, line);
          vector<PetscScalar> temp = Get_Values(line);
          if (temp.size() != (unsigned short)numDims)
            cout << "Radii for BC " << TYPE << " are specified incorrectly\n";
          radius = Eigen::Map<ArrayXPS>(temp.data(), temp.size());
          continue;
        }
        if (!line.compare(0,6,"Limits"))
        {
          getline(file, line);
          vector<PetscScalar> temp = Get_Values(line);
          if (temp.size()/2 != (unsigned short)numDims)
            cout << "Limits for BC " << TYPE << " are specified incorrectly\n";
          limits = Eigen::Map<ArrayXXPS>(temp.data(), 2, temp.size()/2);
          limits.transposeInPlace();
          continue;
        }
        if (!line.compare(0,6,"Values"))
        {
          getline(file, line);
          vector<PetscScalar> temp = Get_Values(line);
          if (temp.size() != (unsigned short)numDims)
            cout << "Values for BC " << TYPE << " are specified incorrectly\n";
          values = Eigen::Map<ArrayXPS>(temp.data(), temp.size());
          continue;
        }
        break;
      }

      ierr = Set_BC(center, radius, limits, values, TYPE); CHKERRQ(ierr);
    }
  }
  return 0;
}