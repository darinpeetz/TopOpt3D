ifeq ($(DEBUG),yes)
    OPT_FLAG = -g3
    PETSC_ARCH = arch-linux-debug
    BUILD_DIR = debug
else
    OPT_FLAG = -g0 -O3 -march=native -mtune=native
    PETSC_ARCH = arch-linux-opt
    BUILD_DIR = opt
endif

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

# This avoids grabbing include files from other libraries (Eigen, Petsc, etc.)
# For use with intel compiler
#DEPEND = mpicxx -isystem${MYLIB_DIR}/Eigen -isystem${SLEPC_DIR}/include \
          -isystem${SLEPC_DIR}/${PETSC_ARCH}/include -isystem${PETSC_DIR}/include \
          -isystem${PETSC_DIR}/${PETSC_ARCH}/include
# For use with gcc
DEPEND = g++

COMPILE = mpicxx -fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas ${OPT_FLAG} \
          -I${MYLIB_DIR}/Eigen -I${SLEPC_DIR}/include \
          -I${SLEPC_DIR}/${PETSC_ARCH}/include -I${PETSC_DIR}/include \
          -I${PETSC_DIR}/${PETSC_ARCH}/include

LINK  =   mpicxx -fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas ${OPT_FLAG}

# All the source files in this directory
CPPS = $(wildcard *.cpp)
# All the compiled source files
OBJS = $(CPPS:.cpp=.o)
# Extensionless filenames
SOURCE = $(foreach file, $(CPPS), $(filter-out Ignore_%, $(notdir $(basename $(file)))))

all: TopOpt_${BUILD_DIR}

tidy:
	rm -f ${BUILD_DIR}/*.o

#SOURCE = Main TopOpt Inputs RecMesh Filter Interpolation MMA FEAnalysis Functions Volume Compliance Perimeter Buckling Dynamic EigenPeetz PRINVIT JDMG LOPGMRES

TopOpt_${BUILD_DIR}: $(patsubst %,${BUILD_DIR}/%.o, ${SOURCE})
	${LINK} $(patsubst %,${BUILD_DIR}/%.o, ${SOURCE}) -o TopOpt_${BUILD_DIR} ${SLEPC_EPS_LIB} -lparmetis -lmetis

# Build any needed object files
${BUILD_DIR}/%.o: %.cpp
	${COMPILE} -c $< -o $@

# Make a list of dependencies
depends:
	@ rm -f ${BUILD_DIR}/depends.mk
	@ for f in $(SOURCE); do echo $$f; echo "${BUILD_DIR}/""$$(${DEPEND} -MM -MG $$f.cpp -MT $$f.o)" >> ${BUILD_DIR}/depends.mk; done

# Include dependencies list
include ${BUILD_DIR}/depends.mk
