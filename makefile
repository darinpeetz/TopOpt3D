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
SOURCE = $(foreach file, $(CPPS),$(notdir $(basename $(file))))

all: TopOpt

tidy:
	rm -f ${BUILD_DIR}/*.o

#File_List = Main TopOpt Inputs RecMesh Filter Interpolation MMA FEAnalysis Functions Volume Compliance Perimeter Buckling Dynamic EigenPeetz PRINVIT JDMG LOPGMRES

TopOpt: $(patsubst %,${BUILD_DIR}/%.o, ${SOURCE})
	${LINK} $(patsubst %,${BUILD_DIR}/%.o, ${SOURCE}) -o TopOpt_${BUILD_DIR} ${SLEPC_EPS_LIB} -lparmetis -lmetis

# Build any needed object files
${BUILD_DIR}/%.o: %.cpp
	${COMPILE} -c $< -o $@

# Make a list of dependencies
depends:
	@ rm -f depends.mk
	@ for f in $(SOURCE); do echo $$f; g++ -MM -MG $$f.cpp -MT $$f.o >> depends.mk; done

# Include dependencies list
include depends.mk
