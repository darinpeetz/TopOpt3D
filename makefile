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

all: TopOpt

tidy:
	rm -f ${BUILD_DIR}/*.o

File_List = Main TopOpt Inputs RecMesh Filter Interpolation MMA FEAnalysis Functions Volume Compliance Perimeter Buckling Dynamic EigenPeetz PRINVIT JDMG LOPGMRES

TopOpt: $(patsubst %,${BUILD_DIR}/%.o, ${File_List})
	${LINK} $(patsubst %,${BUILD_DIR}/%.o, ${File_List}) -o TopOpt_${BUILD_DIR} ${SLEPC_EPS_LIB} -lparmetis -lmetis

${BUILD_DIR}/TopOpt.o: TopOpt.cpp TopOpt.h
	${COMPILE} TopOpt.cpp -c -o ${BUILD_DIR}/TopOpt.o

${BUILD_DIR}/Main.o: Main.cpp TopOpt.h MMA.h Functions.h
	${COMPILE} Main.cpp -c -o ${BUILD_DIR}/Main.o

${BUILD_DIR}/Inputs.o: Inputs.cpp TopOpt.h MMA.h EigLab.h Functions.h
	${COMPILE} Inputs.cpp -c -o ${BUILD_DIR}/Inputs.o

${BUILD_DIR}/RecMesh.o: RecMesh.cpp TopOpt.h EigLab.h
	${COMPILE} RecMesh.cpp -c -o ${BUILD_DIR}/RecMesh.o

${BUILD_DIR}/Filter.o: Filter.cpp TopOpt.h
	${COMPILE} Filter.cpp -c -o ${BUILD_DIR}/Filter.o

${BUILD_DIR}/Interpolation.o: Interpolation.cpp TopOpt.h
	${COMPILE} Interpolation.cpp -c -o ${BUILD_DIR}/Interpolation.o

${BUILD_DIR}/MMA.o: MMA.cpp MMA.h
	${COMPILE} MMA.cpp -c -o ${BUILD_DIR}/MMA.o

${BUILD_DIR}/FEAnalysis.o: FEAnalysis.cpp TopOpt.h
	${COMPILE} FEAnalysis.cpp -c -o ${BUILD_DIR}/FEAnalysis.o

${BUILD_DIR}/Functions.o: Functions.cpp Functions.h TopOpt.h
	${COMPILE} Functions.cpp -c -o ${BUILD_DIR}/Functions.o

${BUILD_DIR}/Volume.o: Volume.cpp Functions.h TopOpt.h
	${COMPILE} Volume.cpp -c -o ${BUILD_DIR}/Volume.o

${BUILD_DIR}/Compliance.o: Compliance.cpp Functions.h TopOpt.h
	${COMPILE} Compliance.cpp -c -o ${BUILD_DIR}/Compliance.o

${BUILD_DIR}/Perimeter.o: Perimeter.cpp Functions.h TopOpt.h
	${COMPILE} Perimeter.cpp -c -o ${BUILD_DIR}/Perimeter.o

${BUILD_DIR}/Buckling.o: Buckling.cpp Functions.h TopOpt.h EigenPeetz.h LOPGMRES.h
	${COMPILE} Buckling.cpp -c -o ${BUILD_DIR}/Buckling.o

${BUILD_DIR}/Dynamic.o: Dynamic.cpp Functions.h TopOpt.h EigenPeetz.h LOPGMRES.h
	${COMPILE} Dynamic.cpp -c -o ${BUILD_DIR}/Dynamic.o

${BUILD_DIR}/EigenPeetz.o: EigenPeetz.cpp EigenPeetz.h
	${COMPILE} EigenPeetz.cpp -c -o ${BUILD_DIR}/EigenPeetz.o

${BUILD_DIR}/PRINVIT.o: PRINVIT.cpp EigenPeetz.h PRINVIT.h
	${COMPILE} PRINVIT.cpp -c -o ${BUILD_DIR}/PRINVIT.o

${BUILD_DIR}/JDMG.o: JDMG.cpp EigenPeetz.h PRINVIT.h JDMG.h
	${COMPILE} JDMG.cpp -c -o ${BUILD_DIR}/JDMG.o

${BUILD_DIR}/LOPGMRES.o: LOPGMRES.cpp EigenPeetz.h PRINVIT.h LOPGMRES.h
	${COMPILE} LOPGMRES.cpp -c -o ${BUILD_DIR}/LOPGMRES.o
