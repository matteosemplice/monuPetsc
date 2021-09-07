#Create a PetscDef.make file with the appropriate defines to find petsc
# make sure PETSC_DIR and PETSC_ARCH are defined as environment variables
# or in the PetscDef.make file
include PetscDef.make

CXXFLAGS         = -g

# defines: passed to compiler on command-line
DEFS =
# NDEBUG         : turns off assert and extra debugging code
#DEFS += -D NDEBUG

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

EXE=monuCart
MAIN=monuCart.cpp

EXTRASRC=appctx.cpp hdf5Output.cpp levelSet.cpp levelSetTest.cpp test.cpp sulfation.cpp
EXTRAOBJS=${EXTRASRC:%.cpp=%.o}
EXTRAHDR=${EXTRASRC:%.cpp=%.h}

all: ${EXE}

${EXE}: ${MAIN:.cpp=.o} ${EXTRAOBJS} ${EXTRAHDR}
	-${CXXLINKER} -o ${EXE} ${DEFS} ${MAIN:.cpp=.o} ${EXTRAOBJS} ${PETSC_DM_LIB} ${PETSC_SNES_LIB} ${PETSC_KSP_LIB} ${PETSC_SYS_LIB}

testHDF5: testHDF5.o appctx.h
	-${CXXLINKER} -o testHDF5 ${DEFS} testHDF5.o ${PETSC_DM_LIB} ${PETSC_SNES_LIB} ${PETSC_KSP_LIB} ${PETSC_SYS_LIB}

%.o: %.cpp %.h appctx.h
	-${PETSC_CXXCOMPILE_SINGLE} -c ${DEFS} -o $@ $*.cpp

test.cpp: test.c

test.c: test0.py
	-./test0.py
