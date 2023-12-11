# GNU C++ Compiler
CPP         = g++ -std=c++17      # C++ Compiler

# compiler flags -g debug, -O2 optimized version -c create a library object
CPPFLAGS    = -g -O0

# The directory where the include files needed to create the library objects are
INC_DIR = ../include

# The bin directory to put exes
BIN_DIR = ../bin

ALLTESTS = test_basic_matrix

# The Lib directory 
LIB_DIR = ../lib


#####################################################################################
#####################################################################################
#####################################################################################

test_basic_matrix: test_basic_matrix.cpp Matrix.h
	$(CPP) $(CPPFLAGS) test_basic_matrix.cpp -o test_basic_matrix -llapacke -llapack -lcblas  

testcblas: testcblas.cpp
	$(CPP) $(CPPFLAGS) testcblas.cpp -o testcblas -lcblas 

testlapack: testlapack.cpp
	$(CPP) $(CPPFLAGS) testlapack.cpp -o testlapack -llapacke -llapack -lcblas  

testeigenvectors: testeigenvectors.cpp
	$(CPP) $(CPPFLAGS) testeigenvectors.cpp -o testeigenvectors -llapacke -llapack -lcblas  


ALLTESTS = test_basic_matrix

make all: $(ALLTESTS)
