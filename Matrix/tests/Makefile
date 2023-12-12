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

old_test_basic_matrix: test_basic_matrix.cpp $(INC_DIR)/Matrix.h
	$(CPP) $(CPPFLAGS) test_basic_matrix.cpp -o $(BIN_DIR)/test_basic_matrix -I$(INC_DIR) -llapacke -llapack -lcblas  

test_matrix: test_matrix.cpp $(INC_DIR)/Matrix.h
	$(CPP) $(CPPFLAGS) test_matrix.cpp -o $(BIN_DIR)/test_matrix -I$(INC_DIR) -llapacke -llapack -lcblas  


ALLTESTS = test_matrix

make all: $(ALLTESTS)
