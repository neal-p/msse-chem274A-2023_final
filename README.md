[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/ZQvpIPgx)

# C++: Matrix Implementation

Here, I provide a header-only, barebones, Matrix library at `Matrix/include/Matrix.h`.

## General design

The matrix class is a templated class that is intended for use on any numeric type. Internally, I chose to store the matrix as a contiguous 1D C-style array to maximize data locality and provide easy BLAS/LAPACK compatability. I use a `std::unique_ptr` to manage the underlying array's memory since this class should be the only owner of the array's data. I tried to style some of the matrix operations and API like numpy's python interface. The elements are access with row-major semantics, the shape uses a notion of 'axis', and you can add and multiply the matrix by scalars similarly to numpy. 

I also wrote BLAS/LAPACK support for the matrix class with `cblas` and `lapacke`. Currently, BLAS can be used for matrix matrix multiplication (as well as the naive nested loop approach by calling `setMethod(NAIVE)`), AND LAPACK is used for solving eigen problems. 

I had a lot of fun learning how to specialize for the different types and call the proper BLAS/LAPACK routines. It turned out to be a really good way to learn more C++ and try to take the idea of how I want the class to act and find a clean way to implement it!

## Matrix constructor and init

There are three ways a matrix can be created:

    1. with no arguments: `Matrix<int> m;` will create an empty 0x0 matrix of ints. You must call `reshape(rows, cols)` to later populate the matrix
    2. with desired rows and columns: `Matrix<int>(2, 3);` will create a matrix with 2 rows and 3 columns
    3. as a copy of another matrix: `Matrix<int> m2 = m;` will create a deep copy of matrix `m` to `m2`

## Populating values and acessing elements

There are two ways to access and or modify the elements of a `Matrix`. Both of these operations will throw an `std::out_of_range` exception if the requested element is beyond the bounds of the array.

    1. via parenthesis operator with (row, col): `m(1, 2) = 1;` will access the second row, third column of the matrix and set it to 1
    2. via parenthesis operator with an absolute idx: `m(5) = 1` will access the 6th element in the flat row-major underlying array.

The user must either populate the array point by point, or since accessing the underlying flat array is supported, it is easy to set the values in a loop such as the following:

```
  double a[] = {-1.01, 0.86, -4.60, 3.31,  -4.81, 
                 3.98,  0.53, -7.04, 5.29,  3.55,
                 3.30, 8.26,  -3.89, 8.20,  -1.51, 
                 4.43, 4.96,  -7.66, -7.33, 6.18, 
                 7.31,  -6.43, -6.16, 2.47,  5.58
                 };

  Matrix<double> symm(5, 5);

  for (int idx = 0; idx < 25; idx++) {
    symm(idx) = a[idx];
  }
```

## Matrix sizing

The size of the matrix can be changed with either `void Matrix<T>::reshape(const int rows, const int cols)`, or `void Matrix<T>::reshapeLike(const Matrix<T>& m)`. The former sets the rows and columns as specified, the latter will reshape the array based on the size of another array. Both operations will trigger a new memory allocation to store the re-sized array if the total size is different than the current size. Additionally, both will copy what data existed in the previous array which fits in the size of the new array.

## Matrix Operations

The following sections detail the different operations availible on the `Matrix` class:


`void setMethod(Method method)`
Set method to either `NAIVE` or `BLAS` for matrix multiplication

`Method getMethod() const`
Get current method.

`void reshape(const int rows, const int cols)`
Reshape array to passed rows and cols and preserve what data fits in the new size. Only triggers reallocation if total number of elements changes.

`void reshapeLike(const Matrix<T>& mat)`
Utility function to reshape this array to match the size of another `Matrix`

`Matrix<T> tp() const`
Get the transpose of the matrix

`void tp_ip()`
Transpose in place

`Matrix<T> ccj() const`
Get the complex conjugate of the matrix

`void ccj_ip()`
Complex conjugate in place

`Matrix<T> ct() const`
Get the conjugate transpose of the matrix

`void ct_ip()`
Conjugate transpose in place

`void operator=(const T value)`
Fill matrix with single value. Can also call `m.fill(5.0)` to fill values.

`fill(const T value)`
Fill matrix with single value. Can also call `m = 5.0` to fill values.

`Matrix<T> operator+(const Matrix<T>& m) const`
Matrix addition. Matrices must be same shape.

`Matrix<T> multiplyElements(const Matrix<T>& m) const`
Elementwise multiplication with another matrix. Matrices must be same shape.

`Matrix<T> operator*(const T value) const`
Multiply matrix by scalar value

`Matrix<T> operator*(const Matrix<T>& mat) const`
Matrix-Matrix multiplication. If the `method_` is set to `NAIVE`, will do the straightforward nested loop calculation. If set to `BLAS`, then BLAS routine will be called. Note, calling BLAS requires an extra memory allocation to copy the values of the current matrix since BLAS will mutate them.

`bool operator==(const Matrix<T>& mat) const`
Equality operator. First checks O(1) check for equal shape, then O(N) check of all elements' equality

`bool operator!=(const Matrix<T>& mat) const`
Inequality operator. 

`Matrix<std::complex<U>> eigenvalues() const`
Use LAPACK to calcluate the eigenvalues of the matrix

`Matrix<std::complex<U>> reigenvectors() const`
Use LAPACK to calculate the right eigenvectors of the matrix

`Matrix<std::complex<U>> leigenvectors() const`
Use LAPACK to calculate the left eigenvectors of the matrix

`std::vector<Matrix<std::complex<U>>> alleigen() const`
Use LAPACK to calculate eigenvalues, left eigenvectors, and right eigenvectors. Returned as a vector of matrices in that given order.

`void print() const`
Print the matrix to cout


## Matrix testing and example use

Provided in `Matrix/tests/test_matrix.cpp` is a comprehensive test of all the above functionalities. The program demonstrates each of the required features and generally shows how the matrices can be manipulated. A Makefile is provided at `Matrix/tests/Makefile` that will compile the test program with the target `test_matrix` to `Matrix/bin/test_matrix`. Additionally, to run a Valgrind leak-check, the target `leak_check` target is provided. 

Note! Since I have used BLAS and LAPACK, the `cblas` and `lapacke` headers must be visible to the compiler as well as the `libcblas` and `liblapacke`, `liblapack` library objects.

Output of the test is provided in `Matrix/bin/test_matrix.txt`

# Python: MYKit, a want to be RDKit

I use RDKit every day for work and have a few times tried to hand-roll my own RDKit functionalities, all attempts have ended with the upmost appriciation for the authors of the RDKit code. Modelling molecules is really difficult to do well, but here I attempted to provide some of the simplest functionalities of a cheminformatics software package. 

## General Design

The `Mol` object is the molecule container class in MYKit. This object represents a molecule graph, which is internally implemented as a `NetworkX` graph. Unfortunately, this basic graph structure lacks the chemistry awareness that RDKit provides, not being able to detect aromaticity, correct valences, radicals, invalid structures, or implicit hydrogens. I briefly went down the route of putting in some very basic heuristics based on the number of valence electrons of an atom knowing how many bonds it should form, etc. But, to do so properly quickly became out of scope for this project. 

But, despite this glaring real-world limitation, I did have a lot of fun putting together the class and especially had some fun extending to 3D plotting.

## MYKit directory structure

<div>
.
ÀÄÄ MYKit/
    ÃÄÄ bin/
    ³   ÀÄÄ substructure_search.py
    ÃÄÄ MYKit/
    ³   ÀÄÄ utilities
    ÃÄÄ notebooks/
    ³   ÃÄÄ substructure_search.ipynb
    ³   ÃÄÄ 2D_display.ipynb
    ³   ÀÄÄ 3D_display.ipynb
    ÃÄÄ tests/
    ³   ÀÄÄ test_mol_base.py
    ÃÄÄ Makefile
    ÀÄÄ environment.yml
</div>

 - `bin`: provides command line `substructure_search.py` substructure search utility
 - `MYKit`: source directory with main `Mol` class defined in `mol_base.py`
 - `MYKit/utilities`: contains periodic table, display, and SDF parsing logic
 - `notebooks`: provides notebooks showaseing the substructure search and molecule display
 - `tests`: tests all of the `Mol` functionalities on a few basic structures
 - `Makefile`: contains target for building the environment `environment`, running tests `test`, and linting `lint`
 - `environment.yml`: specify required packages




