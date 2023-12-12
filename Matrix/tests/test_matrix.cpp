#include <cassert>
#include <iostream>
#include "Matrix.h"

#define bannerS \
  "\n#############################################################\n"
#define bannerE \
  "#############################################################\n\n"

int main() {
  // 1. Demonstate different types:
  Matrix<int> m_i;
  Matrix<float> m_s;
  Matrix<double> m_d;
  Matrix<std::complex<double>> m_z;

  // 2. Demonstate alternate constructors
  Matrix<int> m_i4(4, 4);        // specify rows, columns
  Matrix<int> m_i4_copy = m_i4;  // by copy of existing matrix

  // 3. Accessing and modifying elements by parenthesis operator
  std::cout << bannerS;
  std::cout << "# 3. Acessing/modifying elements by paranthesis operator"
            << std::endl;
  std::cout << bannerE;

  std::cout << "initial value of m_i4(0, 0) = " << m_i4(0, 0) << std::endl;
  m_i4(0, 0) = 10;
  std::cout << "after modifying, m_i4(0, 0) = " << m_i4(0, 0) << std::endl;
  assert(m_i4(0, 0) == 10);

  std::cout << "initial value of m_i4(1, 1) = " << m_i4(1, 1) << std::endl;
  m_i4(1, 1) = 50;
  std::cout << "after modifying, m_i4(1, 1) = " << m_i4(1, 1) << std::endl;
  assert(m_i4(1, 1) == 50);

  // 4. Reshaping a matrix
  std::cout << bannerS;
  std::cout << "# 4. Reshaping Matrix" << std::endl;
  std::cout << bannerE;

  std::cout << "Grow m_i4 from 4x4 to 4x8" << std::endl;
  std::cout << "Before: " << std::endl << m_i4 << std::endl;
  m_i4.reshape(4, 8);
  std::cout << "After: " << std::endl << m_i4 << std::endl;
  assert(m_i4.getShape(-1) == 4 * 8);

  // 5. Matrix operations:
  std::cout << bannerS;
  std::cout << "# 5. Matrix Operations" << std::endl;
  std::cout << bannerE;

  // 5.1 Matrix addition
  std::cout << bannerS;
  std::cout << "# 5.1. Matrix Addition" << std::endl;
  std::cout << bannerE;

  m_d.reshape(2, 2);
  for (int idx = 0; idx < 4; idx++) {
    m_d(idx) =
        1;  // since my data is stored under the hood as a flat contiguous
            // array, I also give assess to the raw index in the flat array
  }

  std::cout << "Add 2x2 matrix of 1s to itself" << std::endl;
  std::cout << "Matrix Before: " << std::endl << m_d << std::endl;
  auto m_d_add = m_d + m_d;  // also demonstrates assignment overload
  std::cout << "Matrix After: " << std::endl << m_d_add << std::endl;
  for (int idx = 0; idx < 4; idx++) {
    assert(m_d_add(idx) == 2.0);
  }

  // 5.2 Matrix multiplication
  std::cout << bannerS;
  std::cout << "# 5.2. Matrix Multiplication" << std::endl;
  std::cout << bannerE;

  Matrix<double> A(2, 3);
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(0, 2) = 3;
  A(1, 0) = 4;
  A(1, 1) = 5;
  A(1, 2) = 6;

  Matrix<double> B(3, 2);
  B(0, 0) = 7;
  B(0, 1) = 8;
  B(1, 0) = 9;
  B(1, 1) = 10;
  B(2, 0) = 11;
  B(2, 1) = 12;

  std::vector<double> M_X_M_expecteds{58, 64, 139, 154};

  std::cout << "Matrix A: " << std::endl << A << std::endl;
  std::cout << "Matrix B: " << std::endl << B << std::endl;

  A.setMethod(NAIVE);
  auto AB_naive = A * B;
  std::cout << "Using NAIVE implementation: " << std::endl;
  std::cout << "A X B: " << std::endl << AB_naive << std::endl;
  for (int idx = 0; idx < 4; idx++) {
    assert(M_X_M_expecteds[idx] == AB_naive(idx));
  }

  A.setMethod(BLAS);
  auto AB_BLAS = A * B;
  std::cout << "Using BLAS implementation: " << std::endl;
  std::cout << "A X B: " << std::endl << AB_BLAS << std::endl;
  for (int idx = 0; idx < 4; idx++) {
    assert(M_X_M_expecteds[idx] == AB_BLAS(idx));
  }

  // 5.3 Elementwise Matrix multiplication
  std::cout << bannerS;
  std::cout << "# 5.3. Elementwise Matrix Multiplication" << std::endl;
  std::cout << bannerE;
}
