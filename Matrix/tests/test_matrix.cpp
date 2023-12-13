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

  B.reshape(2, 3);
  std::vector<double> M_X_M_ew_expecteds{7, 16, 27, 40, 55, 72};

  auto AB_elementwise = A.multiplyElements(B);
  std::cout << "Matrix A: " << std::endl << A << std::endl;
  std::cout << "Matrix B: " << std::endl << B << std::endl;
  std::cout << "Elementwise A X B: " << std::endl
            << AB_elementwise << std::endl;
  for (int idx = 0; idx < 4; idx++) {
    assert(M_X_M_ew_expecteds[idx] == AB_elementwise(idx));
  }

  // 5.4 Elementwise Matrix multiplication
  std::cout << bannerS;
  std::cout << "# 5.4. Assignment via = operator" << std::endl;
  std::cout << bannerE;

  std::cout << "Matrix A: " << std::endl << A << std::endl;
  std::cout << "Matrix B: " << std::endl << B << std::endl;

  std::cout << "Assign A = B" << std::endl;
  A = B;
  std::cout << "A: " << std::endl << A << std::endl;

  std::vector<double> copy_expected{7, 8, 9, 10, 11, 12};
  for (int idx = 0; idx < 6; idx++) {
    assert(copy_expected[idx] == A(idx));
  }

  // 5.5 Filling an array
  std::cout << bannerS;
  std::cout << "# 5.4. Fill array" << std::endl;
  std::cout << bannerE;

  std::cout << "A before: " << std::endl << A << std::endl;
  A.fill(5.);
  std::cout << "A .fill(5): " << std::endl << A << std::endl;
  for (int idx = 0; idx < 6; idx++) {
    assert(A(idx) == 5.0);
  }
  A = 10.;
  std::cout << "A = 10.0: " << std::endl << A << std::endl;
  for (int idx = 0; idx < 6; idx++) {
    assert(A(idx) == 10.0);
  }

  // 5.6 Compare matrices
  std::cout << bannerS;
  std::cout << "# 5.6. Compare matrices" << std::endl;
  std::cout << bannerE;

  auto C = B;

  std::cout << "Matrix A: " << std::endl << A << std::endl;
  std::cout << "Matrix B: " << std::endl << B << std::endl;
  std::cout << "Matrix C: " << std::endl << C << std::endl;

  std::cout << "A == B: " << (A == B) << std::endl;
  assert(A != B);
  std::cout << "B == C: " << (B == C) << std::endl;
  assert(B == C);

  // 5.8 Transpose
  std::cout << bannerS;
  std::cout << "# 5.8. Get transpose" << std::endl;
  std::cout << bannerE;

  std::cout << "Matrix A: " << std::endl << A << std::endl;
  auto A_t = A.tp();
  std::cout << "Transposed A: " << std::endl << A_t << std::endl;
  for (int x = 0; x < A.getShape(0); x++) {
    for (int y = 0; y < A.getShape(1); y++) {
      assert(A(y, x) == A_t(x, y));
    }
  }

  std::cout << "Matrix B: " << std::endl << B << std::endl;
  auto B_t = B.tp();
  std::cout << "Transposed B: " << std::endl << B_t << std::endl;
  for (int x = 0; x < B.getShape(0); x++) {
    for (int y = 0; y < B.getShape(1); y++) {
      assert(B(y, x) == B_t(x, y));
    }
  }

  // 5.9 Complex conjugate
  std::cout << bannerS;
  std::cout << "# 5.9. Get complex conjugate" << std::endl;
  std::cout << bannerE;

  std::cout << "Matrix A: " << std::endl << A << std::endl;
  auto A_ccj = A.ccj();
  std::cout << "Complex conjugate A: " << std::endl << A_ccj << std::endl;
  for (int x = 0; x < A.getShape(0); x++) {
    for (int y = 0; y < A.getShape(1); y++) {
      assert(A(y, x) == A_ccj(y, x));
    }
  }

  Matrix<std::complex<double>> D(2, 2);
  D(0, 0) = std::complex<double>(1, 1);
  D(0, 1) = std::complex<double>(1, -1);
  D(1, 0) = std::complex<double>(5, 0.1);
  D(1, 1) = std::complex<double>(5, -1.5);

  std::cout << "Matrix D: " << std::endl << D << std::endl;
  auto D_ccj = D.ccj();
  std::cout << "Complex conjugate D: " << std::endl << D_ccj << std::endl;
  for (int x = 0; x < D.getShape(0); x++) {
    for (int y = 0; y < D.getShape(1); y++) {
      assert(std::conj(D(y, x)) == D_ccj(y, x));
    }
  }

  // 5.10 Conjugate transpose
  std::cout << bannerS;
  std::cout << "# 5.10. Get conjugate transpose" << std::endl;
  std::cout << bannerE;

  auto D_og = D;

  std::cout << "Matrix A: " << std::endl << A << std::endl;
  auto A_ct = A.ct();
  std::cout << "Conjugate transpose A: " << std::endl << A_ct << std::endl;
  for (int x = 0; x < A.getShape(0); x++) {
    for (int y = 0; y < A.getShape(1); y++) {
      assert(A(y, x) == A_ct(x, y));
    }
  }

  std::cout << "Matrix D: " << std::endl << D << std::endl;
  auto D_ct = D.ct();
  std::cout << "Conjugate transpose D: " << std::endl << D_ccj << std::endl;
  for (int x = 0; x < D.getShape(0); x++) {
    for (int y = 0; y < D.getShape(1); y++) {
      assert(std::conj(D(y, x)) == D_ct(x, y));
    }
  }

  // 5.11 Inplace operations
  std::cout << bannerS;
  std::cout << "# 5.11. Inplace operations" << std::endl;
  std::cout << bannerE;

  std::cout << "Matrix D: " << std::endl << D << std::endl;

  D.tp_ip();
  std::cout << "D.tp_ip()" << std::endl;
  std::cout << "Matrix D: " << std::endl << D << std::endl;

  D.ccj_ip();
  std::cout << "D.ccj_ip()" << std::endl;
  std::cout << "Matrix D: " << std::endl << D << std::endl;

  D.ct_ip();
  std::cout << "D.ct_ip()" << std::endl;
  std::cout << "Matrix D: " << std::endl << D << std::endl;

  assert(D == D_og);

  // 5.12 Bonus Eigenvalue operations
  std::cout << bannerS;
  std::cout << "# 5.12. Bonus eigenvalue operations" << std::endl;
  std::cout << bannerE;

  // Use the LAPACK reference test:
  // https://www.intel.com/content/www/us/en/docs/onemkl/code-samples-lapack/2022-1/lapacke-dgeev-example-c-row.html
  double a[] = {-1.01, 0.86, -4.60, 3.31,  -4.81, 3.98,  0.53, -7.04, 5.29,
                3.55,  3.30, 8.26,  -3.89, 8.20,  -1.51, 4.43, 4.96,  -7.66,
                -7.33, 6.18, 7.31,  -6.43, -6.16, 2.47,  5.58};

  Matrix<double> symm(5, 5);

  for (int idx = 0; idx < 25; idx++) {
    symm(idx) = a[idx];
  }

  std::cout << "Matrix: " << std::endl << symm << std::endl;
  std::cout << "Eigenvalues" << std::endl << symm.eigenvalues() << std::endl;
  std::cout << "left Eigenvectors" << std::endl
            << symm.leigenvectors() << std::endl;
  std::cout << "right Eigenvectors" << std::endl
            << symm.reigenvectors() << std::endl;
}
