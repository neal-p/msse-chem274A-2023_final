#include "Matrix.h"

int main() {
  Matrix<float> m(10, 10);
  m.at(0, 0) = 1;
  m.at(9, 9) = 2;

  std::cout << m << std::endl;
  m.reshape(2, 4);
  std::cout << m << std::endl;

  m = 4.0;
  m.print();

  m.reshape(10, 20);

  for (int i = 0; i < 10; i++) {
    m(i, 10) = i;
  }

  m.print();
  m.tp().print();

  m.reshape(2, 3);
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(0, 2) = 3;
  m(1, 0) = 4;
  m(1, 1) = 5;
  m(1, 2) = 6;

  m.print();

  Matrix<float> m2(3, 2);
  m2(0, 0) = 7;
  m2(0, 1) = 8;
  m2(1, 0) = 9;
  m2(1, 1) = 10;
  m2(2, 0) = 11;
  m2(2, 1) = 12;

  m2.print();

  auto prod = m * m2;

  prod.print();

  m.setMethod(NAIVE);

  auto prod2 = m * m2;

  prod2.print();

  std::cout << "transposed" << std::endl;

  prod2.tp().print();

  Matrix<std::complex<float>> complex(2, 3);

  std::complex<float> fillvalue(1.0, -1);

  complex.fill(fillvalue);

  complex(1, 1) = std::complex<float>(5, 2.4);

  complex.print();

  std::cout << "conjugate" << std::endl;
  complex.ccj().print();

  Matrix<float> symm(2, 2);

  symm(0, 0) = -5;
  symm(0, 1) = 2;
  symm(1, 0) = -7;
  symm(1, 1) = 4;

  symm.print();

  std::cout << "eigen values" << std::endl;
  symm.eigenvalues().print();

  Matrix<float> symm2(2, 2);

  symm2(0, 0) = 4;
  symm2(0, 1) = 2;
  symm2(1, 0) = 1;
  symm2(1, 1) = 3;

  symm2.print();
  symm2.eigenvalues().print();

  // symm2.eigenvectors().print();

  std::cout << "      FROM DOCS     " << std::endl;

  double a[] = {-1.01, 0.86, -4.60, 3.31,  -4.81, 3.98,  0.53, -7.04, 5.29,
                3.55,  3.30, 8.26,  -3.89, 8.20,  -1.51, 4.43, 4.96,  -7.66,
                -7.33, 6.18, 7.31,  -6.43, -6.16, 2.47,  5.58};

  Matrix<double> symm3(5, 5);

  for (int idx = 0; idx < 25; idx++) {
    symm3(idx) = a[idx];
  }

  symm3.print();

  std::cout << "eigenvalues" << std::endl;
  symm3.eigenvalues().print();

  std::cout << std::endl << "l eigen vectors: " << std::endl;
  symm3.leigenvectors().print();

  std::cout << std::endl << "r eigen vectors: " << std::endl;
  symm3.reigenvectors().print();

  std::cout << "from alleigen" << std::endl << std::endl;

  auto eigens = symm3.alleigen();

  std::cout << "eigenvalues: " << std::endl;
  eigens[0].print();
  std::cout << std::endl << "l eigen vectors: " << std::endl;
  eigens[1].print();
  std::cout << std::endl << "r eigen vectors: " << std::endl;
  eigens[2].print();

  std::cout << std::endl;
  std::cout << "oriignal array: " << std::endl;
  symm3.print();

  return 0;
}
