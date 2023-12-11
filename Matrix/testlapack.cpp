#include <cblas.h>
#include <lapacke.h>
#include <iostream>

int main() {
  // Define the matrix A
  double A[] = {2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0};
  int n = 3;  // Size of the matrix A

  // Declare variables for eigenvalues and workspace
  double eigenvalues[n];
  double work[3 * n - 1];
  int info;

  // Call LAPACK function to compute eigenvalues
  info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', n, A, n, eigenvalues);

  // Check for success
  if (info == 0) {
    std::cout << "Eigenvalues: ";
    for (int i = 0; i < n; ++i) {
      std::cout << eigenvalues[i] << " ";
    }
    std::cout << std::endl;
  } else {
    std::cerr << "Error computing eigenvalues (info = " << info << ")"
              << std::endl;
  }

  return 0;
}
