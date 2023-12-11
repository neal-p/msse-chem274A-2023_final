#include <stdio.h>
#include <stdlib.h>
#include "lapacke.h"

/* Auxiliary routines prototypes */
extern void print_eigenvalues(char* desc, int n, double* wr, double* wi);
extern void print_eigenvectors(char* desc,
                               int n,
                               double* wi,
                               double* v,
                               int ldv);

/* Parameters */
#define N 5
#define LDA N
#define LDVL N
#define LDVR N

/* Main program */
int main() {
  /* Locals */
  int n = N, lda = LDA, ldvl = LDVL, ldvr = LDVR, info;
  /* Local arrays */
  double wr[N], wi[N], vl[LDVL * N], vr[LDVR * N];
  double a[LDA * N] = {-1.01, 0.86,  -4.60, 3.31,  -4.81, 3.98,  0.53,
                       -7.04, 5.29,  3.55,  3.30,  8.26,  -3.89, 8.20,
                       -1.51, 4.43,  4.96,  -7.66, -7.33, 6.18,  7.31,
                       -6.43, -6.16, 2.47,  5.58};
  /* Executable statements */
  printf("LAPACKE_dgeev (row-major, high-level) Example Program Results\n");
  /* Solve eigenproblem */
  info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'V', 'V', n, a, lda, wr, wi, vl, ldvl,
                       vr, ldvr);
  /* Check for convergence */
  if (info > 0) {
    printf("The algorithm failed to compute eigenvalues.\n");
    exit(1);
  }
  /* Print eigenvalues */
  print_eigenvalues("Eigenvalues", n, wr, wi);
  /* Print left eigenvectors */
  print_eigenvectors("Left eigenvectors", n, wi, vl, ldvl);
  /* Print right eigenvectors */
  print_eigenvectors("Right eigenvectors", n, wi, vr, ldvr);
  exit(0);
} /* End of LAPACKE_dgeev Example */

/* Auxiliary routine: printing eigenvalues */
void print_eigenvalues(char* desc, int n, double* wr, double* wi) {
  int j;
  printf("\n %s\n", desc);
  for (j = 0; j < n; j++) {
    if (wi[j] == (double)0.0) {
      printf(" %6.2f", wr[j]);
    } else {
      printf(" (%6.2f,%6.2f)", wr[j], wi[j]);
    }
  }
  printf("\n");
}

/* Auxiliary routine: printing eigenvectors */
void print_eigenvectors(char* desc, int n, double* wi, double* v, int ldv) {
  int i, j;
  printf("\n %s\n", desc);
  for (i = 0; i < n; i++) {
    j = 0;
    while (j < n) {
      if (wi[j] == (double)0.0) {
        printf(" %6.2f", v[i * ldv + j]);
        j++;
      } else {
        printf(" (%6.2f,%6.2f)", v[i * ldv + j], v[i * ldv + (j + 1)]);
        printf(" (%6.2f,%6.2f)", v[i * ldv + j], -v[i * ldv + (j + 1)]);
        j += 2;
      }
    }
    printf("\n");
  }
}
