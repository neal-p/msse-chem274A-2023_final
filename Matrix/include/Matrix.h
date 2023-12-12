#pragma once

extern "C" {
#include <cblas.h>
}
#include <lapacke.h>
#include <algorithm>
#include <complex>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

enum Method { NAIVE, BLAS };

template <typename T>
class Matrix {
 private:
  std::unique_ptr<T[]> a_;

  int x_size_;
  int y_size_;
  int size_;
  Method method_;

  Matrix<T> multiplyNaive(const Matrix<T>& mat) const {
    if (x_size_ != mat.y_size_) {
      throw std::runtime_error("Matrices are incompatible, cannot multiply");
    }

    Matrix<T> prod(y_size_, mat.x_size_);

    int idx = 0;

    for (int lrow = 0; lrow < y_size_; lrow++) {
      for (int rcol = 0; rcol < mat.x_size_; rcol++) {
        T sum = 0;

        for (int i = 0; i < x_size_; i++) {
          sum += at(lrow, i) * mat(i, rcol);
        }

        prod(idx) = sum;
        idx++;
      }
    }
    return prod;
  }

  template <typename U = T>
  Matrix<U> multiplyBlas(const Matrix<U>& mat) const {
    Matrix<U> prod(y_size_, mat.x_size_);

    int m = y_size_;
    int n = mat.x_size_;
    int k = x_size_;
    U* A = new U[size_];
    memcpy(A, a_.get(), sizeof(U) * size_);
    U* B = new U[mat.size_];
    memcpy(B, mat.a_.get(), sizeof(U) * mat.size_);
    U* C = prod.a_.get();

    if constexpr (std::is_same_v<U, float>) {
      U i(1.0);
      U j(0.0);

      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, i, A, k,
                  B, n, j, C, n);
    } else if constexpr (std::is_same_v<U, double>) {
      U i(1.0);
      U j(0.0);

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, i, A, k,
                  B, n, j, C, n);
    } else if constexpr (std::is_same_v<U, std::complex<float>>) {
      U i(1.0, 0.0);
      U j(0.0, 0.0);

      cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, i, A, k,
                  B, n, j, C, n);
    } else if constexpr (std::is_same_v<U, std::complex<double>>) {
      U i(1.0, 0.0);
      U j(0.0, 0.0);

      cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, i, A, k,
                  B, n, j, C, n);
    } else {
      throw std::runtime_error("unsupported BLAS type");
    }

    delete[] A;
    delete[] B;
    return prod;
  }

 public:
  Matrix() : x_size_(0), y_size_(0), size_(0), a_(nullptr) {}
  Matrix(const int rows, const int cols)
      : x_size_(cols), y_size_(rows), size_(rows * cols), method_(BLAS) {
    a_ = std::make_unique<T[]>(x_size_ * y_size_);
  }

  Matrix(const Matrix<T>& m)
      : x_size_(m.x_size_), y_size_(m.y_size_), size_(m.size_) {
    a_ = std::make_unique<T[]>(size_);
    memcpy(a_.get(), m.a_.get(), sizeof(T) * (size_));
  }

  void operator=(const Matrix<T>& m) {
    std::unique_ptr<T[]> new_a = std::make_unique<T[]>(m.size_);
    memcpy(a_.get(), m.a_.get(), sizeof(T) * (m.size_));

    a_ = std::move(new_a);
    x_size_ = m.x_size_;
    y_size_ = m.y_size_;
    size_ = m.size_;
  }

  void setMethod(Method method) { method_ = method; }
  Method getMethod() const { return method_; }

  void reshape(const int rows, const int cols) {
    int new_size = rows * cols;
    std::unique_ptr<T[]> new_a = std::make_unique<T[]>(new_size);

    // Copy what elements do fit into the new array
    int to_copy = std::min(new_size, size_);
    memcpy(new_a.get(), a_.get(), sizeof(T) * to_copy);

    a_ = std::move(new_a);
    x_size_ = cols;
    y_size_ = rows;
    size_ = new_size;
  }

  void reshapeLike(const Matrix<T>& mat) { reshape(mat.y_size_, mat.x_size_); }

  inline int getShape(int axis) const {
    if (axis == 0) {
      return x_size_;
    } else if (axis == 1) {
      return y_size_;
    } else if (axis == -1) {
      return size_;
    } else {
      throw std::out_of_range("invalid axis");
    }
  }

  inline T& at(const int y, const int x) {
    if (x < 0 || x >= x_size_) {
      throw std::out_of_range("invalid x dimension");
    }
    if (y < 0 || y >= y_size_) {
      throw std::out_of_range("invalid y dimension");
    }

    return a_[(x_size_ * y) + x];
  }

  inline T at(const int y, const int x) const {
    if (x < 0 || x >= x_size_) {
      throw std::out_of_range("invalid x dimension");
    }
    if (y < 0 || y >= y_size_) {
      throw std::out_of_range("invalid y dimension");
    }

    return a_[(x_size_ * y) + x];
  }

  inline T& at(const int idx) {
    if (idx < 0 || idx >= size_) {
      throw std::out_of_range("invalid idx");
    }
    return a_[idx];
  }

  inline T at(const int idx) const {
    if (idx < 0 || idx >= size_) {
      throw std::out_of_range("invalid idx");
    }
    return a_[idx];
  }

  T& operator()(const int y, const int x) { return at(y, x); }
  T operator()(const int y, const int x) const { return at(y, x); }

  T& operator()(const int idx) { return at(idx); }
  T operator()(const int idx) const { return at(idx); }

  Matrix<T> tp() const {
    Matrix<T> transpose(x_size_, y_size_);

    for (int y = 0; y < y_size_; y++) {
      for (int x = 0; x < x_size_; x++) {
        transpose.a_[(x * y_size_) + y] = a_[(y * x_size_) + x];
      }
    }

    return transpose;
  }

  void tp_ip() {
    Matrix<T> tmp = tp();
    reshapeLike(tmp);
    a_ = std::move(tmp.a_);
  }

  Matrix<T> ccj() const {
    Matrix<T> ccj(y_size_, x_size_);

    for (int idx = 0; idx < size_; idx++) {
      ccj.a_[idx] = std::conj(a_[idx]);
    }

    return ccj;
  }

  void ccj_ip() {
    Matrix<T> tmp = ccj();
    a_ = std::move(tmp.a_);
  }

  Matrix<T> ct() const { return ccj().tp(); }

  void ct_ip() {
    Matrix<T> tmp = ct();
    reshapeLike(tmp);
    a_ = std::move(tmp.a_);
  }

  void fill(const T value) { std::fill(a_.get(), a_.get() + size_, value); }
  void operator=(const T value) { fill(value); }
  void print() const {
    std::cout << "[";

    for (int y = 0; y < y_size_; y++) {
      for (int x = 0; x < x_size_; x++) {
        std::cout << at(y, x) << ", ";
      }
      std::cout << std::endl;
    }

    std::cout << "]" << std::endl;
  }

  Matrix<T> operator+(const Matrix<T>& m) const {
    if (m.x_size_ != x_size_ || m.y_size_ != y_size_) {
      throw std::runtime_error("Cannot add matrices of different sizes!");
    }

    Matrix<T> add(x_size_, y_size_);
    for (int idx = 0; idx < size_; idx++) {
      add(idx) = m(idx) + a_[idx];
    }
    return add;
  }

  Matrix<T> multiplyElements(const Matrix<T>& m) const {
    if (m.x_size_ != x_size_ || m.y_size_ != y_size_) {
      throw std::runtime_error("Cannot add matrices of different sizes!");
    }

    Matrix<T> prod(x_size_, y_size_);
    for (int idx = 0; idx < size_; idx++) {
      prod(idx) = m(idx) * a_[idx];
    }
    return prod;
  }

  Matrix<T> operator*(const T value) const {
    Matrix<T> prod(x_size_, y_size_);
    for (int idx = 0; idx < size_; idx++) {
      prod(idx) = value * a_[idx];
    }
    return prod;
  }

  Matrix<T> operator*(const Matrix<T>& mat) const {
    if (x_size_ != mat.y_size_) {
      throw std::runtime_error("Matrices are incompatible, cannot multiply");
    }

    Matrix<T> prod(y_size_, mat.x_size_);

    switch (method_) {
      case NAIVE: {
        return multiplyNaive(mat);
      } break;

      case BLAS: {
        return multiplyBlas(mat);
      } break;

      default:
        throw std::runtime_error(
            "matrix type not supported by BLAS, use naive method_");
        break;
    }

    return prod;
  }

  bool operator==(const Matrix<T>& mat) const {
    // check low-cost first before the O(N) every-element check
    if (x_size_ != mat.x_size_) {
      return false;
    } else if (y_size_ != mat.y_size) {
      return false;
    }

    return std::equal(a_, a_ + size_, mat.a_);
  }

  template <typename U = T>
  Matrix<std::complex<U>> eigenvalues() {
    Matrix<std::complex<U>> ev(1, x_size_);

    U* wr = new U[x_size_];
    U* wi = new U[x_size_];

    U* vl = nullptr;
    U* vr = nullptr;

    int n = x_size_;
    int lda = x_size_;
    int ldvl = x_size_;
    int ldvr = x_size_;

    U* A = new U[size_];
    memcpy(A, a_.get(), sizeof(U) * size_);

    int info;

    if constexpr (std::is_same_v<U, float>) {
      info = LAPACKE_sgeev(LAPACK_ROW_MAJOR, 'N', 'N', n, A, lda, wr, wi, vl,
                           ldvl, vr, ldvr);
    } else if constexpr (std::is_same_v<U, double>) {
      info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', n, A, lda, wr, wi, vl,
                           ldvl, vr, ldvr);
    } else {
      throw std::runtime_error("unsupported BLAS type");
    }

    if (info != 0) {
      throw std::runtime_error("could not compute eigenvalues for matrix");
    }

    // combine real and imaginary components
    for (int idx = 0; idx < x_size_; idx++) {
      ev(idx) = std::complex<U>(wr[idx], wi[idx]);
    }

    delete[] A;
    delete[] wr;
    delete[] wi;

    return ev;
  }

  template <typename U = T>
  Matrix<std::complex<U>> reigenvectors() {
    Matrix<std::complex<U>> ev(x_size_, x_size_);

    U* wr = new U[x_size_];
    U* wi = new U[x_size_];

    U* vl = nullptr;
    U* vr = new U[x_size_ * x_size_];

    int n = x_size_;
    int lda = x_size_;
    int ldvl = x_size_;
    int ldvr = x_size_;

    U* A = new U[size_];
    memcpy(A, a_.get(), sizeof(U) * size_);

    int info;

    if constexpr (std::is_same_v<U, float>) {
      info = LAPACKE_sgeev(LAPACK_ROW_MAJOR, 'N', 'V', n, A, lda, wr, wi, vl,
                           ldvl, vr, ldvr);
    } else if constexpr (std::is_same_v<U, double>) {
      info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', n, A, lda, wr, wi, vl,
                           ldvl, vr, ldvr);
    } else {
      throw std::runtime_error("unsupported BLAS type");
    }

    if (info != 0) {
      throw std::runtime_error("could not compute eigenvalues for matrix");
    }

    // Real and Complex components are contiguous in memory
    // must check eigenvalue to see if complex
    // see
    // https://www.intel.com/content/www/us/en/docs/onemkl/code-samples-lapack/2022-1/lapacke-dgeev-example-c-row.html
    int idx = 0;
    for (int i = 0; i < x_size_; i++) {
      int j = 0;
      while (j < x_size_) {
        if (wi[j] == (U)0.0) {
          ev(idx) = std::complex<U>(vr[i * ldvr + j], 0.0);
          idx++;
          j++;
        } else {
          ev(idx) = std::complex<U>(vr[i * ldvr + j], vr[i * ldvr + (j + 1)]);
          idx++;
          ev(idx) = std::complex<U>(vr[i * ldvr + j], -vr[i * ldvr + (j + 1)]);
          idx++;
          j += 2;
        }
      }
    }

    delete[] A;
    delete[] wr;
    delete[] wi;
    delete[] vr;

    return ev;
  }

  template <typename U = T>
  Matrix<std::complex<U>> leigenvectors() {
    Matrix<std::complex<U>> ev(x_size_, x_size_);

    U* wr = new U[x_size_];
    U* wi = new U[x_size_];

    U* vl = new U[x_size_ * x_size_];
    U* vr = nullptr;

    int n = x_size_;
    int lda = x_size_;
    int ldvl = x_size_;
    int ldvr = x_size_;

    U* A = new U[size_];
    memcpy(A, a_.get(), sizeof(U) * size_);

    int info;

    if constexpr (std::is_same_v<U, float>) {
      info = LAPACKE_sgeev(LAPACK_ROW_MAJOR, 'V', 'N', n, A, lda, wr, wi, vl,
                           ldvl, vr, ldvr);
    } else if constexpr (std::is_same_v<U, double>) {
      info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'V', 'N', n, A, lda, wr, wi, vl,
                           ldvl, vr, ldvr);
    } else {
      throw std::runtime_error("unsupported BLAS type");
    }

    if (info != 0) {
      throw std::runtime_error("could not compute eigenvalues for matrix");
    }

    // Real and Complex components are contiguous in memory
    // must check eigenvalue to see if complex
    // see
    // https://www.intel.com/content/www/us/en/docs/onemkl/code-samples-lapack/2022-1/lapacke-dgeev-example-c-row.html
    int idx = 0;
    for (int i = 0; i < x_size_; i++) {
      int j = 0;
      while (j < x_size_) {
        if (wi[j] == (U)0.0) {
          ev(idx) = std::complex<U>(vl[i * ldvl + j], 0.0);
          idx++;
          j++;
        } else {
          ev(idx) = std::complex<U>(vl[i * ldvl + j], vl[i * ldvl + (j + 1)]);
          idx++;
          ev(idx) = std::complex<U>(vl[i * ldvl + j], -vl[i * ldvl + (j + 1)]);
          idx++;
          j += 2;
        }
      }
    }

    delete[] A;
    delete[] wr;
    delete[] wi;
    delete[] vl;

    return ev;
  }

  template <typename U = T>
  std::vector<Matrix<std::complex<U>>> alleigen() {
    Matrix<std::complex<U>> ev(1, x_size_);
    Matrix<std::complex<U>> lev(x_size_, x_size_);
    Matrix<std::complex<U>> rev(x_size_, x_size_);

    U* wr = new U[x_size_];
    U* wi = new U[x_size_];

    U* vl = new U[x_size_ * x_size_];
    U* vr = new U[x_size_ * x_size_];

    int n = x_size_;
    int lda = x_size_;
    int ldvl = x_size_;
    int ldvr = x_size_;

    U* A = new U[size_];
    memcpy(A, a_.get(), sizeof(U) * size_);

    int info;

    if constexpr (std::is_same_v<U, float>) {
      info = LAPACKE_sgeev(LAPACK_ROW_MAJOR, 'V', 'V', n, A, lda, wr, wi, vl,
                           ldvl, vr, ldvr);
    } else if constexpr (std::is_same_v<U, double>) {
      info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'V', 'V', n, A, lda, wr, wi, vl,
                           ldvl, vr, ldvr);
    } else {
      throw std::runtime_error("unsupported BLAS type");
    }

    if (info != 0) {
      throw std::runtime_error("could not compute eigenvalues for matrix");
    }

    for (int idx = 0; idx < x_size_; idx++) {
      std::cout << "Wr: " << wr[idx] << " Wi: " << wi[idx] << " -> "
                << std::complex<U>(wr[idx], wi[idx]) << std::endl;
    }

    // combine real and imaginary components
    for (int idx = 0; idx < x_size_; idx++) {
      ev(idx) = std::complex<U>(wr[idx], wi[idx]);
    }

    std::cout << "print ev" << std::endl;
    ev.print();

    // Real and Complex components are contiguous in memory
    // must check eigenvalue to see if complex
    // see
    // https://www.intel.com/content/www/us/en/docs/onemkl/code-samples-lapack/2022-1/lapacke-dgelev-example-c-row.html
    int idx = 0;
    for (int i = 0; i < x_size_; i++) {
      int j = 0;
      while (j < x_size_) {
        if (wi[j] == (U)0.0) {
          lev(idx) = std::complex<U>(vl[i * ldvl + j], 0.0);
          idx++;
          j++;
        } else {
          lev(idx) = std::complex<U>(vl[i * ldvl + j], vl[i * ldvl + (j + 1)]);
          idx++;
          lev(idx) = std::complex<U>(vl[i * ldvl + j], -vl[i * ldvl + (j + 1)]);
          idx++;
          j += 2;
        }
      }
    }

    idx = 0;
    for (int i = 0; i < x_size_; i++) {
      int j = 0;
      while (j < x_size_) {
        if (wi[j] == (U)0.0) {
          rev(idx) = std::complex<U>(vr[i * ldvr + j], 0.0);
          idx++;
          j++;
        } else {
          rev(idx) = std::complex<U>(vr[i * ldvr + j], vr[i * ldvr + (j + 1)]);
          idx++;
          rev(idx) = std::complex<U>(vr[i * ldvr + j], -vr[i * ldvr + (j + 1)]);
          idx++;
          j += 2;
        }
      }
    }

    delete[] A;
    delete[] wr;
    delete[] wi;
    delete[] vl;
    delete[] vr;

    std::cout << "print ev after others" << std::endl;
    ev.print();

    return std::vector<Matrix<std::complex<U>>>{ev, lev, rev};
  }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& m) {
  m.print();
  return os;
}
