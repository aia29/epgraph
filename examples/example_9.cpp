///////////////////////////////////////////////////////////////////////
//
// This example demonstrates usage of EPGraph coupled with Eigen
//
///////////////////////////////////////////////////////////////////////

#include <epgraph>
#include <iostream>
#include <vector>
#include <Eigen/Core>

float randu(const float a, const float b) {
  return (((float)rand()) / (RAND_MAX + 1.0f)) * (b - a) + a;
}

void fill(std::vector<epg::Scalar> &s) {
  for(auto i=0; i<s.size(); i++) {
    s[i] = randu(-10.0f, 10.0f);
  }
}

void copy(float *s_basic, const std::vector<epg::Scalar> &s_epg) {
  for(auto i=0; i<s_epg.size(); i++) {
    s_basic[i] = s_epg[i].get_value();
  }
}

void print_grad(const std::vector<epg::Scalar> &s_epg, const float *s_basic) {
  for(auto i=0; i<s_epg.size(); i++) {
    std::cout<<s_epg[i].get_grad()<<" "<<s_basic[i]<<'\n';
  }
}

void print_value(const std::vector<epg::Scalar> &s_epg, const float *s_basic) {
  for(auto i=0; i<s_epg.size(); i++) {
    std::cout<<s_epg[i].get_value()<<" "<<s_basic[i]<<'\n';
  }
}

int main() {
  using namespace epg;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix_epg;
  typedef Eigen::Map<Matrix_epg> epgMap;

  int M = 16;
  int N = 4;
  int P = 6;
  int Q = 12;

  std::vector<Scalar> A_vector(N * M);
  std::vector<Scalar> X_vector(N * P);
  std::vector<Scalar> B_vector(M * P);
  std::vector<Scalar> W_vector(Q * M);
  std::vector<Scalar> Z_vector(Q * P);

  fill(A_vector);
  fill(X_vector);
  fill(B_vector);
  fill(W_vector);
  fill(Z_vector);

  epgMap A_eigen = epgMap(A_vector.data(), N, M);
  epgMap X_eigen = epgMap(X_vector.data(), N, P);
  epgMap B_eigen = epgMap(B_vector.data(), M, P);
  epgMap W_eigen = epgMap(W_vector.data(), Q, M);
  epgMap Z_eigen = epgMap(Z_vector.data(), Q, P);

  Z_eigen = W_eigen * (A_eigen.transpose() * X_eigen  + B_eigen);

  for(auto c = Z_eigen.data(); c < Z_eigen.data() + Z_eigen.size(); c++) {
    zero_grad(*c);
    eval(*c);
  }

  for(auto c = Z_eigen.data(); c < Z_eigen.data() + Z_eigen.size(); c++) {
    diff(*c);
  }

  Eigen::MatrixXf A_float = Eigen::MatrixXf::Zero(N, M);
  Eigen::MatrixXf X_float = Eigen::MatrixXf::Zero(N, P);
  Eigen::MatrixXf B_float = Eigen::MatrixXf::Zero(M, P);
  Eigen::MatrixXf W_float = Eigen::MatrixXf::Zero(Q, M);

  copy(A_float.data(), A_vector);
  copy(X_float.data(), X_vector);
  copy(B_float.data(), B_vector);
  copy(W_float.data(), W_vector);

  Eigen::MatrixXf Z_float = W_float * (A_float.transpose() * X_float + B_float);

  print_value(Z_vector, Z_float.data());

  // TODO: Need to check differentiation. I have a suspicion,
  //       than the differentiation in wrong.
  //       Use https://www.matrixcalculus.org/
}
