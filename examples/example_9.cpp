///////////////////////////////////////////////////////////////////////
//
// This example demonstrates usage of EPGraph coupled with Eigen
//
///////////////////////////////////////////////////////////////////////

#include <epgraph>
#include <iostream>
#include <vector>
#include <Eigen/Core>

int main() {
  typedef Eigen::Matrix<epg::Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix_epg;

  using namespace epg;
  int M = 4;
  int N = 4;
  int K = 4;
  Matrix_epg A_eigen = Matrix_epg::Random(N, K);
  Matrix_epg X_eigen = Matrix_epg::Random(K, M);
  Matrix_epg b_eigen = Matrix_epg::Random(1, N);
  Matrix_epg C_eigen = A_eigen * X_eigen;

  for(auto c = C_eigen.data(); c < C_eigen.data() + C_eigen.size(); c++) {
    zero_grad(*c);
    eval(*c);
  }

  for(auto c = C_eigen.data(); c < C_eigen.data() + C_eigen.size(); c++) {
    diff(*c);
  }

  for(auto c = C_eigen.data(); c < C_eigen.data() + C_eigen.size(); c++) {
    std::cout<<*c<<'\n';
  }

  for(auto c = A_eigen.data(); c < A_eigen.data() + A_eigen.size(); c++) {
    std::cout<<*c<<'\n';
  }
}
