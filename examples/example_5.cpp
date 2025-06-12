/////////////////////////////////////////////////
//
// This example demonstrates usage of EPGraph for
// a non-linear stiff ODE:
//   dy/dt = -50*(y - cos(t)), y(0)=0
// using an implicit midpoint method
//
// The exact solution to the IVP is:
//   (50/2501)*(sin(t) + 50*cos(t)) - (2500/2501)*exp(-50*t)
//
/////////////////////////////////////////////////

#include <epgraph>
#include <iostream>
#include <fstream>

float y_exact(float t) {
  return (50.0f / 2501.0f) * (sin(t) + 50.0f * cos(t))
         - (2500.0f / 2501.0f) * exp(-50.0f * t);
}

int main(int argc, char* argv[]) {
  using namespace epg;

  const int N = 50;
  const float dt = 1.0f / 25.0f;

  float t0 = 0.0f;
  float y0 = 0.0f;

  std::ofstream file("out_example_5.csv");
  file << "t,y_numeric,y_exact" << std::endl;
  file << t0 << "," << y0 << "," << y_exact(t0) << std::endl;

  for (int n = 0; n < N; n++) {
    Scalar y = y0;
    Scalar F = (y - y0) / dt + 50.0f * (0.5 * (y + y0) - cos(t0 + 0.5 * dt));

    for (int i = 0; i < 8; i++) {
      zero_grad(F);
      eval(F);
      diff(F);
      y = y.get_value() - F.get_value() / y.get_grad();
    }

    y0 = y.get_value();
    t0 = t0 + dt;

    file << t0 << "," << y0 << "," << y_exact(t0) << std::endl;
  }

  return 0;
}
