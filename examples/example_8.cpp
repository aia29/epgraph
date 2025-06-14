///////////////////////////////////////////////////////////////////////
//
// This example demonstrates usage of EPGraph for
// a simple logistic regression
//   f = sigmoid(sum(w*x) + b)
// on Pima Diabetes Database.
// https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
//
///////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <epgraph>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <vector>

float randu(const float a, const float b) {
  return (((float)rand()) / (RAND_MAX + 1.0f)) * (b - a) + a;
}

uint16_t count_words(std::string& str) {
  std::replace(str.begin(), str.end(), ',', ' ');
  std::stringstream stream(str);
  return std::distance(
      std::istream_iterator<std::string>(stream),
      std::istream_iterator<std::string>());
}

std::vector<std::vector<float>> load_csv(const std::string& path) {
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<std::vector<float>> values;
  uint16_t rows = 0;
  uint16_t cols = 0;
  std::getline(indata, line);
  cols = count_words(line);

  while (std::getline(indata, line)) {
    uint16_t col = 0;
    std::vector<float> tmp(cols, 0.0);
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      assert(col < cols);
      tmp[col] = std::stof(cell);
      col++;
    }
    assert(col == cols);
    ++rows;
    values.push_back(tmp);
  }
  return values;
}

void normalize(std::vector<std::vector<float>>& data) {
  assert(data.size() > 0);
  const int Ndata = data.size();
  const int Nvar = data.at(0).size() - 1;
  std::vector<float> std_dev(Nvar, 0.0f);
  std::vector<float> mean(Nvar, 0.0f);

  for (int sample = 0; sample < Ndata; sample++) {
    for (int v = 0; v < Nvar; v++) {
      mean[v] = mean[v] + data[sample][v];
      std_dev[v] = std_dev[v] + data[sample][v] * data[sample][v];
    }
  }

  for (int v = 0; v < Nvar; v++) {
    mean[v] = mean[v] / Ndata;
    std_dev[v] = std::sqrt(std_dev[v] / Ndata - mean[v] * mean[v]);
  }

  for (int sample = 0; sample < Ndata; sample++) {
    for (int v = 0; v < Nvar; v++) {
      data[sample][v] = (data[sample][v] - mean[v]) / std_dev[v];
    }
  }
}

epg::Scalar linear_layer(
    const std::vector<epg::Scalar>& w,
    const std::vector<float>& x) {
  const int nvars = x.size();
  epg::Scalar fwd = 0.0f;
  for (int i = 0; i < nvars; i++) {
    fwd = fwd + x[i] * w[i];
  }
  return fwd + w[nvars];
}

int main(int argc, char* argv[]) {
  auto rng = std::default_random_engine{};

  std::vector<std::vector<float>> data = load_csv("diabetes.csv");
  assert(data.size() > 0);
  normalize(data);

  std::cout << data.size() << " " << data.at(0).size() << std::endl;

  const uint16_t third_size = data.size() / 3;
  const uint16_t nvars = data.at(0).size() - 1;

  std::shuffle(std::begin(data), std::end(data), rng);
  std::vector<std::vector<float>> test_data(
      data.begin(), data.begin() + third_size);
  std::vector<std::vector<float>> train_data(
      data.begin() + third_size, data.end());

  float alpha = 0.005f;
  std::vector<float> x(nvars, 0.0f);
  std::vector<epg::Scalar> w(nvars + 1);
  for (int i = 0; i < nvars + 1; i++) {
    w[i] = randu(-1.0f, 1.0f);
  }

  for (int iter = 0; iter < 25; iter++) {
    epg::Scalar loss = 0.0f;
    for (int sample = 0; sample < train_data.size(); sample++) {
      for (int v = 0; v < nvars; v++) {
        x[v] = train_data[sample][v];
      }
      const float yk = train_data[sample][nvars];
      epg::Scalar fwd = epg::sigmoid(linear_layer(w, x));
      loss = loss - yk * epg::log(fwd) - (1.0f - yk) * epg::log(1.0f - fwd);
    }
    assert(loss.get_value() == loss.get_value());
    zero_grad(loss);
    eval(loss);
    diff(loss);

    for (int v = 0; v < nvars + 1; v++) {
      w[v] = w[v].get_value() - alpha * w[v].get_grad();
    }
    std::cout << "loss = " << loss.get_value() << std::endl;
  }

  int TP = 0;
  int TN = 0;
  int FP = 0;
  int FN = 0;
  for (int sample = 0; sample < test_data.size(); sample++) {
    for (int v = 0; v < nvars; v++) {
      x[v] = test_data[sample][v];
    }

    const float yk = test_data[sample][nvars];
    epg::Scalar fwd = epg::sigmoid(linear_layer(w, x));
    eval(fwd);
    bool prediction = (fwd.get_value() >= 0.45);
    bool truth = (yk >= 0.45);

    TP = TP + ((prediction) && (truth));
    TN = TN + ((!prediction) && (!truth));
    FP = FP + ((prediction) && (!truth));
    FN = FN + ((!prediction) && (truth));
  }

  std::cout << "TP = " << TP << std::endl;
  std::cout << "TN = " << TN << std::endl;
  std::cout << "FP = " << FP << std::endl;
  std::cout << "FN = " << FN << std::endl;
  std::cout << "out of = " << test_data.size() << " samples." << std::endl;

  return 0;
}
