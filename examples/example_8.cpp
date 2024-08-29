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
#include <random>
#include <sstream>
#include <vector>

uint16_t count_words(std::string &str) {
  std::replace(str.begin(), str.end(), ',', ' ');
  std::stringstream stream(str);
  return std::distance(std::istream_iterator<std::string>(stream),
                       std::istream_iterator<std::string>());
}

std::vector<std::vector<float>> load_csv(const std::string &path) {
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

void normalize(std::vector<std::vector<float>> &data) {
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

int main(int argc, char *argv[]) {
  auto rng = std::default_random_engine{};

  std::vector<std::vector<float>> data = load_csv("diabetes.csv");
  assert(data.size() > 0);
  normalize(data);

  std::cout << data.size() << " " << data.at(0).size() << '\n';

  const uint16_t third_size = data.size() / 3;
  const uint16_t nvars = data.at(0).size() - 1;

  std::shuffle(std::begin(data), std::end(data), rng);
  std::vector<std::vector<float>> test_data(data.begin(), data.begin() + third_size);
  std::vector<std::vector<float>> train_data(data.begin() + third_size, data.end());

  float alpha = 0.01f;
  std::vector<epg::Scalar> w(nvars + 1);
  for (int i = 0; i < nvars + 1; i++) {
    w[i] = epg::make_variable();
  }

  std::vector<epg::Scalar> x(nvars);
  for (int i = 0; i < nvars; i++) {
    x[i] = epg::make_const();
  }

  epg::Scalar fwd = epg::make_variable();
  for (int i = 0; i < nvars; i++) {
    fwd = fwd + x[i] * w[i];
  }
  fwd = epg::sigmoid(fwd + w[nvars]);

  for (int iter = 0; iter < 10; iter++) {
    float total_loss = 0.0f;
    for (int sample = 0; sample < train_data.size(); sample++) {
      for (int v = 0; v < nvars; v++) {
        x[v]->value = train_data[sample][v];
      }

      epg::Scalar loss = train_data[sample][nvars] * epg::log(fwd)
                       + (1.0f - train_data[sample][nvars]) * epg::log(1.0f - fwd);

      zero_grad(loss);
      eval(loss);
      diff(loss);

      assert(loss->value == loss->value);
      total_loss = total_loss + loss->value;

      for (int v = 0; v < nvars + 1; v++) {
        w[v]->value = w[v]->value + alpha * w[v]->grad;
      }
    }
    std::cout << "total_loss = " << total_loss << std::endl;
  }

  int err = 0;
  for (int sample = 0; sample < test_data.size(); sample++) {
    for (int v = 0; v < nvars; v++) {
      x[v]->value = test_data[sample][v];
    }
    eval(fwd);

    err = err + std::fabs(((float)(fwd->value >= 0.55)) - test_data[sample][nvars]);
  }
  std::cout << "accuracy = "
            << 100.0f * (1.0f - ((float)err) / ((float)test_data.size()))
            << std::endl;

  return 0;
}
