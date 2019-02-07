#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

using namespace std::string_literals;
using namespace std::complex_literals;
using complex = std::complex<double>;
constexpr double PI = 3.14159'26535'89793'23846'26433'83279'50288;

struct pa {
  complex p;
  complex v;
  pa(complex p_, complex v_) : p(p_), v(v_) {}
};

void write_image(cv::Mat const &image, std::string const &filename) {
  auto dir = "../data/"s;
  boost::filesystem::create_directories(dir);
  cv::imwrite(dir + filename, image);
}

std::vector<std::vector<pa>> generate_particles() {
  constexpr int N = 400;
  std::mt19937 mt(0);
  std::uniform_real_distribution<> dir_rand(0, PI * 2);
  std::uniform_real_distribution<> pos_rand(-1, 1);
  std::vector<std::vector<pa>> r;
  r.reserve(N);
  double v0 = 3e-3;
  for (int i = 0; i < N; ++i) {
    // double t = (2 * PI) * i / N;
    auto d = dir_rand(mt);
    auto x = pos_rand(mt);
    auto y = pos_rand(mt);
    r.push_back(std::vector<pa>{pa(x + y * 1.0i, std::polar(v0, d))});
  }
  return r;
}

void move_particles(std::vector<std::vector<pa>> &parts) {
  int tmax = 2000;
  for (int t = 0; t < tmax; ++t) {
    for (auto &me : parts) {
      complex a = 0;
      for (auto const &o : parts) {
        if (&me == &o) {
          continue;
        }
        auto dist = me.back().p - o.back().p;
        auto power = std::polar(1.0 / (std::abs(dist) + 1e-10), std::arg(dist));
        a -= power;
      }
      auto v = me.back().v;
      me.back().v += a * 1e-6 + std::polar(1e-4, std::arg(v) + PI / 2);
    }
    for (auto &me : parts) {
      auto last = me.back();
      last.p += last.v;
      me.push_back(last);
    }
  }
}

std::uint8_t colu(double t0) {
  double t = std::fmod(std::fmod(t0, 3.0) + 3.0, 3.0);
  if (t < 2) {
    return static_cast<std::uint8_t>(
        std::lround((1 - std::cos(t * PI)) * (255 / 2.0)));
  } else {
    return 0;
  }
}

cv::Vec3b color(double t0, double w) {
  double t = t0;
  return {static_cast<std::uint8_t>(colu(t) * w + 255 * (1 - w)),
          static_cast<std::uint8_t>(colu(t + 1) * w + 255 * (1 - w)),
          static_cast<std::uint8_t>(colu(t + 2) * w + 255 * (1 - w))};
}

void draw(cv::Mat &image, std::vector<std::vector<pa>> const &parts) {
  int w = image.cols;
  int h = image.rows;
  double z = w / 2;
  double xc = w / 2;
  double yc = h / 2;
  auto trans = [z, xc, yc](complex const &p) -> cv::Point2d {
    return cv::Point2d{p.real() * z + xc, p.imag() * z + yc};
  };
  auto last = parts.front().size() - 1;
  for (int t = 0; t < last; ++t) {
    for (size_t ix = 0; ix < parts.size(); ++ix) {
      auto col = color(ix * 3.0 / parts.size(), t * 1.0 / last);
      auto const &o0 = parts[ix][t];
      auto const &o1 = parts[ix][t + 1];
      auto const &p0 = trans(o0.p);
      auto const &p1 = trans(o1.p);
      auto thick = std::min<double>(100, 500 * std::abs(o0.v + o1.v) + 1);
      cv::line(image, p0, p1, col, thick, cv::LINE_AA);
      // cv::circle( image, trans(p.p), 5, col, -1/*fill*/, cv::LINE_AA, 0);
    }
  }
}

int main() {
  auto parts = generate_particles();
  move_particles(parts);
  cv::Mat image = cv::Mat::zeros(3000, 3000, CV_8UC3);
  image = cv::Scalar(255, 255, 255);
  draw(image, parts);
  write_image(image, "hoge.png");
}
