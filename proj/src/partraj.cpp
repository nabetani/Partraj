#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <random>

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
  constexpr int N = 100;
  std::mt19937 mt(0);
  std::uniform_real_distribution<> dir_rand(0,PI*2);
  std::uniform_real_distribution<> pos_rand(-1,1);
  std::vector<std::vector<pa>> r;
  r.reserve(N);
  double v0 = 3e-3;
  for (int i = 0; i < N; ++i) {
    //double t = (2 * PI) * i / N;
    auto d = dir_rand(mt);
    auto x = pos_rand(mt);
    auto y = pos_rand(mt);
    r.push_back(std::vector<pa>{pa(x+y*1.0i, std::polar(v0, d))});
  }
  return r;
}

void move_particles(std::vector<std::vector<pa>> &parts) {
  int tmax = 2000;
  for( int t=0 ; t<tmax ; ++t ){
    for( auto & me : parts ){
      complex a=0;
      for( auto const & o : parts ){
        if (&me==&o){
          continue;
        }
        auto dist = me.back().p - o.back().p;
        auto power = std::polar(1.0/(std::abs(dist)+1e-10), std::arg(dist));
        a-=power;
      }
      me.back().v+=a*1e-6;
    }
    for( auto & me : parts ){
      auto last = me.back();
      last.p += last.v;
      me.push_back(last);
    }
  }
}

void draw( cv::Mat & image, std::vector<std::vector<pa>> const & parts ){
  int w = image.cols;
  int h = image.rows;
  double z = w/2;
  double xc = w/2;
  double yc = h/2;
  auto trans = [z, xc, yc]( complex const & p )->cv::Point2d {
    return cv::Point2d{
      p.real() * z + xc,
      p.imag() * z + yc
    };
  };
  for( auto const & pas : parts ){
    for( auto const & p : pas ){
      cv::circle( image, trans(p.p), 5, cv::Scalar(0,0,0), -1/*fill*/, cv::LINE_AA, 0);
    }
  }
}

int main() {
  auto parts = generate_particles();
  move_particles(parts);
  cv::Mat image = cv::Mat::zeros(1000, 1000, CV_8UC3);
  image = cv::Scalar(255,255,255);
  draw(image, parts);
  write_image(image, "hoge.png");
}
