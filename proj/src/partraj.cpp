#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <complex>
#include <cstdint>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace std::string_literals;

void write_image( cv::Mat const & image, std::string const & filename )
{
  auto dir = "../data/"s;
  boost::filesystem::create_directories(dir);
  cv::imwrite( dir + filename, image );
}

int main()
{
  cv::Mat image = cv::Mat::zeros(100, 100, CV_8UC3);
  write_image( image, "hoge.png" );
}