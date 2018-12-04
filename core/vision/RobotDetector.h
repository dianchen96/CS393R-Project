#pragma once

#include <vision/ObjectDetector.h>
#include <common/WorldObject.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define TOP_K 10
#define HEIGHT_RATIO 8
#define WIDTH_RATIO 6

#define NUM_SHAPE_BANK 13


#define BLOB_THRESHOLD 500
#define IOU_THRESHOLD 0.6

// struct RobotCandidate {
// 	unsigned char


// }

class TextLogger;

/// @ingroup vision
class RobotDetector : public ObjectDetector {
 public:
  RobotDetector(DETECTOR_DECLARE_ARGS);
  void init(TextLogger* tl){ textlogger = tl; }
  unsigned char* getSegImg();
  void findRobots(vector<Blob> &blobs);
 private:
  TextLogger* textlogger;
  void findRobotsByIoU(vector<Blob> &blobs);
  cv::Mat* shape_bank;
  float getIoU(cv::Mat& mask1, cv::Mat& mask2);
  WorldObject* popRobotCandidate(int numMatched);
};
