#pragma once

#include <vision/ObjectDetector.h>
#include <common/WorldObject.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define TOP_K 10

#define NUM_FACADE 5
#define NUM_SHAPE_BANK 10

#define BLOB_THRESHOLD 500
#define IOU_THRESHOLD 0.55

#define NUM_ITER_REGION_PROPOSAL
#define STOP_ITER_THRESHOLD

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
  int getPose(int);
  WorldObject* popRobotCandidate(int numMatched);
};
