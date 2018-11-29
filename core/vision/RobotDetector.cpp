#include <vision/BeaconDetector.h>
#include <memory/TextLogger.h>
#include <vision/Logging.h>

RobotDetector::RobotDetector(DETECTOR_DECLARE_ARGS) : DETECTOR_INITIALIZE {
}

unsigned char* BeaconDetector::getSegImg(){
    if(camera_ == Camera::TOP)
        return vblocks_.robot_vision->getSegImgTop();
    return vblocks_.robot_vision->getSegImgBottom();
}

void BeaconDetector::findBeacons(vector<Blob> &blobs) {

	

}