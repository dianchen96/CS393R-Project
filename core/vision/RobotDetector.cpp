#include <vision/RobotDetector.h>
#include <memory/TextLogger.h>
#include <vision/Logging.h>
#include <VisionCore.h>

// #define IS_RUNNING_CORE (core_ && core_->vision_ && ((UTMainWnd*)parent_)->runCoreRadio->isChecked())

RobotDetector::RobotDetector(DETECTOR_DECLARE_ARGS) : DETECTOR_INITIALIZE {
	// Load shape banks
	shape_bank = new cv::Mat[NUM_SHAPE_BANK];

	for (int i = 0; i < NUM_SHAPE_BANK; i++) {
		std::ostringstream filename;
		if (std::getenv("NAO_HOME")) {
			filename << std::getenv("NAO_HOME") << "/shape_bank/masks/" << i << ".png";
		} else {
			filename << "/home/nao/shape_bank/masks/" << i << ".png";
		}

		shape_bank[i] = cv::imread(filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
	}

	std::cout << "Loaded shape back mask" << endl;
}

unsigned char* RobotDetector::getSegImg(){
	if(camera_ == Camera::TOP)
		return vblocks_.robot_vision->getSegImgTop();
	return vblocks_.robot_vision->getSegImgBottom();
}


float RobotDetector::getIoU(cv::Mat& mask1, cv::Mat& mask2) {

	cv::Mat _intersection;
	cv::Mat _union;

	// cout << "mask 1: (" << mask1.cols << " " << mask1.rows << ")" << endl;
	// cout << "mask 2: (" << mask2.cols << " " << mask2.rows << ")" << endl;


	cv::bitwise_and(mask1, mask2, _intersection);
	cv::bitwise_or(mask1, mask2, _union);

	float iou = cv::countNonZero(_intersection)*1. / cv::countNonZero(_union);

	// std::cout << "IoU: " << iou << endl;

	// if (iou > 0.55) {
	// 	// imshow("candidate", mask1);
	// 	imshow("mask", mask2);
	// 	cv::waitKey(0);
	// }
	
	return iou;

}

void RobotDetector::findRobots(vector<Blob> &blobs) {

	findRobotsByIoU(blobs);

	// TODO: Implement point matching methods

}

WorldObject* RobotDetector::popRobotCandidate(int numMatched) {
	switch (numMatched) {
		case 0:	return &vblocks_.world_object->objects_[WO_ROBOT_1];
				break;
		case 1: return &vblocks_.world_object->objects_[WO_ROBOT_2];
				break;
		case 2: return &vblocks_.world_object->objects_[WO_ROBOT_3];
				break;
		case 3: return &vblocks_.world_object->objects_[WO_ROBOT_4];
				break;
		default: return NULL; 
	}
}

int RobotDetector::getPose(int poseNum) {
	if (poseNum <= NUM_FACADE) {
		return 0;
	} else{
		return 1;
	}
}


void RobotDetector::findRobotsByIoU(vector<Blob> &blobs) {

	const float heightRatios[] = {8,8};
	const float widthRatios[] = {6,4.5};
	const int ratioSize = 2;

	// const int dxs[] = {-10,0,10};
	// const int dys[] = {-10,0,10};
	// const int offsetSize = 3;

	if(camera_ == Camera::BOTTOM) return;

	auto whiteBlobs = filterBlobs(blobs, c_WHITE, BLOB_THRESHOLD);

	int numMatched = 0;


	for (int i = 0; i < whiteBlobs.size(); i++) {

		auto blob = whiteBlobs[i];


		// std::cout << blob << endl;
		// std::cout << "====" << endl;

		// Extract image
		int width = 0;
		int height = 0;
		int pose = -1;

		// for (int dxi = 0; dxi < offsetSize; dxi++) {
		// 	for (int dyi = 0; dyi < offsetSize; dyi++) {
		for (int ri = 0; ri < ratioSize; ri++) {
				if (pose >= 0) break;

				// int dx = dxs[dxi];
				// int dy = dys[dyi];
				int widthRatio = widthRatios[ri];
				int heightRatio = heightRatios[ri];

				int xi = max(0., blob.avgX - 0.5 * widthRatio * blob.avgWidth);
				int xf = min(320., blob.avgX + 0.5 * widthRatio * blob.avgWidth);
				int yi = max(0., blob.avgY - 0.5 * heightRatio * blob.avgWidth);
				int yf = min(240., blob.avgY + 0.5 * heightRatio * blob.avgWidth);

				width = xf - xi;
				height = yf - yi;

				// Crop image
				auto image = getSegImg();
				int imageWidth = iparams_.width;
				int imageHeight = iparams_.height;
				cv::Mat cropped_image(height, width, CV_8UC1, cv::Scalar(0));

				for (int x = 0; x < width; x++) {
					for (int y = 0; y < height; y++) {
						int imageX = x + xi;
						int imageY = y + yi;
						auto color = image[imageY * imageWidth + imageX];
						if (color == c_WHITE) {
							cropped_image.at<uchar>(y,x) = 255;
						}
					}
				}

				// Reshape image
				cv::Mat reshaped_image;
				cv::resize(cropped_image, reshaped_image, cv::Size(64,64));

				for (int j = 0; j < NUM_SHAPE_BANK; j++) {
					cv::Mat shape = shape_bank[j];
					float iou = getIoU(reshaped_image, shape);
					if (iou > IOU_THRESHOLD) {
						pose = j;
						break;
					}
				}


				// If matched
				if (pose >= 0 && numMatched < NUM_ROBOTS) {
					// Matched robot
					auto robot = popRobotCandidate(numMatched);

					robot->imageCenterX = blob.avgX;
					robot->imageCenterY = blob.avgY;
					robot->imageHeight = height;
					robot->imageWidth = width;
					robot->seen = true;
					robot->fromTopCamera = (camera_ == Camera::TOP);

					robot->pose = getPose(pose);
					
					// cout << pose << " " << robot->pose << endl;

					numMatched++;

					// std::cout << "Matched, x=" << blob.avgX << " y=" << blob.avgY << endl;
					
				}
			// }
		}


		// // Visualization code for debugging
		// std::ostringstream filename;
		// filename << i;
		// imshow( filename.str(), reshaped_image ); 
		// cv::waitKey(0);
	}

	// cout << "====" << endl;

	// Mark the rest of the robots unseen
	for (int i = numMatched; i < NUM_ROBOTS; i++) {
		auto robot = popRobotCandidate(i);
		robot->seen = false;
	}

}