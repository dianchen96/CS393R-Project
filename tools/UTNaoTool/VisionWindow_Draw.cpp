#include <UTMainWnd.h>
#include <VisionWindow.h>
#include <yuview/YUVImage.h>
#include <tool/Util.h>
#include <vision/Classifier.h>
#include <common/ColorSpaces.h>

#define MIN_PEN_WIDTH 3
#define IS_RUNNING_CORE (core_ && core_->vision_ && ((UTMainWnd*)parent_)->runCoreRadio->isChecked())

int count(0); // global variable


void VisionWindow::redrawImages() {
  if(!enableDraw_) return;

  if (((UTMainWnd*)parent_)->streamRadio->isChecked()) {
    int ms = timer_.elapsed();
    if(ms < MS_BETWEEN_FRAMES)
      return;
    timer_.start();
  }
  setImageSizes();

  redrawImages(rawImageTop,    segImageTop,    objImageTop,    horizontalBlobImageTop,    verticalBlobImageTop,    transformedImageTop);
  redrawImages(rawImageBottom, segImageBottom, objImageBottom, horizontalBlobImageBottom, verticalBlobImageBottom, transformedImageBottom);

  updateBigImage();
}

void VisionWindow::updateBigImage(ImageWidget* source) {
  ImageProcessor* processor = getImageProcessor(source);
  bigImage->setImageSource(source->getImage());
}

void VisionWindow::updateBigImage() {
  switch(currentBigImageType_) {
    case RAW_IMAGE:
      if (currentBigImageCam_ == Camera::TOP)
        updateBigImage(rawImageTop);
      else
        updateBigImage(rawImageBottom);
        break;
    case SEG_IMAGE:
      if (currentBigImageCam_ == Camera::TOP)
        updateBigImage(segImageTop);
      else
        updateBigImage(segImageBottom);
        break;
    case OBJ_IMAGE:
      if (currentBigImageCam_ == Camera::TOP)
        updateBigImage(objImageTop);
      else
        updateBigImage(objImageBottom);
        break;
    case HORIZONTAL_BLOB_IMAGE:
      if (currentBigImageCam_ == Camera::TOP)
        updateBigImage(horizontalBlobImageTop);
      else
        updateBigImage(horizontalBlobImageBottom);
        break;
    case VERTICAL_BLOB_IMAGE:
      if (currentBigImageCam_ == Camera::TOP)
        updateBigImage(verticalBlobImageTop);
      else
        updateBigImage(verticalBlobImageBottom);
        break;
    case TRANSFORMED_IMAGE:
      if (currentBigImageCam_ == Camera::TOP)
        updateBigImage(transformedImageTop);
      else
        updateBigImage(transformedImageBottom);
        break;
  }

  // draw all pixels of seg image in big window
  if (currentBigImageType_ == SEG_IMAGE){
    drawSegmentedImage(bigImage);
    if (cbxOverlay->isChecked()) {
      drawGoal(bigImage);
      drawBall(bigImage);
      drawRobots(bigImage);
      drawBallCands(bigImage);
      drawBeacons(bigImage);
    }
  }

  bigImage->update();

}

void VisionWindow::redrawImages(ImageWidget* rawImage, ImageWidget* segImage, ImageWidget* objImage, ImageWidget* horizontalBlobImage, ImageWidget* verticalBlobImage, ImageWidget* transformedImage) {
  drawRawImage(rawImage);
  drawSmallSegmentedImage(segImage);

  objImage->fill(0);
  drawBall(objImage);

  if(cbxHorizon->isChecked()) {
    drawHorizonLine(rawImage);
    drawHorizonLine(segImage);
    drawHorizonLine(horizontalBlobImage);
    drawHorizonLine(verticalBlobImage);
  }

  // Save image
  auto _rawImage = segImage->getImage();
  std::ostringstream filename;
  filename << std::getenv("NAO_HOME") << "/evaluations/" << ::count << "_seg.jpg";
  ::count++;
  _rawImage->save(QString::fromStdString(filename.str()), "JPEG");


  // if overlay is on, then draw objects on the raw and seg image as well
  if (cbxOverlay->isChecked()) {
    drawGoal(rawImage);
    drawBall(rawImage);
    drawRobots(rawImage);
    drawBallCands(rawImage);
    drawBeacons(rawImage);

    drawGoal(segImage);
    drawBall(segImage);
    drawRobots(segImage);
    drawBallCands(segImage);
    drawBeacons(segImage);
  }

  // ::count++;

  // auto _rawImage = rawImage->getImage();
  // std::ostringstream filename;
  // filename << std::getenv("NAO_HOME") << "/evaluations/" << ::count << "_pred.jpg";
  // ::count++;
  // _rawImage->save(QString::fromStdString(filename.str()), "JPEG");


  drawBall(verticalBlobImage);
  drawBallCands(verticalBlobImage);

  transformedImage->fill(0);

  rawImage->update();
  segImage->update();
  objImage->update();
  horizontalBlobImage->update();
  verticalBlobImage->update();
  transformedImage->update();
}

void VisionWindow::drawRawImage(ImageWidget* widget) {
  ImageProcessor* processor = getImageProcessor(widget);
  unsigned char* image = processor->getImg();
  const ImageParams& iparams = processor->getImageParams();
  const CameraMatrix& cmatrix = processor->getCameraMatrix();
  if (!processor->isImageLoaded()) {
    widget->fill(0);
    return;
  }
  auto yuv = yuview::YUVImage::CreateFromRawBuffer(image, iparams.width, iparams.height);
  auto q = util::yuvToQ(yuv);
  widget->setImageSource(&q);
}

void VisionWindow::drawSmallSegmentedImage(ImageWidget *image) {
  ImageProcessor* processor = getImageProcessor(image);
  const ImageParams& iparams = processor->getImageParams();
  unsigned char* segImg = processor->getSegImg();
  int hstep, vstep;
  processor->color_segmenter_->getStepSize(hstep, vstep);
  if (robot_vision_block_ == NULL || segImg == NULL) {
    image->fill(0);
    return;
  }

  // This will be changed on the basis of the scan line policy
  for (int y = 0; y < iparams.height; y+=vstep) {
    for (int x = 0; x < iparams.width; x+=hstep) {
      int c = segImg[iparams.width * y + x];
      for (int smallY = 0; smallY < vstep; smallY++) {
        for (int smallX = 0; smallX < hstep; smallX++) {
          image->setPixel(x + smallX, y + smallY, segRGB[c]);
        }
      }
    }
  }
}

void VisionWindow::drawSegmentedImage(ImageWidget *image) {
  ImageProcessor* processor = getImageProcessor(image);
  const ImageParams& iparams = processor->getImageParams();
  if (doingClassification_) {
    if (image_block_ == NULL) {
      image->fill(0);
      return;
    }

    // Classify the entire image from the raw image
    unsigned char *rawImg = processor->getImg();
    unsigned char* colorTable = processor->getColorTable();
    const ImageParams& iparams = processor->getImageParams();

    for (uint16_t y = 0; y < iparams.height; y++) {
      for (uint16_t x = 0; x < iparams.width; x++) {
        Color c = ColorTableMethods::xy2color(rawImg, colorTable, x, y, iparams.width);
        image->setPixel(x, y, segRGB[c]);
      }
    }
  }
  else {
    unsigned char* segImg = processor->getSegImg();
    if (robot_vision_block_ == NULL || segImg == NULL) {
      image->fill(0);
      return;
    }

    // Seg image from memory
    for (int y = 0; y < iparams.height; y++) {
      for (int x = 0; x < iparams.width; x++) {
        int c = segImg[iparams.width * y + x];
        image->setPixel(x, y, segRGB[c]);
      }
    }
  }
  if(cbxHorizon->isChecked())
    drawHorizonLine(image);
}

void VisionWindow::drawBall(ImageWidget* image) {
  if(!config_.all) return;
  if(!config_.ball) return;
  QPainter painter(image->getImage());
  painter.setPen(QPen(QColor(0, 255, 127), 3));
  if(IS_RUNNING_CORE) {
    ImageProcessor* processor = getImageProcessor(image);

    BallCandidate* best = processor->getBestBallCandidate();
    if(!best) return;

    int r = best->radius;
    painter.drawEllipse(
      (int)best->centerX - r - 1,
      (int)best->centerY - r - 1, 2 * r + 2, 2 * r + 2);
  }
  else if (world_object_block_ != NULL) {
    WorldObject* ball = &world_object_block_->objects_[WO_BALL];
    if(!ball->seen) return;
    if( (ball->fromTopCamera && _widgetAssignments[image] == Camera::BOTTOM) ||
        (!ball->fromTopCamera && _widgetAssignments[image] == Camera::TOP) ) return;
    int radius = ball->radius;
    painter.drawEllipse(ball->imageCenterX - radius, ball->imageCenterY - radius, radius * 2, radius * 2);
  }
}

void VisionWindow::drawRobots(ImageWidget* image) {
  if(!config_.all) return;
  if(world_object_block_ == NULL) return;

  // auto _image = image->getImage();

  QPainter painter(image->getImage());
  painter.setPen(QPen(QColor(232, 40, 193), 3));

  // Start filestream write

  // ofstream prediction;
  // std::ostringstream filename;
  // filename << std::getenv("NAO_HOME") << "/evaluations/3_060/" << ::count << "_pred.txt";
  // prediction.open (filename.str());

  for (int i = 0; i < NUM_ROBOTS; i++) {
    // std::cout << "Seen " << i << endl;

    auto &robot = world_object_block_->objects_[WO_ROBOT_1 + i];
    if(not robot.seen) return;
    if(robot.fromTopCamera and _widgetAssignments[image] == Camera::BOTTOM) return;
    if(not robot.fromTopCamera and _widgetAssignments[image] == Camera::TOP) return;
    // std::cout << "Seen " << i << endl;
    int width = robot.imageWidth;
    int height = robot.imageHeight;
    int x = robot.imageCenterX - width/2;
    int y = robot.imageCenterY - height/2;

    painter.drawRect(QRect(x, y, width, height));

    if (robot.pose == 0) {
      painter.drawText(QPointF(x + 0.75 * width, y + 0.75 * height), "Facade");
    } else {
      painter.drawText(QPointF(x + 0.75 * width, y + 0.75 * height), "Side");
    }


    // write to file

    // prediction << max(0,x) << "," << max(0,y) << ";";
    // prediction << max(0,x) << "," << min(240,y + height) << ";";
    // prediction << min(320,x + width) << "," << max(0,y) << ";";
    // prediction << min(320,x + width) << "," << min(240,y + height) << ";";
    // prediction << endl;

  }
  // std::cout << "====" << endl;

  // prediction.close();

}

void VisionWindow::drawGoal(ImageWidget* image) {
  if(!config_.all) return;
  if(world_object_block_ == NULL) return;

  auto processor = getImageProcessor(image);
  const auto& cmatrix = processor->getCameraMatrix();
  QPainter painter(image->getImage());
  painter.setRenderHint(QPainter::Antialiasing);

  auto& goal = world_object_block_->objects_[WO_UNKNOWN_GOAL];
  if(not goal.seen) return;
  if(goal.fromTopCamera and _widgetAssignments[image] == Camera::BOTTOM) return;
  if(not goal.fromTopCamera and _widgetAssignments[image] == Camera::TOP) return;

  QPen pen(segCol[c_BLUE]);

  int width = cmatrix.getCameraWidthByDistance(goal.visionDistance, 1000);
  int height = cmatrix.getCameraHeightByDistance(goal.visionDistance, 500);
  int x1 = goal.imageCenterX - width / 2;
  
  // Draw top
  int ty1 = goal.imageCenterY - height;
  QPainterPath path;
  path.addRoundedRect(QRect(x1, ty1, width, height), 5, 5);
  painter.setPen(pen);
  painter.fillPath(path, QBrush(segCol[c_BLUE]));

}

void VisionWindow::drawBallCands(ImageWidget* image) {
}

void VisionWindow::drawHorizonLine(ImageWidget *image) {
  if(!config_.horizon) return;
  if(!config_.all) return;
  if (robot_vision_block_ && _widgetAssignments[image] == Camera::TOP) {
    HorizonLine horizon = robot_vision_block_->horizon;
    if (horizon.exists) {
      QPainter painter(image->getImage());
      QPen wpen = QPen(segCol[c_BLUE], MIN_PEN_WIDTH);
      painter.setPen(wpen);

      ImageProcessor* processor = getImageProcessor(image);
      const ImageParams& iparams = processor->getImageParams();

      int x1 = 0;
      int x2 = iparams.width - 1;
      int y1 = horizon.gradient * x1 + horizon.offset;
      int y2 = horizon.gradient * x2 + horizon.offset;
      painter.drawLine(x1, y1, x2, y2);
    }
  }
}

void VisionWindow::drawWorldObject(ImageWidget* image, QColor color, int worldObjectID) {
  if (world_object_block_ != NULL) {
    QPainter painter(image->getImage());
    QPen wpen = QPen(color, 5);   // 2
    painter.setPen(wpen);
    WorldObject* object = &world_object_block_->objects_[worldObjectID];
    if(!object->seen) return;
    if( (object->fromTopCamera && _widgetAssignments[image] == Camera::BOTTOM) ||
        (!object->fromTopCamera && _widgetAssignments[image] == Camera::TOP) ) return;
    int offset = 10;      // 5
    int x1, y1, x2, y2;

    x1 = object->imageCenterX - offset,
    y1 = object->imageCenterY - offset,
    x2 = object->imageCenterX + offset,
    y2 = object->imageCenterY + offset;

    painter.drawLine(x1, y1, x2, y2);

    x1 = object->imageCenterX - offset,
    y1 = object->imageCenterY + offset,
    x2 = object->imageCenterX + offset,
    y2 = object->imageCenterY - offset;

    painter.drawLine(x1, y1, x2, y2);
  }
}

void VisionWindow::drawBeacons(ImageWidget* image) {
  if(!config_.all) return;
  if(world_object_block_ == NULL) return;
  map<WorldObjectType,vector<QColor>> beacons = {
    { WO_BEACON_BLUE_YELLOW, { segCol[c_BLUE], segCol[c_YELLOW] } },
    { WO_BEACON_YELLOW_BLUE, { segCol[c_YELLOW], segCol[c_BLUE] } },
    { WO_BEACON_BLUE_PINK, { segCol[c_BLUE], segCol[c_PINK] } },
    { WO_BEACON_PINK_BLUE, { segCol[c_PINK], segCol[c_BLUE] } },
    { WO_BEACON_PINK_YELLOW, { segCol[c_PINK], segCol[c_YELLOW] } },
    { WO_BEACON_YELLOW_PINK, { segCol[c_YELLOW], segCol[c_PINK] } }
  };
  auto processor = getImageProcessor(image);
  const auto& cmatrix = processor->getCameraMatrix();
  QPainter painter(image->getImage());
  painter.setRenderHint(QPainter::Antialiasing);
  for(auto beacon : beacons) {
    auto& object = world_object_block_->objects_[beacon.first];
    if(!object.seen) continue;
    if(object.fromTopCamera && _widgetAssignments[image] == Camera::BOTTOM) continue;
    if(!object.fromTopCamera && _widgetAssignments[image] == Camera::TOP) continue;
    QPen tpen(beacon.second[0]), bpen(beacon.second[1]);

    int width = cmatrix.getCameraWidthByDistance(object.visionDistance, 110);
    int height = cmatrix.getCameraHeightByDistance(object.visionDistance, 100);
    int x1 = object.imageCenterX - width / 2;
    
    // Draw top
    int ty1 = object.imageCenterY - height;
    QPainterPath tpath;
    tpath.addRoundedRect(QRect(x1, ty1, width, height), 5, 5);
    painter.setPen(tpen);
    painter.fillPath(tpath, QBrush(beacon.second[0]));

    // Draw bottom
    int by1 = object.imageCenterY, by2 = object.imageCenterY + height;
    QPainterPath bpath;
    bpath.addRoundedRect(QRect(x1, by1, width, height), 5, 5);
    painter.setPen(bpen);
    painter.fillPath(bpath, QBrush(beacon.second[1]));

    // Draw pointer if occluded
    if(object.occluded) {
      painter.setPen(QPen(Qt::red));
      painter.setFont(QFont("Helvetica", 8));
      painter.drawText(QPointF(object.imageCenterX - width, object.imageCenterY + 1.5 * height), "Occluded");
    }

  }
}
