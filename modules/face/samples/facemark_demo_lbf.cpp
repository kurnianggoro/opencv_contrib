/*----------------------------------------------
 * Usage:
 * facemark_demo_lbf
 *
 *
 * Notes:
 * the user should provides the list of training images_train
 * accompanied by their corresponding landmarks location in separated files.
 * example of contents for images_train.txt:
 * ../trainset/image_0001.png
 * ../trainset/image_0002.png
 * example of contents for points_train.txt:
 * ../trainset/image_0001.pts
 * ../trainset/image_0002.pts
 * where the image_xxxx.pts contains the position of each face landmark.
 * example of the contents:
 *  version: 1
 *  n_points:  68
 *  {
 *  115.167660 220.807529
 *  116.164839 245.721357
 *  120.208690 270.389841
 *  ...
 *  }
 * example of the dataset is available at https://ibug.doc.ic.ac.uk/download/annotations/ibug.zip
 *--------------------------------------------------*/

 #include <stdio.h>
 #include <fstream>
 #include <sstream>
 #include "opencv2/core.hpp"
 #include "opencv2/highgui.hpp"
 #include "opencv2/imgproc.hpp"
 #include "opencv2/face.hpp"

 using namespace std;
 using namespace cv;
 using namespace cv::face;

 bool loadDatasetList(String imageList, String groundTruth, std::vector<String> & images, std::vector<String> & landmarks);

 int main()
 {
     /*create the facemark instance*/
     FacemarkLBF::Params params;
     params.model_filename = "ibug68.model";
     params.cascade_face = "../data/haarcascade_frontalface_alt.xml";
     Ptr<Facemark> facemark = FacemarkLBF::create(params);

     /*Loads the dataset*/
     String imageFiles = "../data/images_train.txt";
     String ptsFiles = "../data/points_train.txt";
     std::vector<String> images_train;
     std::vector<String> landmarks_train;
     loadDatasetList(imageFiles,ptsFiles,images_train,landmarks_train);

     Mat image;
     std::vector<Point2f> facial_points;
     for(size_t i=0;i<images_train.size();i++){
         image = imread(images_train[i].c_str());
         loadFacePoints(landmarks_train[i],facial_points);
         facemark->addTrainingSample(image, facial_points);
     }

     /*train the Algorithm*/
     facemark->training();

     String testFiles = "../data/images_test.txt";
     String testPts = "../data/points_test.txt";
     std::vector<String> images;
     std::vector<String> facePoints;
     loadDatasetList(testFiles, testPts, images, facePoints);

     std::vector<Rect> rects;
     CascadeClassifier cc(params.cascade_face.c_str());
     for(int i=0;i<(int)images.size();i++){
         Mat img = imread(images[i]);

         std::vector<std::vector<Point2f> > landmarks;
         rects.push_back(Rect(0,0,img.cols, img.rows));
         facemark->fit(img, rects, landmarks);
         drawFacemarks(img, landmarks[0], Scalar(0,0,255));
         imshow("result", img);
         waitKey(0);
     }

 }

 bool loadDatasetList(String imageList, String groundTruth, std::vector<String> & images, std::vector<String> & landmarks){
     std::string line;

     /*clear the output containers*/
     images.clear();
     landmarks.clear();

     /*open the files*/
     std::ifstream infile;
     infile.open(imageList.c_str(), std::ios::in);
     std::ifstream ss_gt;
     ss_gt.open(groundTruth.c_str(), std::ios::in);
     if ((!infile) || !(ss_gt)) {
        printf("No valid input file was given, please check the given filename.\n");
        return false;
     }

      /*load the images path*/
     while (getline (infile, line)){
         images.push_back(line);
     }

     /*load the points*/
     while (getline (ss_gt, line)){
         landmarks.push_back(line);
     }

     return true;
 }
