/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.
                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)
Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.
This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

/*----------------------------------------------
 * Usage:
 * facemark_demo_aam <face_cascade_model> <eyes_cascade_model> <training_images> <annotation_files> [test_files]
 *
 * Example:
 * facemark_demo_aam ../face_cascade.xml ../eyes_cascade.xml ../images_train.txt ../points_train.txt ../test.txt
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
 * example of the dataset is available at http://www.ifp.illinois.edu/~vuongle2/helen/
 *--------------------------------------------------*/

 #include <stdio.h>
 #include <fstream>
 #include <sstream>
 #include "opencv2/core.hpp"
 #include "opencv2/highgui.hpp"
 #include "opencv2/imgproc.hpp"
 #include "opencv2/face.hpp"

 #include <iostream>
 #include <fstream>
 #include <string>

 using namespace std;
 using namespace cv;
 using namespace cv::face;

 bool myDetector( InputArray image, OutputArray ROIs, CascadeClassifier face_cascade);
 bool getInitialFitting(Mat image, Rect face, std::vector<Point2f> s0,
     CascadeClassifier eyes_cascade, Mat & R, Point2f & Trans, float & scale);
 bool parseArguments(int argc, char** argv, CommandLineParser & , String & cascade,
     String & model, String & images, String & annotations, String & testImages
 );

 int main(int argc, char** argv )
 {
     CommandLineParser parser(argc, argv,"");
     String cascade_path,eyes_cascade_path,images_path, annotations_path, test_images_path;
     if(!parseArguments(argc, argv, parser,cascade_path,eyes_cascade_path,images_path, annotations_path, test_images_path))
        return -1;

     /*create the facemark instance*/
     Ptr<FacemarkAAM> facemark = FacemarkAAM::create();

     /*Loads the dataset*/
     std::vector<String> images_train;
     std::vector<String> landmarks_train;
     loadDatasetList(images_path,annotations_path,images_train,landmarks_train);

     Mat image;
     std::vector<Point2f> facial_points;
     for(size_t i=0;i<images_train.size();i++){
         image = imread(images_train[i].c_str());
         loadFacePoints(landmarks_train[i],facial_points);
         facemark->addTrainingSample(image, facial_points);
     }

     /* trained model will be saved to AAM.yml */
     facemark->training();

     /*test using some images*/
     String testFiles(images_path), testPts(annotations_path);
     if(!test_images_path.empty()){
         testFiles = test_images_path;
         testPts = test_images_path; //unused
     }
     std::vector<String> images;
     std::vector<String> facePoints;
     loadDatasetList(testFiles, testPts, images, facePoints);

     float scale ;
     Point2f T;
     Mat R;

     FacemarkAAM::Model config;
     facemark->getParams(config);
     std::vector<Point2f> s0 = config.s0;

     /*fitting process*/
     std::vector<Rect> faces;
     CascadeClassifier face_cascade(cascade_path);
     CascadeClassifier eyes_cascade(eyes_cascade_path);
     for(int i=0;i<(int)images.size();i++){
         printf("image #%i ", i);
         image = imread(images[i]);
         myDetector(image, faces, face_cascade);
         if(faces.size()>0){
             std::vector<FacemarkAAM::Config> conf;
             std::vector<Rect> faces_eyes;
             for(unsigned j=0;j<faces.size();j++){
                 if(getInitialFitting(image,faces[j],s0,eyes_cascade, R,T,scale)){
                     conf.push_back(FacemarkAAM::Config(R,T,scale));
                     faces_eyes.push_back(faces[j]);
                 }
             }
             if(conf.size()>0){

                 printf(" - face with eyes found %i", (int)conf.size());
                 std::vector<std::vector<Point2f> > landmarks;

                 facemark->fit(image, faces_eyes, landmarks, (void*)&conf);
                //  cout<<Mat(landmarks[0])<<endl;
                 for(unsigned j=0;j<landmarks.size();j++){
                     drawFacemarks(image, landmarks[j]);
                 }
             }
         }
         printf("\n");
         imshow("fitting", image);
         waitKey(0);
     } //for
 }

 bool myDetector( InputArray image, OutputArray ROIs, CascadeClassifier face_cascade){
     Mat gray;
     std::vector<Rect> & faces = *(std::vector<Rect>*) ROIs.getObj();
     faces.clear();

     if(image.channels()>1){
         cvtColor(image.getMat(),gray,CV_BGR2GRAY);
     }else{
         gray = image.getMat().clone();
     }
     equalizeHist( gray, gray );

     face_cascade.detectMultiScale( gray, faces, 1.2, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30) );
     return true;
 }

 bool getInitialFitting(Mat image, Rect face, std::vector<Point2f> s0 ,CascadeClassifier eyes_cascade, Mat & R, Point2f & Trans, float & scale){
     std::vector<Point2f> mybase;
     std::vector<Point2f> T;
     std::vector<Point2f> base = Mat(Mat(s0)+Scalar(image.cols/2,image.rows/2)).reshape(2);

     std::vector<Point2f> base_shape,base_shape2 ;
     Point2f e1 = Point2f((float)((base[39].x+base[36].x)/2.0),(float)((base[39].y+base[36].y)/2.0)); //eye1
     Point2f e2 = Point2f((float)((base[45].x+base[42].x)/2.0),(float)((base[45].y+base[42].y)/2.0)); //eye2

     if(face.width==0 || face.height==0) return false;

     std::vector<Point2f> eye;
     bool found=false;

         Mat faceROI = image( face);
         std::vector<Rect> eyes;

         //-- In each face, detect eyes
         eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(20, 20) );
         if(eyes.size()==2){
             found = true;
             int j=0;
             Point2f c1( (float)(face.x + eyes[j].x + eyes[j].width*0.5), (float)(face.y + eyes[j].y + eyes[j].height*0.5));

             j=1;
             Point2f c2( (float)(face.x + eyes[j].x + eyes[j].width*0.5), (float)(face.y + eyes[j].y + eyes[j].height*0.5));

             Point2f pivot;
             double a0,a1;
             if(c1.x<c2.x){
                 pivot = c1;
                 a0 = atan2(c2.y-c1.y, c2.x-c1.x);
             }else{
                 pivot = c2;
                 a0 = atan2(c1.y-c2.y, c1.x-c2.x);
             }

             scale = (float)(norm(Mat(c1)-Mat(c2))/norm(Mat(e1)-Mat(e2)));

             mybase= Mat(Mat(s0)*scale).reshape(2);
             Point2f ey1 = Point2f((float)((mybase[39].x+mybase[36].x)/2.0),(float)((mybase[39].y+mybase[36].y)/2.0));
             Point2f ey2 = Point2f((float)((mybase[45].x+mybase[42].x)/2.0),(float)((mybase[45].y+mybase[42].y)/2.0));


             #define TO_DEGREE 180.0/3.14159265
             a1 = atan2(ey2.y-ey1.y, ey2.x-ey1.x);
             Mat rot = getRotationMatrix2D(Point2f(0,0), (a1-a0)*TO_DEGREE, 1.0);

             rot(Rect(0,0,2,2)).convertTo(R, CV_32F);

             base_shape = Mat(Mat(R*scale*Mat(Mat(s0).reshape(1)).t()).t()).reshape(2);
             ey1 = Point2f((float)((base_shape[39].x+base_shape[36].x)/2.0),(float)((base_shape[39].y+base_shape[36].y)/2.0));
             ey2 = Point2f((float)((base_shape[45].x+base_shape[42].x)/2.0),(float)((base_shape[45].y+base_shape[42].y)/2.0));

             T.push_back(Point2f(pivot.x-ey1.x,pivot.y-ey1.y));
             Trans = Point2f(pivot.x-ey1.x,pivot.y-ey1.y);
             return true;
         }else{
             // T.push_back(Point2f( face.x + face.width*0.5, face.y + face.height*0.5));
             Trans = Point2f( (float)(face.x + face.width*0.5),(float)(face.y + face.height*0.5));
         }


     // Trans = T[0];

     return found;
 }

 bool parseArguments(int argc, char** argv, CommandLineParser & parser,
     String & cascade,
     String & model,
     String & images,
     String & annotations,
     String & test_images
 ){
    const String keys =
        "{ @f face-cascade    |      | (required) path to the cascade model file for the face detector }"
        "{ @e eyes-cascade    |      | (required) path to the cascade model file for the eyes detector }"
        "{ @i images          |      | (required) path of a text file contains the list of paths to all training images}"
        "{ @a annotations     |      | (required) Path of a text file contains the list of paths to all annotations files}"
        "{ t test-images      |      | Path of a text file contains the list of paths to the test images}"
        "{ help h usage ?     |      | facemark_demo_aam -face-cascade -eyes-cascade -images -annotations [-t]\n"
             " example: facemark_demo_aam ../face_cascade.xml ../eyes_cascade.xml ../images_train.txt ../points_train.txt ../test.txt}"
    ;
    parser = CommandLineParser(argc, argv,keys);
    parser.about("hello");

    if (parser.has("help")){
        parser.printMessage();
        return false;
    }

    cascade = String(parser.get<String>("face-cascade"));
    model = String(parser.get<string>("eyes-cascade"));
    images = String(parser.get<string>("images"));
    annotations = String(parser.get<string>("annotations"));
    test_images = String(parser.get<string>("t"));

    if(cascade.empty() || model.empty() || images.empty() || annotations.empty()){
        std::cerr << "one or more required arguments are not found" << '\n';
        cout<<"face-cascade : "<<cascade.c_str()<<endl;
        cout<<"eyes-cascade : "<<model.c_str()<<endl;
        cout<<"images : "<<images.c_str()<<endl;
        cout<<"annotations : "<<annotations.c_str()<<endl;
        parser.printMessage();
        return false;
    }

    return true;
 }
