/*----------------------------------------------
 * Usage:
 * facemark_demo_aam <image_id>
 *
 * Example:
 * facemark_demo_aam 87
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

Mat loadCSV(std::string filename);
bool loadDatasetList(String imageList, String groundTruth, std::vector<String> & images, std::vector<String> & landmarks);

int main(int argc, char** argv )
{
    Ptr<FacemarkAAM> facemark = FacemarkAAM::create();

    /*--------------- TRAINING -----------------*/
    String imageFiles = "../data/lfpw_images_train.txt";
    String ptsFiles = "../data/lfpw_points_train.txt";
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

    /* trained model will be saved to AAM.yml */
    facemark->training();

    /*--------------- FITTING -----------------*/
    imageFiles = "../data/images_test.txt";
    ptsFiles = "../data/points_test.txt";
    std::vector<String> images;
    std::vector<String> facePoints;
    loadDatasetList(imageFiles, ptsFiles, images, facePoints);

    /*load the selected image*/
    int tId = 0;
    if(argc>1)tId = atoi(argv[1]);
    image = imread(images[tId]);

    /*load the face detection result from another code
    *alternatively, custom face detector can be utilized
    */
    Mat initial = loadCSV("../data/faces.csv");
    float scale = initial.at<float>(tId,0);
    Point2f T = Point2f(initial.at<float>(tId,1),initial.at<float>(tId,2));
    Mat R=Mat::eye(2, 2, CV_32F);

    /*fitting process*/
    std::vector<Point2f> landmarks;
    facemark->fitSingle(image, landmarks, R,T, scale);
    drawFacemarks(image, landmarks);
    imshow("fitting", image);
    waitKey(0);
}

Mat loadCSV(std::string filename){
    ifstream inputfile(filename.c_str());
    std::string current_line;
    // vector allows you to add data without knowing the exact size beforehand
    vector< vector<float> > all_data;
    // Start reading lines as long as there are lines in the file
    while(getline(inputfile, current_line)){
       // Now inside each line we need to seperate the cols
       vector<float> values;
       stringstream temp(current_line);
       string single_value;
       while(getline(temp,single_value,',')){
            // convert the string element to a integer value
            values.push_back((float)atof(single_value.c_str()));
       }
       // add the row to the complete data vector
       all_data.push_back(values);
    }

    // Now add all the data into a Mat element
    Mat vect = Mat::zeros((int)all_data.size(), (int)all_data[0].size(), CV_32F);
    // Loop over vectors and add the data
    for(int rows = 0; rows < (int)all_data.size(); rows++){
       for(int cols= 0; cols< (int)all_data[0].size(); cols++){
          vect.at<float>(rows,cols) = all_data[rows][cols];
       }
    }
    inputfile.close();
    return vect.clone();
}

bool loadDatasetList(String imageList, String groundTruth, std::vector<String> & images, std::vector<String> & landmarks){
    std::string line;

    /*clear the output containers*/
    images.clear();
    landmarks.clear();

    /*open the files*/
    std::ifstream infile;
    std::ifstream ss_gt;
    infile.open(imageList.c_str(), std::ios::in);
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
