#include "opencv2/face.hpp"
#include "opencv2/core.hpp"
#include "precomp.hpp"

/*dataset parser*/
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>     /* atoi */

#undef BOILERPLATE_CODE
#define BOILERPLATE_CODE(name,classname)\
    if(facemarkType==name){\
        return classname::create();\
}

namespace cv {
namespace face {

    bool getFacesHaar( const Mat image, std::vector<Rect> & faces, String face_cascade_name ){
        Mat gray;

        CascadeClassifier face_cascade;
        if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face_cascade\n"); return false; };

        cvtColor( image, gray, CV_BGR2GRAY );
        equalizeHist( gray, gray );
        face_cascade.detectMultiScale( gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

        return true;
    }

    bool loadTrainingData(String filename, std::vector<String> & images, std::vector<std::vector<Point2f> > & facePoints, char delim, float offset){
        std::string line;
        std::string item;
        std::vector<Point2f> pts;
        std::vector<float> raw;

        std::ifstream infile;
        infile.open(filename.c_str(), std::ios::in);
        if (!infile) {
           std::string error_message = "No valid input file was given, please check the given filename.";
           CV_Error(CV_StsBadArg, error_message);
        }

        /*clear the output containers*/
        images.clear();
        facePoints.clear();

        /*the main loading process*/
        while (getline (infile, line)){
            std::istringstream ss(line); // string stream for the current line

            /*pop the image path*/
            getline (ss, item, delim);
            images.push_back(item);

            /*load all numbers*/
            raw.clear();
            while (getline (ss, item, delim)){
                raw.push_back((float)atof(item.c_str()));
            }

            /*convert to opencv points*/
            pts.clear();
            for(unsigned i = 0;i< raw.size();i+=2){
                pts.push_back(Point2f(raw[i]+offset,raw[i+1]+offset));
            }
            facePoints.push_back(pts);
        } // main loading process

        return true;
    }

    bool loadTrainingData(String imageList, String groundTruth, std::vector<String> & images, std::vector<std::vector<Point2f> > & facePoints, float offset){
        std::string line;
        std::vector<Point2f> facePts;

        /*clear the output containers*/
        images.clear();
        facePoints.clear();

        /*load the images path*/
        std::ifstream infile;
        infile.open(imageList.c_str(), std::ios::in);
        if (!infile) {
           std::string error_message = "No valid input file was given, please check the given filename.";
           CV_Error(CV_StsBadArg, error_message);
        }

        while (getline (infile, line)){
            images.push_back(line);
        }

        /*load the points*/
        std::ifstream ss_gt(groundTruth.c_str());
        while (getline (ss_gt, line)){
            facePts.clear();
            loadFacePoints(line, facePts, offset);
            facePoints.push_back(facePts);
        }

        return true;
    }

    bool loadFacePoints(String filename, std::vector<Point2f> & pts, float offset){
        std::string line, item;

        std::ifstream infile(filename.c_str());

        /*pop the version*/
        std::getline(infile, line);
        CV_Assert(line.compare(0,7,"version")==0);

        /*pop the number of points*/
        std::getline(infile, line);
        CV_Assert(line.compare(0,8,"n_points")==0);

        /*get the number of points*/
        std::string item_npts;
        int npts;

        std::istringstream linestream(line);
        linestream>>item_npts>>npts;

        /*pop out '{' character*/
        std::getline(infile, line);

        /*main process*/
        int cnt = 0;
        std::string x, y;
        pts.clear();
        while (std::getline(infile, line) && cnt<npts )
        {
            cnt+=1;

            std::istringstream ss(line);
            ss>>x>>y;
            pts.push_back(Point2f((float)atof(x.c_str())+offset,(float)atof(y.c_str())+offset));

        }

        return true;
    }

    void drawFacemarks(Mat & image, std::vector<Point2f> pts, Scalar color){
        for(size_t i=0;i<pts.size();i++){
            circle(image, pts[i],3, color,-1);
        }
    } //drawPoints

} /* namespace face */
} /* namespace cv */
