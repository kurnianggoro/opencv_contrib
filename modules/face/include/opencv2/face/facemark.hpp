
#ifndef __OPENCV_FACELANDMARK_HPP__
#define __OPENCV_FACELANDMARK_HPP__

#include "opencv2/face.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/objdetect/objdetect_c.h"
#include "opencv2/imgproc/types_c.h"

namespace cv {
namespace face {

    class CV_EXPORTS_W Facemark : public virtual Algorithm
    {
    public:

        virtual void read( const FileNode& fn )=0;
        virtual void write( FileStorage& fs ) const=0;

        /**
        * \brief training the facemark model, input are the file names of image list and landmark annotation
        */
        virtual void training(String imageList, String groundTruth);
        virtual void saveModel(String fs)=0;
        virtual void loadModel(String fs)=0;

        virtual bool loadTrainingData(String filename , std::vector<String> & images, std::vector<std::vector<Point2f> > & facePoints, char delim = ' ', float offset = 0.0);
        virtual bool loadTrainingData(String imageList, String groundTruth, std::vector<String> & images, std::vector<std::vector<Point2f> > & facePoints, float offset = 0.0);
        virtual bool loadFacePoints(String filename, std::vector<Point2f> & pts, float offset = 0.0);
        virtual void drawPoints(Mat & image, std::vector<Point2f> pts, Scalar color = Scalar(255,0,0));

        /**
        * \brief extract landmark points from a face
        */
        // CV_WRAP bool detect( InputArray image, Rect2d& boundingBox );
        virtual bool fit( const Mat image, std::vector<Point2f> & landmarks );//!< from a face
        virtual bool fit( const Mat image, Rect face, std::vector<Point2f> & landmarks );//!< from an ROI
        virtual bool fit( const Mat image, std::vector<Rect> faces, std::vector<std::vector<Point2f> >& landmarks );//!< from many ROIs
        virtual bool fit( const Mat image, std::vector<Point2f>& landmarks, Mat R, Point2f T, float scale );

        static Ptr<Facemark> create( const String& facemarkType );

        //!<  default face detector
        virtual bool getFacesHaar( const Mat image , std::vector<Rect> & faces, String face_cascade_name);

        //!<  set the custom face detector
        virtual bool setFaceDetector(bool(*f)(const Mat , std::vector<Rect> & ));
        //!<  get faces using the custom detector
        virtual bool getFaces( const Mat image , std::vector<Rect> & faces);

        //!<  get faces and then extract landmarks for each of them
        virtual bool process(const Mat image,std::vector<Rect> & faces, std::vector<std::vector<Point2f> >& landmarks );

        //!<  using the default face detector (haarClassifier), xml of the model should be provided
        virtual bool process(const Mat image,std::vector<Rect> & faces, std::vector<std::vector<Point2f> >& landmarks, String haarModel );

    protected:
        virtual bool fitImpl( const Mat image, std::vector<Point2f> & landmarks )=0;
        virtual bool fitImpl( const Mat, std::vector<Point2f>& landmarks, Mat R, Point2f T, float scale )=0; //temporary
        virtual void trainingImpl(String imageList, String groundTruth)=0;

        /*circumventable face extractor function*/
        bool(*faceDetector)(const Mat , std::vector<Rect> &  ) ;
        bool isSetDetector;

    }; /* Facemark*/

} /* namespace face */
} /* namespace cv */


#endif //__OPENCV_FACELANDMARK_HPP__
