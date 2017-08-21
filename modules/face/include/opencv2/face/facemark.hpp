
#ifndef __OPENCV_FACELANDMARK_HPP__
#define __OPENCV_FACELANDMARK_HPP__

#include "opencv2/face.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/objdetect/objdetect_c.h"
#include "opencv2/imgproc/types_c.h"

namespace cv {
namespace face {

    CV_EXPORTS_W bool getFacesHaar( const Mat image,
                                    std::vector<Rect> & faces,
                                    String face_cascade_name );

    CV_EXPORTS_W bool loadTrainingData( String filename , std::vector<String> & images,
                                        std::vector<std::vector<Point2f> > & facePoints,
                                        char delim = ' ', float offset = 0.0);

    CV_EXPORTS_W bool loadTrainingData( String imageList, String groundTruth,
                                        std::vector<String> & images,
                                        std::vector<std::vector<Point2f> > & facePoints,
                                        float offset = 0.0);

    CV_EXPORTS_W bool loadFacePoints( String filename, std::vector<Point2f> & pts,
                                      float offset = 0.0);

    CV_EXPORTS_W void drawFacemarks( Mat & image, std::vector<Point2f> pts,
                                     Scalar color = Scalar(255,0,0));

    class CV_EXPORTS_W Facemark : public virtual Algorithm
    {
    public:

        // virtual void read( const FileNode& fn )=0;
        // virtual void write( FileStorage& fs ) const=0;

        /**
        * \brief training the facemark model, input are the file names of image list and landmark annotation
        */
        virtual void training(String imageList, String groundTruth)=0;
        // virtual void saveModel(String fs)=0;
        virtual void loadModel(String fs)=0;

        /**
        * \brief extract landmark points from a face
        */
        // CV_WRAP bool detect( InputArray image, Rect2d& boundingBox );
        virtual bool fit( InputArray image, std::vector<Rect> faces, std::vector<std::vector<Point2f> > & landmarks )=0;//!< from many ROIs

        //!<  set the custom face detector
        virtual bool setFaceDetector(bool(*f)(const Mat , std::vector<Rect> & ))=0;
        //!<  get faces using the custom detector
        virtual bool getFaces( const Mat image , std::vector<Rect> & faces)=0;

    }; /* Facemark*/

} /* namespace face */
} /* namespace cv */


#endif //__OPENCV_FACELANDMARK_HPP__
