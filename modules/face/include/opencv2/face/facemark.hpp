
#ifndef __OPENCV_FACELANDMARK_HPP__
#define __OPENCV_FACELANDMARK_HPP__

/**
@defgroup facemark Face Landmark Detection
- @ref tutorial_table_of_content_facemark
*/

#include "opencv2/face.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/objdetect/objdetect_c.h"
#include "opencv2/imgproc/types_c.h"

namespace cv {
namespace face {

//! @addtogroup facemark
//! @{

    /** @brief Abstract base class for all facemark models
    @code
    std::vector<cv::Rect> roi;
    cv::face::getFacesHaar(frame, roi, "haarcascade_frontalface_alt.xml");
    for(int j=0;j<rects.size();j++){
        cv::rectangle(frame, rects[j], cv::Scalar(255,0,255));
    }
    @endcode
    */
    CV_EXPORTS_W bool getFacesHaar( InputArray image,
                                    OutputArray faces,
                                    String face_cascade_name );

    CV_EXPORTS_W bool loadTrainingData( String filename , std::vector<String> & images,
                                        OutputArray facePoints,
                                        char delim = ' ', float offset = 0.0);

    CV_EXPORTS_W bool loadTrainingData( String imageList, String groundTruth,
                                        std::vector<String> & images,
                                        OutputArray facePoints,
                                        float offset = 0.0);

    CV_EXPORTS_W bool loadFacePoints( String filename, OutputArray points,
                                      float offset = 0.0);

    CV_EXPORTS_W void drawFacemarks( InputOutputArray image, InputArray points,
                                     Scalar color = Scalar(255,0,0));

    /** @brief Abstract base class for all facemark models

    All facemark models in OpenCV are derived from the abstract base class Facemark, which
    provides a unified access to all facemark algorithms in OpenCV.

    ### Description

    Facemark is a base class which provides universal access to any specific facemark algorithm.
    Therefore, the users should declare a desired algorithm before they can use it in their application.

    Here is an example on how to declare facemark algorithm:
    @code
    // Using Facemark in your code:
    Ptr<Facemark> facemark = FacemarkLBF::create();
    @endcode

    The typical pipeline for facemark detection is listed as follows:
    - (Non-mandatory) Set a user defined face detection using Facemark::setFaceDetector.
      The facemark algorithms are desgined to fit the facial points into a face.
      Therefore, the face information should be provided to the facemark algorithm.
      Some algorithms might provides a default face recognition function.
      However, the users might prefer to use their own face detector to obtains the best possible detection result.
    - (Non-mandatory) Training the model for a specific algorithm using Facemark::training.
      In this case, the model should be automatically saved by the algorithm.
      If the user already have a trained model, then this part can be omitted.
    - Load the trained model using Facemark::loadModel.
    - Perform the fitting via the Facemark::fit.
    */
    class CV_EXPORTS_W Facemark : public virtual Algorithm
    {
    public:

        // virtual void read( const FileNode& fn )=0;
        // virtual void write( FileStorage& fs ) const=0;

        /**
        * \brief training the facemark model, input are the file names of image list and landmark annotation
        */
        virtual void training(String imageList, String groundTruth)=0;
        virtual void loadModel(String fs)=0;
        // virtual void saveModel(String fs)=0;

        /**
        * \brief extract landmark points from a face
        */
        virtual bool fit( InputArray image, InputArray faces, InputOutputArray landmarks )=0;//!< from many ROIs

        virtual bool setFaceDetector(bool(*f)(InputArray , OutputArray ))=0;
        //!<  set the custom face detector
        virtual bool getFaces( InputArray image , OutputArray faces)=0;
        //!<  get faces using the custom detector

    }; /* Facemark*/

//! @}

} /* namespace face */
} /* namespace cv */


#endif //__OPENCV_FACELANDMARK_HPP__
