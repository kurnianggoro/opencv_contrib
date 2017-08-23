
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

    /** @brief Default face detector
    This function is mainly utilized by the implementation of a Facemark Algorithm.
    End users are advised to use function Facemark::getFaces which can be manually defined
    and circumvented to the algorithm by Facemark::setFaceDetector.

    @param image The input image to be processed.
    @param faces Output of the function which represent region of interest of the detected faces.
    Each face is stored in cv::Rect container.
    @param face_cascade_model The filename of a cascade model for face detection.

    <B>Example of usage</B>
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
                                    String face_cascade_model );

    /** @brief A utility to load facial landmark dataset from a single file.

    @param filename The filename of a file that contains the dataset information.
    Each line contains the filename of an image followed by
    pairs of x and y values of facial landmarks points separated by a space.
    Example
    @code
    /home/user/ibug/image_003_1.jpg 336.820955 240.864510 334.238298 260.922709 335.266918 ...
    /home/user/ibug/image_005_1.jpg 376.158428 230.845712 376.736984 254.924635 383.265403 ...
    @endcode
    @param images A vector where each element represent the filename of image in the dataset.
    Images are not loaded by default to save the memory.
    @param facePoints The loaded landmark points for all training data.

    <B>Example of usage</B>
    @code
    cv::String imageFiles = "../data/images_train.txt";
    cv::String ptsFiles = "../data/points_train.txt";
    std::vector<String> images;
    std::vector<std::vector<Point2f> > facePoints;
    loadTrainingData(imageFiles, ptsFiles, images, facePoints, 0.0);
    @endcode
    */

    CV_EXPORTS_W bool loadTrainingData( String filename , std::vector<String> & images,
                                        OutputArray facePoints,
                                        char delim = ' ', float offset = 0.0);


    /** @brief A utility to load facial landmark information from the dataset.

    @param imageList A file contains the list of image filenames in the training dataset.
    @param groundTruth A file contains the list of filenames
    where the landmarks points information are stored.
    The content in each file should follow the standard format (see face::loadFacePoints).
    @param images A vector where each element represent the filename of image in the dataset.
    Images are not loaded by default to save the memory.
    @param facePoints The loaded landmark points for all training data.

    <B>Example of usage</B>
    @code
    cv::String imageFiles = "../data/images_train.txt";
    cv::String ptsFiles = "../data/points_train.txt";
    std::vector<String> images;
    std::vector<std::vector<Point2f> > facePoints;
    loadTrainingData(imageFiles, ptsFiles, images, facePoints, 0.0);
    @endcode

    example of content in the images_train.txt
    @code
    /home/user/ibug/image_003_1.jpg
    /home/user/ibug/image_004_1.jpg
    /home/user/ibug/image_005_1.jpg
    /home/user/ibug/image_006.jpg
    @endcode

    example of content in the points_train.txt
    @code
    /home/user/ibug/image_003_1.pts
    /home/user/ibug/image_004_1.pts
    /home/user/ibug/image_005_1.pts
    /home/user/ibug/image_006.pts
    @endcode
    */

    CV_EXPORTS_W bool loadTrainingData( String imageList, String groundTruth,
                                        std::vector<String> & images,
                                        OutputArray facePoints,
                                        float offset = 0.0);

    /** @brief A utility to load facial landmark information from a given file.

    @param filename The filename of file contains the facial landmarks data.
    @param points The loaded facial landmark points.
    @param offset An offset value to adjust the loaded points.

    <B>Example of usage</B>
    @code
    std::vector<Point2f> points;
    face::loadFacePoints("filename.txt", points, 0.0);
    @endcode

    The annotation file should follow the default format which is
    @code
    version: 1
    n_points:  68
    {
    212.716603 499.771793
    230.232816 566.290071
    ...
    }
    @endcode
    where n_points is the number of points considered
    and each point is represented as its position in x and y.
    */

    CV_EXPORTS_W bool loadFacePoints( String filename, OutputArray points,
                                      float offset = 0.0);

    /** @brief Utility to draw the detected facial landmark points

    @param image The input image to be processed.
    @param points Contains the data of points which will be drawn.
    @param color The color of points in BGR format represented by cv::Scalar.

    <B>Example of usage</B>
    @code
    std::vector<Rect> faces;
    std::vector<std::vector<Point2f> > landmarks;
    facemark->getFaces(img, faces);
    facemark->fit(img, faces, landmarks);
    for(int j=0;j<rects.size();j++){
        face::drawFacemarks(frame, landmarks[j], Scalar(0,0,255));
    }
    @endcode
    */
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
