#ifndef __OPENCV_FACEMARK_LBF_HPP__
#define __OPENCV_FACEMARK_LBF_HPP__

#include "opencv2/face/facemark.hpp"

namespace cv {
namespace face {

//! @addtogroup facemark
//! @{

    class CV_EXPORTS_W FacemarkLBF : public Facemark
    {
    public:
        struct CV_EXPORTS Params
        {
            /**
            * \brief Constructor
            */
            Params();

            double shape_offset;
            //!<  offset for the loaded face landmark points
            String cascade_face;
            //!<  filename of the face detector model

            int n_landmarks;
            //!<  number of landmark points
            int initShape_n;
            //!<  multiplier for augment the training data

            int stages_n;
            //!<  number of refinement stages
            int tree_n;
            //!<  number of tree in the model for each landmark point refinement
            int tree_depth;
            //!<  the depth of decision tree, defines the size of feature
            double bagging_overlap;
            //!<  overlap ratio for training the LBF feature

            std::string model_filename;
            //!<  number of refinement stages
            std::vector<int> feats_m;
            std::vector<double> radius_m;
            std::vector<int> pupils[2];
            //!<  index of facemark points on pupils of left and right eye

            Rect detectROI;

            // void read(const FileNode& /*fn*/);
            // void write(FileStorage& /*fs*/) const;

        };

        class BBox {
        public:
            BBox();
            ~BBox();
            BBox(double x, double y, double w, double h);

            cv::Mat project(const cv::Mat &shape) const;
            cv::Mat reproject(const cv::Mat &shape) const;

            double x, y;
            double x_center, y_center;
            double x_scale, y_scale;
            double width, height;
        };

        static Ptr<FacemarkLBF> create(const FacemarkLBF::Params &parameters = FacemarkLBF::Params() );
        virtual ~FacemarkLBF(){};
    }; /* LBF */

//! @}

} /* namespace face */
}/* namespace cv */

#endif
