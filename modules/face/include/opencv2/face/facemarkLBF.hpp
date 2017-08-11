#ifndef __OPENCV_FACEMARK_LBF_HPP__
#define __OPENCV_FACEMARK_LBF_HPP__

#include "opencv2/face/facemark.hpp"

namespace cv {
namespace face {
    class CV_EXPORTS_W FacemarkLBF : public Facemark
    {
    public:
        struct CV_EXPORTS Params
        {
            /**
            * \brief Constructor
            */
            Params();

            /*read only parameters - just for example*/
            double detect_thresh;         //!<  detection confidence threshold
            double sigma;                 //!<  another parameter
            double shape_offset;
            String cascade_face;

            int n_landmarks;
            int initShape_n;

            int stages_n;
            int tree_n;
            int tree_depth;
            double bagging_overlap;

            std::string saved_file_name;
            std::vector<int> feats_m;
            std::vector<double> radius_m;
            std::vector<int> pupils[2];

            Rect detectROI;

            void read(const FileNode& /*fn*/);
            void write(FileStorage& /*fs*/) const;

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

        Params params;
        bool fit( const Mat image, std::vector<Rect> faces, std::vector<std::vector<Point2f> >& landmarks );//!< from many ROIs

        static Ptr<FacemarkLBF> create(const FacemarkLBF::Params &parameters);
        CV_WRAP static Ptr<FacemarkLBF> create();
        virtual ~FacemarkLBF(){};
    }; /* LBF */
} /* namespace face */
}/* namespace cv */

#endif
