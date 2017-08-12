#ifndef __OPENCV_FACEMARK_AAM_HPP__
#define __OPENCV_FACEMARK_AAM_HPP__

#include "opencv2/face/facemark.hpp"
namespace cv {
namespace face {
    class CV_EXPORTS_W FacemarkAAM : public Facemark
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

            /**
            * \brief Read parameters from file, currently unused
            */
            void read(const FileNode& /*fn*/);

            /**
            * \brief Read parameters from file, currently unused
            */
            void write(FileStorage& /*fs*/) const;
        };

        struct CV_EXPORTS Model{
            int npts;
            int max_n;
            std::vector<int>scales;

            /*warping*/
            std::vector<Vec3i> triangles;

            struct Texture{
                int max_m;
                Rect resolution;
                Mat A0,A,AA0,AA;
                std::vector<std::vector<Point> > textureIdx;
                std::vector<Point2f> base_shape;
                std::vector<int> ind1, ind2;
            };
            std::vector<Texture> textures;

            /*shape*/
            std::vector<Point2f> s0;
            Mat S,Q;
        };

        static Ptr<FacemarkAAM> create(const FacemarkAAM::Params &parameters = FacemarkAAM::Params() );
        virtual ~FacemarkAAM() {}

    }; /* AAM */

} /* namespace face */
} /* namespace cv */
#endif
