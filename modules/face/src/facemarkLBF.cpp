#include "opencv2/face.hpp"
#include "opencv2/imgcodecs.hpp"
#include "precomp.hpp"
#include "liblinear.hpp"
#include <fstream>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cassert>
#include <cstdarg>

namespace cv {
namespace face {

    #define TIMER_BEGIN { double __time__ = (double)getTickCount();
    #define TIMER_NOW   ((getTickCount() - __time__) / getTickFrequency())
    #define TIMER_END   }

    #define SIMILARITY_TRANSFORM(x, y, scale, rotate) do {            \
        double x_tmp = scale * (rotate(0, 0)*x + rotate(0, 1)*y); \
        double y_tmp = scale * (rotate(1, 0)*x + rotate(1, 1)*y); \
        x = x_tmp; y = y_tmp;                                     \
    } while(0)

    FacemarkLBF::Params::Params(){

        cascade_face = "../data/haarcascade_frontalface_alt.xml";
        shape_offset = 0.0;
        n_landmarks = 68;
        initShape_n = 10;
        stages_n=5;
        tree_n=6;
        tree_depth=5;
        bagging_overlap = 0.4;
        model_filename = "ibug.model";
        verbose = true;

        int _pupils[][6] = { { 36, 37, 38, 39, 40, 41 }, { 42, 43, 44, 45, 46, 47 } };
        for (int i = 0; i < 6; i++) {
            pupils[0].push_back(_pupils[0][i]);
            pupils[1].push_back(_pupils[1][i]);
        }

        int _feats_m[] = { 500, 500, 500, 300, 300, 300, 200, 200, 200, 100 };
        double _radius_m[] = { 0.3, 0.2, 0.15, 0.12, 0.10, 0.10, 0.08, 0.06, 0.06, 0.05 };
        for (int i = 0; i < 10; i++) {
            feats_m.push_back(_feats_m[i]);
            radius_m.push_back(_radius_m[i]);
        }

        detectROI = Rect(-1,-1,-1,-1);
    }

    // void FacemarkLBF::Params::read( const cv::FileNode& fn ){
    //     *this = FacemarkLBF::Params();
    //
    //     if (!fn["detect_thresh"].empty())
    //         fn["detect_thresh"] >> detect_thresh;
    //
    //     if (!fn["sigma"].empty())
    //         fn["sigma"] >> sigma;
    //
    // }
    //
    // void FacemarkLBF::Params::write( cv::FileStorage& fs ) const{
    //     fs << "detect_thresh" << detect_thresh;
    //     fs << "sigma" << sigma;
    // }

    class FacemarkLBFImpl : public FacemarkLBF {
    public:
        FacemarkLBFImpl( const FacemarkLBF::Params &parameters = FacemarkLBF::Params() );

        // void read( const FileNode& /*fn*/ );
        // void write( FileStorage& /*fs*/ ) const;

        // void saveModel(String fs);
        void loadModel(String fs);

        bool setFaceDetector(bool(*f)(InputArray , OutputArray ));
        bool getFaces( InputArray image , OutputArray faces);

        Params params;

    protected:

        bool fit( InputArray image, InputArray faces, InputOutputArray landmarks );//!< from many ROIs
        bool fitImpl( const Mat image, std::vector<Point2f> & landmarks );//!< from a face

        bool addTrainingSample(InputArray image, InputArray landmarks);
        void training();

        Rect getBBox(Mat &img, const Mat_<double> shape);
        void prepareTrainingData(Mat img, std::vector<Point2f> facePoints,
            std::vector<Mat> & cropped, std::vector<Mat> & shapes, std::vector<BBox> &boxes);
        void data_augmentation(std::vector<Mat> &imgs, std::vector<Mat> &gt_shapes, std::vector<BBox> &bboxes);
        Mat getMeanShape(std::vector<Mat> &gt_shapes, std::vector<BBox> &bboxes);

        bool configFaceDetector();
        bool defaultFaceDetector(const Mat image, std::vector<Rect> & faces);

        CascadeClassifier face_cascade;
        bool(*faceDetector)(InputArray , OutputArray);
        bool isSetDetector;

        /*training data*/
        std::vector<std::vector<Point2f> > data_facemarks; //original position
        std::vector<Mat> data_faces; //face ROI
        std::vector<BBox> data_boxes;
        std::vector<Mat> data_shapes; //position in the face ROI

    private:
        bool isModelTrained;

        /*---------------LBF Class---------------------*/
        class LBF {
        public:
            void calcSimilarityTransform(const Mat &shape1, const Mat &shape2, double &scale, Mat &rotate);
            std::vector<Mat> getDeltaShapes(std::vector<Mat> &gt_shapes, std::vector<Mat> &current_shapes,
                                       std::vector<BBox> &bboxes, Mat &mean_shape);
            double calcVariance(const Mat &vec);
            double calcVariance(const std::vector<double> &vec);
            double calcMeanError(std::vector<Mat> &gt_shapes, std::vector<Mat> &current_shapes, int landmark_n , std::vector<int> &left, std::vector<int> &right );

        };

        /*---------------RandomTree Class---------------------*/
        class RandomTree : public LBF {
        public:
            RandomTree(){};
            ~RandomTree(){};

            void initTree(int landmark_id, int depth, std::vector<int>, std::vector<double>);
            void train(std::vector<Mat> &imgs, std::vector<Mat> &current_shapes, std::vector<BBox> &bboxes,
                       std::vector<Mat> &delta_shapes, Mat &mean_shape, std::vector<int> &index, int stage);
            void splitNode(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &current_shapes, std::vector<BBox> &bboxes,
                          cv::Mat &delta_shapes, cv::Mat &mean_shape, std::vector<int> &root, int idx, int stage);
            void write(FILE *fd);
            void read(FILE *fd);

            int depth;
            int nodes_n;
            int landmark_id;
            cv::Mat_<double> feats;
            std::vector<int> thresholds;

            std::vector<int> params_feats_m;
            std::vector<double> params_radius_m;
        };
        /*---------------RandomForest Class---------------------*/
        class RandomForest : public LBF {
        public:
            RandomForest(){};
            ~RandomForest(){};

            void initForest(int landmark_n, int trees_n, int tree_depth, double ,  std::vector<int>, std::vector<double>, bool);
            void train(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &current_shapes, \
                       std::vector<BBox> &bboxes, std::vector<cv::Mat> &delta_shapes, cv::Mat &mean_shape, int stage);
            Mat generateLBF(Mat &img, Mat &current_shape, BBox &bbox, Mat &mean_shape);
            void write(FILE *fd);
            void read(FILE *fd);

            bool verbose;
            int landmark_n;
            int trees_n, tree_depth;
            double overlap_ratio;
            std::vector<std::vector<RandomTree> > random_trees;

            std::vector<int> feats_m;
            std::vector<double> radius_m;
        };
        /*---------------Regressor Class---------------------*/
        class Regressor  : public LBF {
        public:
            Regressor(){};
            ~Regressor(){};

            void initRegressor(Params);
            void trainRegressor(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &gt_shapes, \
                       std::vector<cv::Mat> &current_shapes, std::vector<BBox> &bboxes, \
                       cv::Mat &mean_shape, int start_from, Params );
            void globalRegressionTrain(std::vector<Mat> &lbfs, std::vector<Mat> &delta_shapes, int stage, Params);
            Mat globalRegressionPredict(const Mat &lbf, int stage);
            Mat predict(Mat &img, BBox &bbox);

            void write(FILE *fd, Params params);
            void read(FILE *fd, Params & params);

            int stages_n;
            int landmark_n;
            cv::Mat mean_shape;
            std::vector<RandomForest> random_forests;
            std::vector<cv::Mat> gl_regression_weights;
        }; // LBF

        Regressor regressor;
    }; // class

    /*
    * Constructor
    */
    Ptr<FacemarkLBF> FacemarkLBF::create(const FacemarkLBF::Params &parameters){
        return Ptr<FacemarkLBFImpl>(new FacemarkLBFImpl(parameters));
    }

    FacemarkLBFImpl::FacemarkLBFImpl( const FacemarkLBF::Params &parameters )
    {
        isSetDetector =false;
        isModelTrained = false;
        params = parameters;
    }

    bool FacemarkLBFImpl::setFaceDetector(bool(*f)(InputArray , OutputArray )){
        faceDetector = f;
        isSetDetector = true;
        return true;
    }


    bool FacemarkLBFImpl::getFaces( InputArray image , OutputArray roi){

        if(!isSetDetector){
            return false;
        }

        std::vector<Rect> faces;
        faces.clear();

        faceDetector(image.getMat(), faces);
        Mat(faces).copyTo(roi);
        return true;
    }

    bool FacemarkLBFImpl::configFaceDetector(){
        if(!isSetDetector){
            /*check the cascade classifier file*/
            std::ifstream infile;
            infile.open(params.cascade_face.c_str(), std::ios::in);
            if (!infile) {
               std::string error_message = "The cascade classifier model is not found.";
               CV_Error(CV_StsBadArg, error_message);

               return false;
            }

            face_cascade.load(params.cascade_face.c_str());
        }
        return true;
    }

    bool FacemarkLBFImpl::defaultFaceDetector(const Mat image, std::vector<Rect> & faces){
        Mat gray;

        faces.clear();

        if(image.channels()>1){
            cvtColor(image,gray,CV_BGR2GRAY);
        }else{
            gray = image;
        }

        equalizeHist( gray, gray );
        face_cascade.detectMultiScale( gray, faces, 1.05, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

        return true;
    }

    bool FacemarkLBFImpl::addTrainingSample(InputArray image, InputArray landmarks){
        std::vector<Point2f> & _landmarks = *(std::vector<Point2f>*)landmarks.getObj();
        configFaceDetector();
        prepareTrainingData(image.getMat(), _landmarks, data_faces, data_shapes, data_boxes);
        return true;
    }

    void FacemarkLBFImpl::training(){
        if (data_faces.size()<1) {
           std::string error_message =
            "Training data is not provided. Consider to add using addTrainingSample() function!";
           CV_Error(CV_StsBadArg, error_message);
        }

        // flip the image and swap the landmark position
        data_augmentation(data_faces, data_shapes, data_boxes);

        Mat mean_shape = getMeanShape(data_shapes, data_boxes);

        int N = (int)data_faces.size();
        int L = N*params.initShape_n;
        std::vector<Mat> imgs(L), gt_shapes(L), current_shapes(L);
        std::vector<BBox> bboxes(L);
        RNG rng(getTickCount());
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < params.initShape_n; j++) {
                int idx = i*params.initShape_n + j;
                int k = 0;
                do {
                    k = rng.uniform(0, N);
                } while (k == i);
                imgs[idx] = data_faces[i];
                gt_shapes[idx] = data_shapes[i];
                bboxes[idx] = data_boxes[i];
                current_shapes[idx] = data_boxes[i].reproject(data_boxes[k].project(data_shapes[k]));
            }
        }

        // random shuffle
        unsigned int seed = (unsigned int)std::time(0);
        std::srand(seed);
        std::random_shuffle(imgs.begin(), imgs.end());
        std::srand(seed);
        std::random_shuffle(gt_shapes.begin(), gt_shapes.end());
        std::srand(seed);
        std::random_shuffle(bboxes.begin(), bboxes.end());
        std::srand(seed);
        std::random_shuffle(current_shapes.begin(), current_shapes.end());


        regressor.initRegressor(params);
        regressor.trainRegressor(imgs, gt_shapes, current_shapes, bboxes, mean_shape, 0, params);

        FILE *fd = fopen(params.model_filename.c_str(), "wb");
        assert(fd);
        regressor.write(fd, params);
        fclose(fd);

        isModelTrained = true;
    }

    bool FacemarkLBFImpl::fit( InputArray image, InputArray roi, InputOutputArray  _landmarks )
    {
        std::vector<Rect> & faces = *(std::vector<Rect>*)roi.getObj();
        std::vector<std::vector<Point2f> > & landmarks =
            *(std::vector<std::vector<Point2f> >*) _landmarks.getObj();

        landmarks.resize(faces.size());

        for(unsigned i=0; i<faces.size();i++){
            params.detectROI = faces[i];
            fitImpl(image.getMat(), landmarks[i]);
        }

        return true;
    }

    bool FacemarkLBFImpl::fitImpl( const Mat image, std::vector<Point2f>& landmarks){
        if (landmarks.size()>0)
            landmarks.clear();

        if (!isModelTrained) {
           std::string error_message = "The LBF model is not trained yet. Please provide a trained model.";
           CV_Error(CV_StsBadArg, error_message);
        }

        Mat img;
        if(image.channels()>1){
            cvtColor(image,img,CV_BGR2GRAY);
        }else{
            img = image;
        }

        Rect box;
        if (params.detectROI.width>0){
            box = params.detectROI;
        }else{
            std::vector<Rect> rects;

            if(!isSetDetector){
                defaultFaceDetector(img, rects);
            }else{
                faceDetector(img, rects);
            }

            if (rects.size() == 0)  return 0; //failed to get face
            box = rects[0];
        }

        double min_x, min_y, max_x, max_y;
        min_x = std::max(0., (double)box.x - box.width / 2);
        max_x = std::min(img.cols - 1., (double)box.x+box.width + box.width / 2);
        min_y = std::max(0., (double)box.y - box.height / 2);
        max_y = std::min(img.rows - 1., (double)box.y + box.height + box.height / 2);

        double w = max_x - min_x;
        double h = max_y - min_y;

        BBox bbox(box.x - min_x, box.y - min_y, box.width, box.height);
        Mat crop = img(Rect((int)min_x, (int)min_y, (int)w, (int)h)).clone();
        Mat shape = regressor.predict(crop, bbox);

        if(params.detectROI.width>0){
            landmarks = Mat(shape.reshape(2)+Scalar(min_x, min_y));
            params.detectROI.width = -1;
        }else{
            landmarks = Mat(shape.reshape(2)+Scalar(min_x, min_y));
        }

        return 1;
    }

    // void FacemarkLBFImpl::read( const cv::FileNode& fn ){
    //     params.read( fn );
    // }
    //
    // void FacemarkLBFImpl::write( cv::FileStorage& fs ) const {
    //     params.write( fs );
    // }

    // void FacemarkLBFImpl::saveModel(String s){
    //
    // }

    void FacemarkLBFImpl::loadModel(String s){
        if(params.verbose) printf("loading data from : %s\n", s.c_str());
        std::ifstream infile;
        infile.open(s.c_str(), std::ios::in);
        if (!infile) {
           std::string error_message = "No valid input file was given, please check the given filename.";
           CV_Error(CV_StsBadArg, error_message);
        }

        FILE *fd = fopen(s.c_str(), "rb");
        regressor.read(fd, params);
        fclose(fd);

        isModelTrained = true;
    }

    Rect FacemarkLBFImpl::getBBox(Mat &img, const Mat_<double> shape) {
        std::vector<Rect> rects;

        if(!isSetDetector){
            defaultFaceDetector(img, rects);
        }else{
            faceDetector(img, rects);
        }

        if (rects.size() == 0) return Rect(-1, -1, -1, -1);
        double center_x=0, center_y=0, min_x, max_x, min_y, max_y;

        min_x = shape(0, 0);
        max_x = shape(0, 0);
        min_y = shape(0, 1);
        max_y = shape(0, 1);

        for (int i = 0; i < shape.rows; i++) {
            center_x += shape(i, 0);
            center_y += shape(i, 1);
            min_x = std::min(min_x, shape(i, 0));
            max_x = std::max(max_x, shape(i, 0));
            min_y = std::min(min_y, shape(i, 1));
            max_y = std::max(max_y, shape(i, 1));
        }
        center_x /= shape.rows;
        center_y /= shape.rows;

        for (int i = 0; i < (int)rects.size(); i++) {
            Rect r = rects[i];
            if (max_x - min_x > r.width*1.5) continue;
            if (max_y - min_y > r.height*1.5) continue;
            if (abs(center_x - (r.x + r.width / 2)) > r.width / 2) continue;
            if (abs(center_y - (r.y + r.height / 2)) > r.height / 2) continue;
            return r;
        }
        return Rect(-1, -1, -1, -1);
    }

    void FacemarkLBFImpl::prepareTrainingData(Mat img, std::vector<Point2f> facePoints,
        std::vector<Mat> & cropped, std::vector<Mat> & shapes, std::vector<BBox> &boxes)
    {
        Mat shape;
        Mat _shape = Mat(facePoints).reshape(1);
        Rect box = getBBox(img, _shape);
        if(box.x != -1){
            _shape.convertTo(shape, CV_64FC1);
            Mat sx = shape.col(0);
            Mat sy = shape.col(1);
            double min_x, max_x, min_y, max_y;
            minMaxIdx(sx, &min_x, &max_x);
            minMaxIdx(sy, &min_y, &max_y);

            min_x = std::max(0., min_x - box.width / 2);
            max_x = std::min(img.cols - 1., max_x + box.width / 2);
            min_y = std::max(0., min_y - box.height / 2);
            max_y = std::min(img.rows - 1., max_y + box.height / 2);

            double w = max_x - min_x;
            double h = max_y - min_y;

            shape = Mat(shape.reshape(2)-Scalar(min_x, min_y)).reshape(1);

            boxes.push_back(BBox(box.x - min_x, box.y - min_y, box.width, box.height));
            Mat crop = img(Rect((int)min_x, (int)min_y, (int)w, (int)h)).clone();
            cropped.push_back(crop);
            shapes.push_back(shape);
        }

    }

    void FacemarkLBFImpl::data_augmentation(std::vector<Mat> &imgs, std::vector<Mat> &gt_shapes, std::vector<BBox> &bboxes) {
        int N = (int)imgs.size();
        imgs.reserve(2 * N);
        gt_shapes.reserve(2 * N);
        bboxes.reserve(2 * N);
        for (int i = 0; i < N; i++) {
            Mat img_flipped;
            Mat_<double> gt_shape_flipped(gt_shapes[i].size());
            flip(imgs[i], img_flipped, 1);
            int w = img_flipped.cols - 1;
            // int h = img_flipped.rows - 1;
            for (int k = 0; k < gt_shapes[i].rows; k++) {
                gt_shape_flipped(k, 0) = w - gt_shapes[i].at<double>(k, 0);
                gt_shape_flipped(k, 1) = gt_shapes[i].at<double>(k, 1);
            }
            int x_b, y_b, w_b, h_b;
            x_b = w - (int)bboxes[i].x - (int)bboxes[i].width;
            y_b = (int)bboxes[i].y;
            w_b = (int)bboxes[i].width;
            h_b = (int)bboxes[i].height;
            BBox bbox_flipped(x_b, y_b, w_b, h_b);

            imgs.push_back(img_flipped);
            gt_shapes.push_back(gt_shape_flipped);
            bboxes.push_back(bbox_flipped);

        }
    #define SWAP(shape, i, j) do { \
            double tmp = shape.at<double>(i-1, 0); \
            shape.at<double>(i-1, 0) = shape.at<double>(j-1, 0); \
            shape.at<double>(j-1, 0) = tmp; \
            tmp =  shape.at<double>(i-1, 1); \
            shape.at<double>(i-1, 1) = shape.at<double>(j-1, 1); \
            shape.at<double>(j-1, 1) = tmp; \
        } while(0)

        if (params.n_landmarks == 29) {
            for (int i = N; i < (int)gt_shapes.size(); i++) {
                SWAP(gt_shapes[i], 1, 2);
                SWAP(gt_shapes[i], 3, 4);
                SWAP(gt_shapes[i], 5, 7);
                SWAP(gt_shapes[i], 6, 8);
                SWAP(gt_shapes[i], 13, 15);
                SWAP(gt_shapes[i], 9, 10);
                SWAP(gt_shapes[i], 11, 12);
                SWAP(gt_shapes[i], 17, 18);
                SWAP(gt_shapes[i], 14, 16);
                SWAP(gt_shapes[i], 19, 20);
                SWAP(gt_shapes[i], 23, 24);
            }
        }
        else if (params.n_landmarks == 68) {
            for (int i = N; i < (int)gt_shapes.size(); i++) {
                for (int k = 1; k <= 8; k++) SWAP(gt_shapes[i], k, 18 - k);
                for (int k = 18; k <= 22; k++) SWAP(gt_shapes[i], k, 45 - k);
                for (int k = 37; k <= 40; k++) SWAP(gt_shapes[i], k, 83 - k);
                SWAP(gt_shapes[i], 42, 47);
                SWAP(gt_shapes[i], 41, 48);
                SWAP(gt_shapes[i], 32, 36);
                SWAP(gt_shapes[i], 33, 35);
                for (int k = 49; k <= 51; k++) SWAP(gt_shapes[i], k, 104 - k);
                SWAP(gt_shapes[i], 60, 56);
                SWAP(gt_shapes[i], 59, 57);
                SWAP(gt_shapes[i], 61, 65);
                SWAP(gt_shapes[i], 62, 64);
                SWAP(gt_shapes[i], 68, 66);
            }
        }
        else {
            printf("Wrong n_landmarks, it must be 29 or 68");
        }

    #undef SWAP

    }

    FacemarkLBFImpl::BBox::BBox() {}
    FacemarkLBFImpl::BBox::~BBox() {}

    FacemarkLBFImpl::BBox::BBox(double _x, double _y, double w, double h) {
        x = _x;
        y = _y;
        width = w;
        height = h;
        x_center = x + w / 2.;
        y_center = y + h / 2.;
        x_scale = w / 2.;
        y_scale = h / 2.;
    }

    // Project absolute shape to relative shape binding to this bbox
    Mat FacemarkLBFImpl::BBox::project(const Mat &shape) const {
        Mat_<double> res(shape.rows, shape.cols);
        const Mat_<double> &shape_ = (Mat_<double>)shape;
        for (int i = 0; i < shape.rows; i++) {
            res(i, 0) = (shape_(i, 0) - x_center) / x_scale;
            res(i, 1) = (shape_(i, 1) - y_center) / y_scale;
        }
        return res;
    }

    // Project relative shape to absolute shape binding to this bbox
    Mat FacemarkLBFImpl::BBox::reproject(const Mat &shape) const {
        Mat_<double> res(shape.rows, shape.cols);
        const Mat_<double> &shape_ = (Mat_<double>)shape;
        for (int i = 0; i < shape.rows; i++) {
            res(i, 0) = shape_(i, 0)*x_scale + x_center;
            res(i, 1) = shape_(i, 1)*y_scale + y_center;
        }
        return res;
    }

    Mat FacemarkLBFImpl::getMeanShape(std::vector<Mat> &gt_shapes, std::vector<BBox> &bboxes) {

        int N = (int)gt_shapes.size();
        Mat mean_shape = Mat::zeros(gt_shapes[0].rows, 2, CV_64FC1);
        for (int i = 0; i < N; i++) {
            mean_shape += bboxes[i].project(gt_shapes[i]);
        }
        mean_shape /= N;
        return mean_shape;
    }

    // Similarity Transform, project shape2 to shape1
    // p1 ~= scale * rotate * p2, p1 and p2 are vector in math
    void FacemarkLBFImpl::LBF::calcSimilarityTransform(const Mat &shape1, const Mat &shape2, double &scale, Mat &rotate) {
        Mat_<double> rotate_(2, 2);
        double x1_center, y1_center, x2_center, y2_center;
        x1_center = cv::mean(shape1.col(0))[0];
        y1_center = cv::mean(shape1.col(1))[0];
        x2_center = cv::mean(shape2.col(0))[0];
        y2_center = cv::mean(shape2.col(1))[0];

        Mat temp1(shape1.rows, shape1.cols, CV_64FC1);
        Mat temp2(shape2.rows, shape2.cols, CV_64FC1);
        temp1.col(0) = shape1.col(0) - x1_center;
        temp1.col(1) = shape1.col(1) - y1_center;
        temp2.col(0) = shape2.col(0) - x2_center;
        temp2.col(1) = shape2.col(1) - y2_center;

        Mat_<double> covar1, covar2;
        Mat_<double> mean1, mean2;
        calcCovarMatrix(temp1, covar1, mean1, CV_COVAR_COLS);
        calcCovarMatrix(temp2, covar2, mean2, CV_COVAR_COLS);

        double s1 = sqrt(cv::norm(covar1));
        double s2 = sqrt(cv::norm(covar2));
        scale = s1 / s2;
        temp1 /= s1;
        temp2 /= s2;

        double num = temp1.col(1).dot(temp2.col(0)) - temp1.col(0).dot(temp2.col(1));
        double den = temp1.col(0).dot(temp2.col(0)) + temp1.col(1).dot(temp2.col(1));
        double normed = sqrt(num*num + den*den);
        double sin_theta = num / normed;
        double cos_theta = den / normed;
        rotate_(0, 0) = cos_theta; rotate_(0, 1) = -sin_theta;
        rotate_(1, 0) = sin_theta; rotate_(1, 1) = cos_theta;
        rotate = rotate_;
    }

    // Get relative delta_shapes for predicting target
    std::vector<Mat> FacemarkLBFImpl::LBF::getDeltaShapes(std::vector<Mat> &gt_shapes, std::vector<Mat> &current_shapes,
                               std::vector<BBox> &bboxes, Mat &mean_shape) {
        std::vector<Mat> delta_shapes;
        int N = (int)gt_shapes.size();
        delta_shapes.resize(N);
        double scale;
        Mat_<double> rotate;
        for (int i = 0; i < N; i++) {
            delta_shapes[i] = bboxes[i].project(gt_shapes[i]) - bboxes[i].project(current_shapes[i]);
            calcSimilarityTransform(mean_shape, bboxes[i].project(current_shapes[i]), scale, rotate);
            // delta_shapes[i] = scale * delta_shapes[i] * rotate.t();
        }
        return delta_shapes;
    }

    double FacemarkLBFImpl::LBF::calcVariance(const Mat &vec) {
        double m1 = cv::mean(vec)[0];
        double m2 = cv::mean(vec.mul(vec))[0];
        double variance = m2 - m1*m1;
        return variance;
    }

    double FacemarkLBFImpl::LBF::calcVariance(const std::vector<double> &vec) {
        if (vec.size() == 0) return 0.;
        Mat_<double> vec_(vec);
        double m1 = cv::mean(vec_)[0];
        double m2 = cv::mean(vec_.mul(vec_))[0];
        double variance = m2 - m1*m1;
        return variance;
    }

    double FacemarkLBFImpl::LBF::calcMeanError(std::vector<Mat> &gt_shapes, std::vector<Mat> &current_shapes, int landmark_n , std::vector<int> &left, std::vector<int> &right ) {
        int N = (int)gt_shapes.size();

        double e = 0;
        // every train data
        for (int i = 0; i < N; i++) {
            const Mat_<double> &gt_shape = (Mat_<double>)gt_shapes[i];
            const Mat_<double> &current_shape = (Mat_<double>)current_shapes[i];
            double x1, y1, x2, y2;
            x1 = x2 = y1 = y2 = 0;
            for (int j = 0; j < (int)left.size(); j++) {
                x1 += gt_shape(left[j], 0);
                y1 += gt_shape(left[j], 1);
            }
            for (int j = 0; j < (int)right.size(); j++) {
                x2 += gt_shape(right[j], 0);
                y2 += gt_shape(right[j], 1);
            }
            x1 /= left.size(); y1 /= left.size();
            x2 /= right.size(); y2 /= right.size();
            double pupils_distance = sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
            // every landmark
            double e_ = 0;
            for (int j = 0; j < landmark_n; j++) {
                e_ += norm(gt_shape.row(j) - current_shape.row(j));
            }
            e += e_ / pupils_distance;
        }
        e /= N*landmark_n;
        return e;
    }

    /*---------------RandomTree Implementation---------------------*/
    void FacemarkLBFImpl::RandomTree::initTree(int _landmark_id, int _depth, std::vector<int> feats_m, std::vector<double> radius_m) {
        landmark_id = _landmark_id;
        depth = _depth;
        nodes_n = 1 << depth;
        feats = Mat::zeros(nodes_n, 4, CV_64FC1);
        thresholds.resize(nodes_n);

        params_feats_m = feats_m;
        params_radius_m = radius_m;
    }

    void FacemarkLBFImpl::RandomTree::train(std::vector<Mat> &imgs, std::vector<Mat> &current_shapes, std::vector<BBox> &bboxes,
                           std::vector<Mat> &delta_shapes, Mat &mean_shape, std::vector<int> &index, int stage) {
        Mat_<double> delta_shapes_((int)delta_shapes.size(), 2);
        for (int i = 0; i < (int)delta_shapes.size(); i++) {
            delta_shapes_(i, 0) = delta_shapes[i].at<double>(landmark_id, 0);
            delta_shapes_(i, 1) = delta_shapes[i].at<double>(landmark_id, 1);
        }
        splitNode(imgs, current_shapes, bboxes, delta_shapes_, mean_shape, index, 1, stage);
    }

    void FacemarkLBFImpl::RandomTree::splitNode(std::vector<Mat> &imgs, std::vector<Mat> &current_shapes, std::vector<BBox> &bboxes,
                               Mat &delta_shapes, Mat &mean_shape, std::vector<int> &root, int idx, int stage) {

        int N = (int)root.size();
        if (N == 0) {
            thresholds[idx] = 0;
            feats.row(idx).setTo(0);
            std::vector<int> left, right;
            // split left and right child in DFS
            if (2 * idx < feats.rows / 2)
                splitNode(imgs, current_shapes, bboxes, delta_shapes, mean_shape, left, 2 * idx, stage);
            if (2 * idx + 1 < feats.rows / 2)
                splitNode(imgs, current_shapes, bboxes, delta_shapes, mean_shape, right, 2 * idx + 1, stage);
            return;
        }

        int feats_m = params_feats_m[stage];
        double radius_m = params_radius_m[stage];
        Mat_<double> candidate_feats(feats_m, 4);
        RNG rng(getTickCount());
        // generate feature pool
        for (int i = 0; i < feats_m; i++) {
            double x1, y1, x2, y2;
            x1 = rng.uniform(-1., 1.); y1 = rng.uniform(-1., 1.);
            x2 = rng.uniform(-1., 1.); y2 = rng.uniform(-1., 1.);
            if (x1*x1 + y1*y1 > 1. || x2*x2 + y2*y2 > 1.) {
                i--;
                continue;
            }
            candidate_feats[i][0] = x1 * radius_m;
            candidate_feats[i][1] = y1 * radius_m;
            candidate_feats[i][2] = x2 * radius_m;
            candidate_feats[i][3] = y2 * radius_m;
        }
        // calc features
        Mat_<int> densities(feats_m, N);
        for (int i = 0; i < N; i++) {
            double scale;
            Mat_<double> rotate;
            const Mat_<double> &current_shape = (Mat_<double>)current_shapes[root[i]];
            BBox &bbox = bboxes[root[i]];
            Mat &img = imgs[root[i]];
            calcSimilarityTransform(bbox.project(current_shape), mean_shape, scale, rotate);
            for (int j = 0; j < feats_m; j++) {
                double x1 = candidate_feats(j, 0);
                double y1 = candidate_feats(j, 1);
                double x2 = candidate_feats(j, 2);
                double y2 = candidate_feats(j, 3);
                SIMILARITY_TRANSFORM(x1, y1, scale, rotate);
                SIMILARITY_TRANSFORM(x2, y2, scale, rotate);

                x1 = x1*bbox.x_scale + current_shape(landmark_id, 0);
                y1 = y1*bbox.y_scale + current_shape(landmark_id, 1);
                x2 = x2*bbox.x_scale + current_shape(landmark_id, 0);
                y2 = y2*bbox.y_scale + current_shape(landmark_id, 1);
                x1 = max(0., min(img.cols - 1., x1)); y1 = max(0., min(img.rows - 1., y1));
                x2 = max(0., min(img.cols - 1., x2)); y2 = max(0., min(img.rows - 1., y2));
                densities(j, i) = (int)img.at<uchar>(int(y1), int(x1)) - (int)img.at<uchar>(int(y2), int(x2));
            }
        }
        Mat_<int> densities_sorted;
        cv::sort(densities, densities_sorted, SORT_EVERY_ROW + SORT_ASCENDING);
        //select a feat which reduces maximum variance
        double variance_all = (calcVariance(delta_shapes.col(0)) + calcVariance(delta_shapes.col(1)))*N;
        double variance_reduce_max = 0;
        int threshold = 0;
        int feat_id = 0;
        std::vector<double> left_x, left_y, right_x, right_y;
        left_x.reserve(N); left_y.reserve(N);
        right_x.reserve(N); right_y.reserve(N);
        for (int j = 0; j < feats_m; j++) {
            left_x.clear(); left_y.clear();
            right_x.clear(); right_y.clear();
            int threshold_ = densities_sorted(j, (int)(N*rng.uniform(0.05, 0.95)));
            for (int i = 0; i < N; i++) {
                if (densities(j, i) < threshold_) {
                    left_x.push_back(delta_shapes.at<double>(root[i], 0));
                    left_y.push_back(delta_shapes.at<double>(root[i], 1));
                }
                else {
                    right_x.push_back(delta_shapes.at<double>(root[i], 0));
                    right_y.push_back(delta_shapes.at<double>(root[i], 1));
                }
            }
            double variance_ = (calcVariance(left_x) + calcVariance(left_y))*left_x.size() + \
                (calcVariance(right_x) + calcVariance(right_y))*right_x.size();
            double variance_reduce = variance_all - variance_;
            if (variance_reduce > variance_reduce_max) {
                variance_reduce_max = variance_reduce;
                threshold = threshold_;
                feat_id = j;
            }
        }
        thresholds[idx] = threshold;
        feats(idx, 0) = candidate_feats(feat_id, 0); feats(idx, 1) = candidate_feats(feat_id, 1);
        feats(idx, 2) = candidate_feats(feat_id, 2); feats(idx, 3) = candidate_feats(feat_id, 3);
        // generate left and right child
        std::vector<int> left, right;
        left.reserve(N);
        right.reserve(N);
        for (int i = 0; i < N; i++) {
            if (densities(feat_id, i) < threshold) left.push_back(root[i]);
            else right.push_back(root[i]);
        }
        // split left and right child in DFS
        if (2 * idx < feats.rows / 2)
            splitNode(imgs, current_shapes, bboxes, delta_shapes, mean_shape, left, 2 * idx, stage);
        if (2 * idx + 1 < feats.rows / 2)
            splitNode(imgs, current_shapes, bboxes, delta_shapes, mean_shape, right, 2 * idx + 1, stage);
    }


    void FacemarkLBFImpl::RandomTree::write(FILE *fd) {
        // int stat;
        for (int i = 1; i < nodes_n / 2; i++) {
            fwrite(feats.ptr<double>(i), sizeof(double), 4, fd);
            fwrite(&thresholds[i], sizeof(int), 1, fd);
        }
    }

    void FacemarkLBFImpl::RandomTree::read(FILE *fd) {
        size_t status;
        // initialize
        for (int i = 1; i < nodes_n / 2; i++) {
            status = fread(feats.ptr<double>(i), sizeof(double), 4, fd);
            status = fread(&thresholds[i], sizeof(int), 1, fd);
        }
        status = status | status;
    }

    /*---------------RandomForest Implementation---------------------*/
    void FacemarkLBFImpl::RandomForest::initForest(
        int _landmark_n,
        int _trees_n,
        int _tree_depth,
        double _overlap_ratio,
        std::vector<int>_feats_m,
        std::vector<double>_radius_m,
        bool verbose_mode
    ) {
        trees_n = _trees_n;
        landmark_n = _landmark_n;
        tree_depth = _tree_depth;
        overlap_ratio = _overlap_ratio;

        feats_m = _feats_m;
        radius_m = _radius_m;

        verbose = verbose_mode;

        random_trees.resize(landmark_n);
        for (int i = 0; i < landmark_n; i++) {
            random_trees[i].resize(trees_n);
            for (int j = 0; j < trees_n; j++) random_trees[i][j].initTree(i, tree_depth, feats_m, radius_m);
        }
    }

    void FacemarkLBFImpl::RandomForest::train(std::vector<Mat> &imgs, std::vector<Mat> &current_shapes, \
                             std::vector<BBox> &bboxes, std::vector<Mat> &delta_shapes, Mat &mean_shape, int stage) {
        int N = (int)imgs.size();
        int Q = int(N / ((1. - overlap_ratio) * trees_n));

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < landmark_n; i++) {
        TIMER_BEGIN
            std::vector<int> root;
            for (int j = 0; j < trees_n; j++) {
                int start = max(0, int(floor(j*Q - j*Q*overlap_ratio)));
                int end = min(int(start + Q + 1), N);
                int L = end - start;
                root.resize(L);
                for (int k = 0; k < L; k++) root[k] = start + k;
                random_trees[i][j].train(imgs, current_shapes, bboxes, delta_shapes, mean_shape, root, stage);
            }
            if(verbose) printf("Train %2dth of %d landmark Done, it costs %.4lf s\n", i+1, landmark_n, TIMER_NOW);
        TIMER_END
        }
    }

    Mat FacemarkLBFImpl::RandomForest::generateLBF(Mat &img, Mat &current_shape, BBox &bbox, Mat &mean_shape) {
        Mat_<int> lbf_feat(1, landmark_n*trees_n);
        double scale;
        Mat_<double> rotate;
        calcSimilarityTransform(bbox.project(current_shape), mean_shape, scale, rotate);

        int base = 1 << (tree_depth - 1);

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < landmark_n; i++) {
            for (int j = 0; j < trees_n; j++) {
                RandomTree &tree = random_trees[i][j];
                int code = 0;
                int idx = 1;
                for (int k = 1; k < tree.depth; k++) {
                    double x1 = tree.feats(idx, 0);
                    double y1 = tree.feats(idx, 1);
                    double x2 = tree.feats(idx, 2);
                    double y2 = tree.feats(idx, 3);
                    SIMILARITY_TRANSFORM(x1, y1, scale, rotate);
                    SIMILARITY_TRANSFORM(x2, y2, scale, rotate);

                    x1 = x1*bbox.x_scale + current_shape.at<double>(i, 0);
                    y1 = y1*bbox.y_scale + current_shape.at<double>(i, 1);
                    x2 = x2*bbox.x_scale + current_shape.at<double>(i, 0);
                    y2 = y2*bbox.y_scale + current_shape.at<double>(i, 1);
                    x1 = max(0., min(img.cols - 1., x1)); y1 = max(0., min(img.rows - 1., y1));
                    x2 = max(0., min(img.cols - 1., x2)); y2 = max(0., min(img.rows - 1., y2));
                    int density = img.at<uchar>(int(y1), int(x1)) - img.at<uchar>(int(y2), int(x2));
                    code <<= 1;
                    if (density < tree.thresholds[idx]) {
                        idx = 2 * idx;
                    }
                    else {
                        code += 1;
                        idx = 2 * idx + 1;
                    }
                }
                lbf_feat(i*trees_n + j) = (i*trees_n + j)*base + code;
            }
        }
        return lbf_feat;
    }


    void FacemarkLBFImpl::RandomForest::write(FILE *fd) {
        for (int i = 0; i < landmark_n; i++) {
            for (int j = 0; j < trees_n; j++) {
                random_trees[i][j].write(fd);
            }
        }
    }

    void FacemarkLBFImpl::RandomForest::read(FILE *fd)
    {
        for (int i = 0; i < landmark_n; i++) {
            for (int j = 0; j < trees_n; j++) {
                random_trees[i][j].initTree(i, tree_depth, feats_m, radius_m);
                random_trees[i][j].read(fd);
            }
        }
    }


    /*---------------Regressor Implementation---------------------*/
    void FacemarkLBFImpl::Regressor::initRegressor(Params config) {
        stages_n = config.stages_n;
        landmark_n = config.n_landmarks;

        random_forests.resize(stages_n);
        for (int i = 0; i < stages_n; i++)
            random_forests[i].initForest(
                config.n_landmarks,
                config.tree_n,
                config.tree_depth,
                config.bagging_overlap,
                config.feats_m,
                config.radius_m,
                config.verbose
            );

        mean_shape.create(config.n_landmarks, 2, CV_64FC1);

        gl_regression_weights.resize(stages_n);
        int F = config.n_landmarks * config.tree_n * (1 << (config.tree_depth - 1));

        for (int i = 0; i < stages_n; i++) {
            gl_regression_weights[i].create(2 * config.n_landmarks, F, CV_64FC1);
        }
    }

    void FacemarkLBFImpl::Regressor::trainRegressor(std::vector<Mat> &imgs, std::vector<Mat> &gt_shapes, std::vector<Mat> &current_shapes,
                            std::vector<BBox> &bboxes, Mat &mean_shape_, int start_from, Params config) {
        assert(start_from >= 0 && start_from < stages_n);
        mean_shape = mean_shape_;
        int N = (int)imgs.size();

        for (int k = start_from; k < stages_n; k++) {
            std::vector<Mat> delta_shapes = getDeltaShapes(gt_shapes, current_shapes, bboxes, mean_shape);

            // train random forest
            if(config.verbose) printf("training random forest %dth of %d stages, ",k+1, stages_n);
            TIMER_BEGIN
                random_forests[k].train(imgs, current_shapes, bboxes, delta_shapes, mean_shape, k);
                if(config.verbose) printf("costs %.4lf s\n",  TIMER_NOW);
            TIMER_END

            // generate lbf of every train data
            std::vector<Mat> lbfs;
            lbfs.resize(N);
            for (int i = 0; i < N; i++) {
                lbfs[i] = random_forests[k].generateLBF(imgs[i], current_shapes[i], bboxes[i], mean_shape);
            }

            // global regression
            if(config.verbose) printf("start train global regression of %dth stage\n", k);
            TIMER_BEGIN
                globalRegressionTrain(lbfs, delta_shapes, k, config);
                if(config.verbose) printf("end of train global regression of %dth stage, costs %.4lf s\n", k, TIMER_NOW);
            TIMER_END

            // update current_shapes
            double scale;
            Mat rotate;
            for (int i = 0; i < N; i++) {
                Mat delta_shape = globalRegressionPredict(lbfs[i], k);
                calcSimilarityTransform(bboxes[i].project(current_shapes[i]), mean_shape, scale, rotate);
                current_shapes[i] = bboxes[i].reproject(bboxes[i].project(current_shapes[i]) + scale * delta_shape * rotate.t());
            }

            // calc mean error
            double e = calcMeanError(gt_shapes, current_shapes, config.n_landmarks, config.pupils[0],config.pupils[1]);
            if(config.verbose) printf("Train %dth stage Done with Error = %lf\n", k, e);

        } // for int k
    }//Regressor::training

    // Global Regression to predict delta shape with LBF
    void FacemarkLBFImpl::Regressor::globalRegressionTrain(std::vector<Mat> &lbfs, std::vector<Mat> &delta_shapes, int stage, Params config) {
        int N = (int)lbfs.size();
        int M = lbfs[0].cols;
        int F = config.n_landmarks*config.tree_n*(1 << (config.tree_depth - 1));
        int landmark_n_ = delta_shapes[0].rows;
        // prepare linear regression config X and Y
        struct liblinear::feature_node **X = (struct liblinear::feature_node **)malloc(N * sizeof(struct liblinear::feature_node *));
        double **Y = (double **)malloc(landmark_n_ * 2 * sizeof(double *));
        for (int i = 0; i < N; i++) {
            X[i] = (struct liblinear::feature_node *)malloc((M + 1) * sizeof(struct liblinear::feature_node));
            for (int j = 0; j < M; j++) {
                X[i][j].index = lbfs[i].at<int>(0, j) + 1; // index starts from 1
                X[i][j].value = 1;
            }
            X[i][M].index = -1;
            X[i][M].value = -1;
        }
        for (int i = 0; i < landmark_n_; i++) {
            Y[2 * i] = (double *)malloc(N*sizeof(double));
            Y[2 * i + 1] = (double *)malloc(N*sizeof(double));
            for (int j = 0; j < N; j++) {
                Y[2 * i][j] = delta_shapes[j].at<double>(i, 0);
                Y[2 * i + 1][j] = delta_shapes[j].at<double>(i, 1);
            }
        }
        // train every landmark
        struct liblinear::problem prob;
        struct liblinear::parameter param;
        prob.l = N;
        prob.n = F;
        prob.x = X;
        prob.bias = -1;
        param.solver_type = liblinear::L2R_L2LOSS_SVR_DUAL;
        param.C = 1. / N;
        param.p = 0;
        param.eps = 0.00001;

        Mat_<double> weight(2 * landmark_n_, F);

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < landmark_n_; i++) {

        #define FREE_MODEL(model)   \
        free(model->w);         \
        free(model->label);     \
        free(model)

            if(config.verbose) printf("train %2dth landmark\n", i);
            struct liblinear::problem prob_ = prob;
            prob_.y = Y[2 * i];
            liblinear::check_parameter(&param);
            struct liblinear::model *model = liblinear::train(&prob_, &param);
            for (int j = 0; j < F; j++) weight(2 * i, j) = liblinear::get_decfun_coef(model, j + 1, 0);
            FREE_MODEL(model);

            prob_.y = Y[2 * i + 1];
            liblinear::check_parameter(&param);
            model = liblinear::train(&prob_, &param);
            for (int j = 0; j < F; j++) weight(2 * i + 1, j) = liblinear::get_decfun_coef(model, j + 1, 0);
            FREE_MODEL(model);

        #undef FREE_MODEL

        }

        gl_regression_weights[stage] = weight;

        // free
        for (int i = 0; i < N; i++) free(X[i]);
        for (int i = 0; i < 2 * landmark_n_; i++) free(Y[i]);
        free(X);
        free(Y);
    } // Regressor:globalRegressionTrain

    Mat FacemarkLBFImpl::Regressor::globalRegressionPredict(const Mat &lbf, int stage) {
        const Mat_<double> &weight = (Mat_<double>)gl_regression_weights[stage];
        Mat_<double> delta_shape(weight.rows / 2, 2);
        const double *w_ptr = NULL;
        const int *lbf_ptr = lbf.ptr<int>(0);

        //#pragma omp parallel for num_threads(2) private(w_ptr)
        for (int i = 0; i < delta_shape.rows; i++) {
            w_ptr = weight.ptr<double>(2 * i);
            double y = 0;
            for (int j = 0; j < lbf.cols; j++) y += w_ptr[lbf_ptr[j]];
            delta_shape(i, 0) = y;

            w_ptr = weight.ptr<double>(2 * i + 1);
            y = 0;
            for (int j = 0; j < lbf.cols; j++) y += w_ptr[lbf_ptr[j]];
            delta_shape(i, 1) = y;
        }
        return delta_shape;
    } // Regressor::globalRegressionPredict

    Mat FacemarkLBFImpl::Regressor::predict(Mat &img, BBox &bbox) {
        Mat current_shape = bbox.reproject(mean_shape);
        double scale;
        Mat rotate;
        Mat lbf_feat;
        for (int k = 0; k < stages_n; k++) {
            // generate lbf
            lbf_feat = random_forests[k].generateLBF(img, current_shape, bbox, mean_shape);
            // update current_shapes
            Mat delta_shape = globalRegressionPredict(lbf_feat, k);
            delta_shape = delta_shape.reshape(0, landmark_n);
            calcSimilarityTransform(bbox.project(current_shape), mean_shape, scale, rotate);
            current_shape = bbox.reproject(bbox.project(current_shape) + scale * delta_shape * rotate.t());
        }
        return current_shape;
    } // Regressor::predict

    void FacemarkLBFImpl::Regressor::write(FILE *fd, Params config) {

        // global parameters
        fwrite(&config.stages_n, sizeof(int), 1, fd);
        fwrite(&config.tree_n, sizeof(int), 1, fd);
        fwrite(&config.tree_depth, sizeof(int), 1, fd);
        fwrite(&config.n_landmarks, sizeof(int), 1, fd);
        // mean_shape
        double *ptr = NULL;
        for (int i = 0; i < mean_shape.rows; i++) {
            ptr = mean_shape.ptr<double>(i);
            fwrite(ptr, sizeof(double), mean_shape.cols, fd);
        }
        // every stages
        for (int k = 0; k < config.stages_n; k++) {
            if(config.verbose) printf("Write %dth stage\n", k);
            random_forests[k].write(fd);
            for (int i = 0; i < 2 * config.n_landmarks; i++) {
                ptr = gl_regression_weights[k].ptr<double>(i);
                fwrite(ptr, sizeof(double), gl_regression_weights[k].cols, fd);
            }
        }
    }

    void FacemarkLBFImpl::Regressor::read(FILE *fd, Params & config){

        size_t status = fread(&config.stages_n, sizeof(int), 1, fd);
        status = fread(&config.tree_n, sizeof(int), 1, fd);
        status = fread(&config.tree_depth, sizeof(int), 1, fd);
        status = fread(&config.n_landmarks, sizeof(int), 1, fd);
        stages_n = config.stages_n;
        landmark_n = config.n_landmarks;

        initRegressor(config);

        // mean_shape
        double *ptr = NULL;

        for (int i = 0; i < mean_shape.rows; i++) {
            ptr = mean_shape.ptr<double>(i);
            status = fread(ptr, sizeof(double), mean_shape.cols, fd);
        }

        // every stages
        for (int k = 0; k < stages_n; k++) {
            random_forests[k].initForest(
                config.n_landmarks,
                config.tree_n,
                config.tree_depth,
                config.bagging_overlap,
                config.feats_m,
                config.radius_m,
                config.verbose
            );
            random_forests[k].read(fd);
            for (int i = 0; i < 2 * config.n_landmarks; i++) {
                ptr = gl_regression_weights[k].ptr<double>(i);
                status = fread(ptr, sizeof(double), gl_regression_weights[k].cols, fd);
            }
        }
        status = status | status;
    }

    #undef TIMER_BEGIN
    #undef TIMER_NOW
    #undef TIMER_END
    #undef SIMILARITY_TRANSFORM
} /* namespace face */
} /* namespace cv */
