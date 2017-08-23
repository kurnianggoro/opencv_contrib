#include "opencv2/face.hpp"
#include "opencv2/imgcodecs.hpp"
#include "precomp.hpp"

namespace cv {
namespace face {

    /*
    * Parameters
    */
    FacemarkAAM::Params::Params(){
        // detect_thresh = 0.5;
        // sigma=0.2;
    }

    // void FacemarkAAM::Params::read( const cv::FileNode& fn ){
    //     *this = FacemarkAAM::Params();
    //
    //     if (!fn["detect_thresh"].empty())
    //         fn["detect_thresh"] >> detect_thresh;
    //
    //     if (!fn["sigma"].empty())
    //         fn["sigma"] >> sigma;
    //
    // }
    //
    // void FacemarkAAM::Params::write( cv::FileStorage& fs ) const{
    //     fs << "detect_thresh" << detect_thresh;
    //     fs << "sigma" << sigma;
    // }

    class FacemarkAAMImpl : public FacemarkAAM {
    public:
        FacemarkAAMImpl( const FacemarkAAM::Params &parameters = FacemarkAAM::Params() );
        // void read( const FileNode& /*fn*/ );
        // void write( FileStorage& /*fs*/ ) const;

        void saveModel(String fs);
        void loadModel(String fs);

        bool setFaceDetector(bool(*f)(InputArray , OutputArray ));
        bool getFaces( InputArray image ,OutputArray faces);

    protected:

        bool fit( InputArray image, InputArray faces, InputOutputArray landmarks );//!< from many ROIs
        bool fitSingle( InputArray image, std::vector<Point2f>& landmarks, Mat R, Point2f T, float scale );
        bool fitImpl( const Mat image, std::vector<Point2f>& landmarks, Mat R, Point2f T, float scale );

        // void trainingImpl(String imageList, String groundTruth, const FacemarkAAM::Params &parameters);
        void training(String imageList, String groundTruth);

        Mat procrustes(std::vector<Point2f> , std::vector<Point2f> , Mat & , Scalar & , float & );
        void calcMeanShape(std::vector<std::vector<Point2f> > ,std::vector<Point2f> & );
        void procrustesAnalysis(std::vector<std::vector<Point2f> > , std::vector<std::vector<Point2f> > & , std::vector<Point2f> & );

        inline Mat linearize(Mat );
        inline Mat linearize(std::vector<Point2f> );
        void getProjection(const Mat , Mat &, int );
        void calcSimilarityEig(std::vector<Point2f> ,Mat , Mat & , Mat & );
        Mat orthonormal(Mat );
        void delaunay(std::vector<Point2f> , std::vector<Vec3i> & );
        Mat createMask(std::vector<Point2f> , Rect );
        Mat createTextureBase(std::vector<Point2f> , std::vector<Vec3i> , Rect , std::vector<std::vector<Point> > & );
        Mat warpImage(Mat , std::vector<Point2f> , std::vector<Point2f> , std::vector<Vec3i> , Rect , std::vector<std::vector<Point> > );
        template <class T>
        Mat getFeature(const Mat , std::vector<int> map);
        void createMaskMapping(const Mat mask, const Mat mask2,  std::vector<int> & , std::vector<int> &, std::vector<int> &);

        void warpUpdate(std::vector<Point2f> & shape, Mat delta, std::vector<Point2f> s0, Mat S, Mat Q, std::vector<Vec3i> triangles,std::vector<std::vector<int> > Tp);
        Mat computeWarpParts(std::vector<Point2f> curr_shape,std::vector<Point2f> s0, Mat ds0, std::vector<Vec3i> triangles,std::vector<std::vector<int> > Tp);
        void image_jacobian(const Mat gx, const Mat gy, const Mat Jx, const Mat Jy, Mat & G);
        void gradient(const Mat M, Mat & gx, Mat & gy);
        void createWarpJacobian(Mat S, Mat Q,  std::vector<Vec3i> , Model::Texture & T, Mat & Wx_dp, Mat & Wy_dp, std::vector<std::vector<int> > & Tp);

        FacemarkAAM::Params params;
        FacemarkAAM::Model AAM;
        bool(*faceDetector)(InputArray , OutputArray);
        bool isSetDetector;

    private:
        bool isModelTrained;
    };

    /*
    * Constructor
    */
    Ptr<FacemarkAAM> FacemarkAAM::create(const FacemarkAAM::Params &parameters){
        return Ptr<FacemarkAAMImpl>(new FacemarkAAMImpl(parameters));
    }

    FacemarkAAMImpl::FacemarkAAMImpl( const FacemarkAAM::Params &parameters ) :
        params( parameters )
    {
        isSetDetector =false;
        isModelTrained = false;
    }

    // void FacemarkAAMImpl::read( const cv::FileNode& fn ){
    //     params.read( fn );
    // }
    //
    // void FacemarkAAMImpl::write( cv::FileStorage& fs ) const {
    //     params.write( fs );
    // }

    bool FacemarkAAMImpl::setFaceDetector(bool(*f)(InputArray , OutputArray )){
        faceDetector = f;
        isSetDetector = true;
        printf("face detector is configured\n");
        return true;
    }


    bool FacemarkAAMImpl::getFaces( InputArray image , OutputArray roi){

        if(!isSetDetector){
            return false;
        }

        std::vector<Rect> faces;
        faces.clear();

        faceDetector(image.getMat(), faces);
        Mat(faces).copyTo(roi);
        return true;
    }


    // void FacemarkAAM::training(String imageList, String groundTruth, const FacemarkAAM::Params &parameters){
    //     trainingImpl(imageList, groundTruth, parameters);
    // }

    // void FacemarkAAMImpl::trainingImpl(String imageList, String groundTruth, const FacemarkAAM::Params &parameters){
    //     params = parameters;
    //     trainingImpl(imageList, groundTruth);
    // }

    void FacemarkAAMImpl::training(String imageList, String groundTruth){

        std::vector<String> images;
        std::vector<std::vector<Point2f> > facePoints, normalized;
        Mat erode_kernel = getStructuringElement(MORPH_RECT, Size(3,3), Point(1,1));
        Mat image;

        int param_max_m = 550;
        int param_max_n = 136;
        float offset = -0.0;

        /* initialize the values TODO: set them based on the params*/
        AAM.scales.push_back(1);
        AAM.scales.push_back(2);
        AAM.textures.resize(AAM.scales.size());

        /*-------------- A. Load the training data---------*/
        if(groundTruth==""){
            loadTrainingData(imageList, images, facePoints);
        }else{
            loadTrainingData(imageList, groundTruth, images, facePoints, offset);
        }
        procrustesAnalysis(facePoints, normalized,AAM.s0);

        /*-------------- B. Create the shape model---------*/
        Mat s0_lin = linearize(AAM.s0).t() ;
        // linearize all shapes  data, all x and then all y for each shape
        Mat M;
        for(unsigned i=0;i<normalized.size();i++){
            M.push_back(linearize(normalized[i]).t()-s0_lin);
        }

        /* get PCA Projection vectors */
        Mat S;
        getProjection(M.t(),S,param_max_n);
        /* Create similarity eig*/
        Mat shape_S,shape_Q;
        calcSimilarityEig(AAM.s0,S,AAM.Q,AAM.S);

        /* ----------C. Create the coordinate frame ------------*/
        delaunay(AAM.s0,AAM.triangles);

        for(size_t scale=0; scale<AAM.scales.size();scale++){
            AAM.textures[scale].max_m = 145;
            printf("Training for scale %i ...\n", AAM.scales[scale]);
            Mat s0_scaled_m = Mat(AAM.s0)/AAM.scales[scale]; // scale the shape
            std::vector<Point2f> s0_scaled = s0_scaled_m.reshape(2); //convert to points

            /*get the min and max of x and y coordinate*/
            double min_x, max_x, min_y, max_y;
            s0_scaled_m = s0_scaled_m.reshape(1);
            Mat s0_scaled_x = s0_scaled_m.col(0);
            Mat s0_scaled_y = s0_scaled_m.col(1);
            minMaxIdx(s0_scaled_x, &min_x, &max_x);
            minMaxIdx(s0_scaled_y, &min_y, &max_y);

            std::vector<Point2f> base_shape = Mat(Mat(s0_scaled)-Scalar(min_x-2.0,min_y-2.0)).reshape(2);
            AAM.textures[scale].base_shape = base_shape;
            AAM.textures[scale].resolution = Rect(0,0,(int)ceil(max_x-min_x+3),(int)ceil(max_y-min_y+3));

            Mat base_texture = createTextureBase(base_shape, AAM.triangles, AAM.textures[scale].resolution, AAM.textures[scale].textureIdx);

            Mat mask1 = base_texture>0;
            Mat mask2;
            erode(mask1, mask1, erode_kernel);
            erode(mask1, mask2, erode_kernel);

            Mat warped;
            std::vector<int> fe_map;
            createMaskMapping(mask1,mask2, AAM.textures[scale].ind1, AAM.textures[scale].ind2,fe_map);//ok

            /* ------------ Part D. Get textures -------------*/
            Mat texture_feats, feat;
            for(size_t i=0; i<images.size();i++){
                image = imread(images[i]);
                warped = warpImage(image,base_shape, facePoints[i], AAM.triangles, AAM.textures[scale].resolution,AAM.textures[scale].textureIdx);
                feat = getFeature<uchar>(warped, AAM.textures[scale].ind1);
                texture_feats.push_back(feat.t());
            }
            Mat T= texture_feats.t();

            /* -------------- E. Create the texture model -----------------*/
            reduce(T,AAM.textures[scale].A0,1, CV_REDUCE_AVG);

            Mat A0_mtx = repeat(AAM.textures[scale].A0,1,T.cols);
            Mat textures_normalized = T - A0_mtx;

            getProjection(textures_normalized, AAM.textures[scale].A ,param_max_m);
            AAM.textures[scale].AA0 = getFeature<float>(AAM.textures[scale].A0, fe_map);

            Mat U_data, ud;
            for(int i =0;i<AAM.textures[scale].A.cols;i++){
                Mat c = AAM.textures[scale].A.col(i);
                ud = getFeature<float>(c,fe_map);
                U_data.push_back(ud.t());
            }
            Mat U = U_data.t();
            AAM.textures[scale].AA = orthonormal(U);
        } // scale

        saveModel("AAM.yml");
        isModelTrained = true;
        printf("training is finished\n");
    }

    bool FacemarkAAMImpl::fit( InputArray image, InputArray roi, InputOutputArray _landmarks )
    {
        std::vector<Rect> faces = roi.getMat();
        std::vector<std::vector<Point2f> > & landmarks =
            *(std::vector<std::vector<Point2f> >*) _landmarks.getObj();
        landmarks.resize(faces.size());

        Mat R =  Mat::eye(2, 2, CV_32F);
        Point2f t = Point2f(0,0);
        float scale = 1.0;
        for(unsigned i=0; i<faces.size();i++){
            fitImpl(image.getMat(), landmarks[i], R, t, scale);
        }

        return true;
    }

    // bool FacemarkAAMImpl::fit( const Mat image, std::vector<Point2f>& landmarks){
    //     /*temporary values...will be updated*/
    //     Mat R =  Mat::eye(2, 2, CV_32F);
    //     Point2f t = Point2f(0,0);
    //     float scale = 1.0;
    //
    //     return fit(image, landmarks, R, t, scale);
    // }
    //
    bool FacemarkAAMImpl::fitSingle( InputArray image, std::vector<Point2f>& landmarks, Mat R, Point2f T, float scale ){
        return fitImpl(image.getMat(), landmarks, R, T, scale);
    }

    bool FacemarkAAMImpl::fitImpl( const Mat image, std::vector<Point2f>& landmarks, Mat R, Point2f T, float scale ){
        if (landmarks.size()>0)
            landmarks.clear();

        CV_Assert(isModelTrained);

        int param_n = 10, param_m = 200;

        /*variables*/
        std::vector<Point2f> s0 = AAM.s0;

        /*pre-computation*/
        Mat S = Mat(AAM.S, Range::all(), Range(0,param_n)).clone(); // chop the shape data
        std::vector<std::vector<int> > Tp;
        Mat Wx_dp, Wy_dp;
        createWarpJacobian(S, AAM.Q, AAM.triangles, AAM.textures[0],Wx_dp, Wy_dp, Tp);

        // initial fitting
        // Mat initial = Mat(scale*(Mat(s0))+Scalar(T.x,T.y)).reshape(1);
        // Mat base_shape = Mat(R*initial.t()).t();
        // std::vector<Point2f> curr_shape = Mat(1.0/scale*(base_shape)).reshape(2);
        std::vector<Point2f> s0_init = Mat(Mat(R*scale*Mat(Mat(s0).reshape(1)).t()).t()).reshape(2);
        std::vector<Point2f> s0_align =  Mat(Mat(s0_init)+Scalar(T.x,T.y));
        std::vector<Point2f> curr_shape = Mat(1.0/scale*Mat(s0_align)).reshape(2);

        Mat imgray;
        Mat img;
        if(image.channels()>1){
            cvtColor(image,imgray,CV_BGR2GRAY);
        }else{
            imgray = image;
        }

        resize(imgray,img,Size(int(image.cols/scale),int(image.rows/scale)));// matlab use bicubic interpolation, the result is float numbers

        /*chop the textures model*/
        Mat A = Mat(AAM.textures[0].A,Range(0,AAM.textures[0].A.rows), Range(0,param_m)).clone();
        Mat AA = Mat(AAM.textures[0].AA,Range(0,AAM.textures[0].AA.rows), Range(0,param_m)).clone();

        /*iteratively update the fitting*/
        Mat I, II, warped, c, gx, gy, Irec, Irec_feat, dc;
        Mat refI, refII, refWarped, ref_c, ref_gx, ref_gy, refIrec, refIrec_feat, ref_dc ;
        for(int t=0;t<50;t++){
            warped = warpImage(img,AAM.textures[0].base_shape, curr_shape,
                               AAM.triangles,
                               AAM.textures[0].resolution ,
                               AAM.textures[0].textureIdx);

            I = getFeature<uchar>(warped, AAM.textures[0].ind1);
            II = getFeature<uchar>(warped, AAM.textures[0].ind2);

            if(t==0){
                c = A.t()*(I-AAM.textures[0].A0); //little bit different to matlab, probably due to datatype
            }else{
                c = c+dc;
            }

            Irec_feat = (AAM.textures[0].A0+A*c);
            Irec = Mat::zeros(AAM.textures[0].resolution.width, AAM.textures[0].resolution.height, CV_32FC1);

            for(int j=0;j<(int)AAM.textures[0].ind1.size();j++){
                Irec.at<float>(AAM.textures[0].ind1[j]) = Irec_feat.at<float>(j);
            }
            Mat irec = Irec.t();

            gradient(irec, gx, gy);

            Mat Jc;
            image_jacobian(Mat(gx.t()).reshape(0,1).t(),Mat(gy.t()).reshape(0,1).t(),Wx_dp, Wy_dp,Jc);

            Mat J;
            std::vector<float> Irec_vec;
            for(size_t j=0;j<AAM.textures[0].ind2.size();j++){
                J.push_back(Jc.row(AAM.textures[0].ind2[j]));
                Irec_vec.push_back(Irec.at<float>(AAM.textures[0].ind2[j]));
            }

            /*compute Jfsic and Hfsic*/
            Mat Jfsic = J - AA*(AA.t()*J);
            Mat Hfsic = Jfsic.t()*Jfsic;
            Mat iHfsic;
            invert(Hfsic, iHfsic);

            /*compute dp dq and dc*/
            Mat dqp = iHfsic*Jfsic.t()*(II-AAM.textures[0].AA0);
            dc = AA.t()*(II-Mat(Irec_vec)-J*dqp);
            warpUpdate(curr_shape, dqp, s0,S, AAM.Q, AAM.triangles,Tp);
        }
        landmarks = Mat(scale*Mat(curr_shape)).reshape(2);
        return true;
    }

    void FacemarkAAMImpl::saveModel(String s){
        FileStorage fs(s.c_str(),FileStorage::WRITE);
        fs << "AAM_tri" << AAM.triangles;
        fs << "scales" << AAM.scales;
        fs << "s0" << AAM.s0;
        fs << "S" << AAM.S;
        fs << "Q" << AAM.Q;

        char x[256];
        for(int i=0;i< (int)AAM.scales.size();i++){
            sprintf(x,"scale%i_max_m",i);
            fs << x << AAM.textures[i].max_m;

            sprintf(x,"scale%i_resolution",i);
            fs << x << AAM.textures[i].resolution;

            sprintf(x,"scale%i_textureIdx",i);
            fs << x << AAM.textures[i].textureIdx;

            sprintf(x,"scale%i_base_shape",i);
            fs << x << AAM.textures[i].base_shape;

            sprintf(x,"scale%i_A",i);
            fs << x << AAM.textures[i].A;

            sprintf(x,"scale%i_A0",i);
            fs << x << AAM.textures[i].A0;

            sprintf(x,"scale%i_AA",i);
            fs << x << AAM.textures[i].AA;

            sprintf(x,"scale%i_AA0",i);
            fs << x << AAM.textures[i].AA0;

            sprintf(x,"scale%i_ind1",i);
            fs << x << AAM.textures[i].ind1;

            sprintf(x,"scale%i_ind2",i);
            fs << x << AAM.textures[i].ind2;

        }
        fs.release();
        printf("The model is successfully saved! \n");
    }

    void FacemarkAAMImpl::loadModel(String s){
        FileStorage fs(s.c_str(),FileStorage::READ);
        char x[256];
        fs["AAM_tri"] >> AAM.triangles;
        fs["scales"] >> AAM.scales;
        fs["s0"] >> AAM.s0;
        fs["S"] >> AAM.S;
        fs["Q"] >> AAM.Q;


        AAM.textures.resize(AAM.scales.size());
        for(int i=0;i< (int)AAM.scales.size();i++){
            sprintf(x,"scale%i_max_m",i);
            fs[x] >> AAM.textures[i].max_m;

            sprintf(x,"scale%i_resolution",i);
            fs[x] >> AAM.textures[i].resolution;

            sprintf(x,"scale%i_textureIdx",i);
            fs[x] >> AAM.textures[i].textureIdx;

            sprintf(x,"scale%i_base_shape",i);
            fs[x] >> AAM.textures[i].base_shape;

            sprintf(x,"scale%i_A",i);
            fs[x] >> AAM.textures[i].A;

            sprintf(x,"scale%i_A0",i);
            fs[x] >> AAM.textures[i].A0;

            sprintf(x,"scale%i_AA",i);
            fs[x] >> AAM.textures[i].AA;

            sprintf(x,"scale%i_AA0",i);
            fs[x] >> AAM.textures[i].AA0;

            sprintf(x,"scale%i_ind1",i);
            fs[x] >> AAM.textures[i].ind1;

            sprintf(x,"scale%i_ind2",i);
            fs[x] >> AAM.textures[i].ind2;
        }

        fs.release();
        isModelTrained = true;
        printf("the model has been loaded\n");
    }

    Mat FacemarkAAMImpl::procrustes(std::vector<Point2f> P, std::vector<Point2f> Q, Mat & rot, Scalar & trans, float & scale){

        // calculate average
        Scalar mx = mean(P);
        Scalar my = mean(Q);

        // zero centered data
        Mat X0 = Mat(P) - mx;
        Mat Y0 = Mat(Q) - my;

        // calculate magnitude
        Mat Xs, Ys;
        multiply(X0,X0,Xs);
        multiply(Y0,Y0,Ys);

        // cout<<Xs<<endl;

        // calculate the sum
        Mat sumXs, sumYs;
        reduce(Xs,sumXs, 0, CV_REDUCE_SUM);
        reduce(Ys,sumYs, 0, CV_REDUCE_SUM);

        //calculate the normrnd
        double normX = sqrt(sumXs.at<float>(0)+sumXs.at<float>(1));
        double normY = sqrt(sumYs.at<float>(0)+sumYs.at<float>(1));

        //normalization
        X0 = X0/normX;
        Y0 = Y0/normY;

        //reshape, convert to 2D Matrix
        Mat Xn=X0.reshape(1);
        Mat Yn=Y0.reshape(1);

        //calculate the covariance matrix
        Mat M = Xn.t()*Yn;

        // decompose
        Mat U,S,Vt;
        SVD::compute(M, S, U, Vt);

        // extract the transformations
        scale = (S.at<float>(0)+S.at<float>(1))*(float)normX/(float)normY;
        rot = Vt.t()*U.t();

        Mat muX(mx),mX; muX.pop_back();muX.pop_back();
        Mat muY(my),mY; muY.pop_back();muY.pop_back();
        muX.convertTo(mX,CV_32FC1);
        muY.convertTo(mY,CV_32FC1);

        Mat t = mX.t()-scale*mY.t()*rot;
        trans[0] = t.at<float>(0);
        trans[1] = t.at<float>(1);

        // calculate the recovered form
        Mat Qmat = Mat(Q).reshape(1);

        return Mat(scale*Qmat*rot+trans).clone();
    }

    void FacemarkAAMImpl::procrustesAnalysis(std::vector<std::vector<Point2f> > shapes, std::vector<std::vector<Point2f> > & normalized, std::vector<Point2f> & new_mean){

        std::vector<Scalar> mean_every_shape;
        mean_every_shape.resize(shapes.size());

        Point2f temp;

        // calculate the mean of every shape
        for(size_t i=0; i< shapes.size();i++){
            mean_every_shape[i] = mean(shapes[i]);
        }

        //normalize every shapes
        Mat tShape;
        normalized.clear();
        for(size_t i=0; i< shapes.size();i++){
            normalized.push_back((Mat)(Mat(shapes[i]) - mean_every_shape[i]));
        }

        // calculate the mean shape
        std::vector<Point2f> mean_shape;
        calcMeanShape(normalized, mean_shape);

        // update the mean shape and normalized shapes iteratively
        int maxIter = 100;
        Mat R;
        Scalar t;
        float s;
        Mat aligned;
        for(int i=0;i<maxIter;i++){
            // align
            for(unsigned k=0;k< normalized.size();k++){
                aligned=procrustes(mean_shape, normalized[k], R, t, s);
                aligned.reshape(2).copyTo(normalized[k]);
            }

            //calc new mean
            calcMeanShape(normalized, new_mean);
            // align the new mean
            aligned=procrustes(mean_shape, new_mean, R, t, s);
            // update
            aligned.reshape(2).copyTo(mean_shape);
        }
    }

    void FacemarkAAMImpl::calcMeanShape(std::vector<std::vector<Point2f> > shapes,std::vector<Point2f> & mean){
        mean.resize(shapes[0].size());
        Point2f tmp;
        for(unsigned i=0;i<shapes[0].size();i++){
            tmp.x=0;
            tmp.y=0;
            for(unsigned k=0;k< shapes.size();k++){
                tmp.x+= shapes[k][i].x;
                tmp.y+= shapes[k][i].y;
            }
            tmp.x/=shapes.size();
            tmp.y/=shapes.size();
            mean[i] = tmp;
        }
    }

    void FacemarkAAMImpl::getProjection(const Mat M, Mat & P,  int n){
        Mat U,S,Vt,S1, Ut;
        int k;
        if(M.rows < M.cols){
            // SVD::compute(M*M.t(), S, U, Vt);
            eigen(M*M.t(), S, Ut); U=Ut.t();

            // find the minimum between number of non-zero eigval,
            // compressed dim, row, and column
            // threshold(S,S1,0.00001,1,THRESH_BINARY);
            k= S.rows; //countNonZero(S1);
            if(k>n)k=n;
            if(k>M.rows)k=M.rows;
            if(k>M.cols)k=M.cols;

            // cut the column of eigen vector
            U.colRange(0,k).copyTo(P);
        }else{
            // SVD::compute(M.t()*M, S, U, Vt);
            eigen(M.t()*M, S, Ut);U=Ut.t();

            // threshold(S,S1,0.00001,1,THRESH_BINARY);
            k= S.rows; //countNonZero(S1);
            if(k>n)k=n;
            if(k>M.rows)k=M.rows;
            if(k>M.cols)k=M.cols;

            // cut the eigen values to k-amount
            Mat D = Mat::zeros(k,k,CV_32FC1);
            Mat diag = D.diag();
            Mat s; pow(S,-0.5,s);
            s(Range(0,k), Range::all()).copyTo(diag);

            // cut the eigen vector to k-column,
            P = Mat(M*U.colRange(0,k)*D).clone();

        }
    }

    Mat FacemarkAAMImpl::orthonormal(Mat Mo){
        Mat M;
        Mo.convertTo(M,CV_32FC1);

        // TODO: float precission is only 1e-7, but MATLAB version use thresh=2.2204e-16
        float thresh = (float)2.2204e-6;

        Mat O = Mat::zeros(M.rows, M.cols, CV_32FC1);

        int k = 0; //storing index

        Mat w,nv;
        float n;
        for(int i=0;i<M.cols;i++){
            Mat v = M.col(i); // processed column to orthogonalize

            // subtract projection over previous vectors
            for(int j=0;j<k;j++){
                Mat o=O.col(j);
                w = v-o*(o.t()*v);
                w.copyTo(v);
            }

            // only keep non zero vector
            n = (float)norm(v);
            if(n>thresh){
                Mat ok=O.col(k);
                // nv=v/n;
                normalize(v,nv);
                nv.copyTo(ok);
                k+=1;
            }

        }

        return O.colRange(0,k).clone();
    }

    void FacemarkAAMImpl::calcSimilarityEig(std::vector<Point2f> s0,Mat S, Mat & Q_orth, Mat & S_orth){
        // cout<<"similarity eig"<<endl;
        int npts = (int)s0.size();

        Mat Q = Mat::zeros(2*npts,4,CV_32FC1);
        Mat c0 = Q.col(0);
        Mat c1 = Q.col(1);
        Mat c2 = Q.col(2);
        Mat c3 = Q.col(3);

        /*c0 = s0(:)*/
        Mat w = linearize(s0);
        // w.convertTo(w, CV_64FC1);
        w.copyTo(c0);

        /*c1 = [-s0(npts:2*npts); s0(0:npts-1)]*/
        Mat s0_mat = Mat(s0).reshape(1);
        // s0_mat.convertTo(s0_mat, CV_64FC1);
        Mat swapper = Mat::zeros(2,npts,CV_32FC1);
        Mat s00 = s0_mat.col(0);
        Mat s01 = s0_mat.col(1);
        Mat sw0 = swapper.row(0);
        Mat sw1 = swapper.row(1);
        Mat(s00.t()).copyTo(sw1);
        s01 = -s01;
        Mat(s01.t()).copyTo(sw0);

        Mat(swapper.reshape(1,2*npts)).copyTo(c1);

        /*c2 - [ones(npts); zeros(npts)]*/
        Mat ones = Mat::ones(1,npts,CV_32FC1);
        Mat c2_mat = Mat::zeros(2,npts,CV_32FC1);
        Mat c20 = c2_mat.row(0);
        ones.copyTo(c20);
        Mat(c2_mat.reshape(1,2*npts)).copyTo(c2);

        /*c3 - [zeros(npts); ones(npts)]*/
        Mat c3_mat = Mat::zeros(2,npts,CV_32FC1);
        Mat c31 = c3_mat.row(1);
        ones.copyTo(c31);
        Mat(c3_mat.reshape(1,2*npts)).copyTo(c3);
        // cout<<Q<<endl;

        Mat Qo = orthonormal(Q);
        // cout<<Qo<<endl;
        // matInfo(Qo);

        Mat all = Qo.t();
        all.push_back(S.t());

        Mat allOrth = orthonormal(all.t());
        // cout<<allOrth.colRange(0,8)<<endl;
        Q_orth =  allOrth.colRange(0,4).clone();
        S_orth =  allOrth.colRange(4,allOrth.cols).clone();

    }

    inline Mat FacemarkAAMImpl::linearize(Mat s){ // all x values and then all y values
        return Mat(s.reshape(1).t()).reshape(1,2*s.rows);
    }
    inline Mat FacemarkAAMImpl::linearize(std::vector<Point2f> s){ // all x values and then all y values
        return linearize(Mat(s));
    }

    void FacemarkAAMImpl::delaunay(std::vector<Point2f> s, std::vector<Vec3i> & triangles){

        triangles.clear();

        std::vector<int> idx;
        std::vector<Vec6f> tp;

        double min_x, max_x, min_y, max_y;
        Mat S = Mat(s).reshape(1);
        Mat s_x = S.col(0);
        Mat s_y = S.col(1);
        minMaxIdx(s_x, &min_x, &max_x);
        minMaxIdx(s_y, &min_y, &max_y);

        // TODO: set the rectangle as configurable parameter
        Subdiv2D subdiv(Rect(-500,-500,1000,1000));
        subdiv.insert(s);

        int a,b;
        subdiv.locate(s.back(),a,b);
        idx.resize(b+1);

        Point2f p;
        for(unsigned i=0;i<s.size();i++){
            subdiv.locate(s[i],a,b);
            idx[b] = i;
        }

        int v1,v2,v3;
        subdiv.getTriangleList(tp);

        for(unsigned i=0;i<tp.size();i++){
            Vec6f t = tp[i];

            //accept only vertex point
            if(t[0]>=min_x && t[0]<=max_x && t[1]>=min_y && t[1]<=max_y
                && t[2]>=min_x && t[2]<=max_x && t[3]>=min_y && t[3]<=max_y
                && t[4]>=min_x && t[4]<=max_x && t[5]>=min_y && t[5]<=max_y
            ){
                subdiv.locate(Point2f(t[0],t[1]),a,v1);
                subdiv.locate(Point2f(t[2],t[3]),a,v2);
                subdiv.locate(Point2f(t[4],t[5]),a,v3);
                triangles.push_back(Vec3i(idx[v1],idx[v2],idx[v3]));
            } //if
        } // for
    }

    Mat FacemarkAAMImpl::createMask(std::vector<Point2f> base_shape,  Rect res){
        Mat mask = Mat::zeros(res.height, res.width, CV_8U);
        std::vector<Point> hull;
        std::vector<Point> shape;
        Mat(base_shape).convertTo(shape, CV_32S);
        convexHull(shape,hull);
        fillConvexPoly(mask, &hull[0], (int)hull.size(), 255, 8 ,0);
        return mask.clone();
    }

    Mat FacemarkAAMImpl::createTextureBase(std::vector<Point2f> shape, std::vector<Vec3i> triangles, Rect res, std::vector<std::vector<Point> > & textureIdx){
        // max supported amount of triangles only 255
        Mat mask = Mat::zeros(res.height, res.width, CV_8U);

        std::vector<Point2f> p(3);
        textureIdx.clear();
        for(size_t i=0;i<triangles.size();i++){
            p[0] = shape[triangles[i][0]];
            p[1] = shape[triangles[i][1]];
            p[2] = shape[triangles[i][2]];


            std::vector<Point> polygon;
            approxPolyDP(p,polygon, 1.0, true);
            fillConvexPoly(mask, &polygon[0], (int)polygon.size(), (double)i+1,8,0 );

            std::vector<Point> list;
            for(int y=0;y<res.height;y++){
                for(int x=0;x<res.width;x++){
                    if(mask.at<uchar>(y,x)==(uchar)(i+1)){
                        list.push_back(Point(x,y));
                    }
                }
            }
            textureIdx.push_back(list);

        }

        return mask.clone();
    }

    Mat FacemarkAAMImpl::warpImage(Mat img, std::vector<Point2f> target_shape, std::vector<Point2f> curr_shape, std::vector<Vec3i> triangles, Rect res, std::vector<std::vector<Point> > textureIdx){
        // TODO: this part can be optimized, collect tranformation pair form all triangles first, then do one time remapping
        Mat warped = Mat::zeros(res.height, res.width, CV_8U);
        Mat warped2 = Mat::zeros(res.height, res.width, CV_8U);
        Mat image,part, warped_part;

        if(img.channels()>1){
            cvtColor(img,image,CV_BGR2GRAY);
        }else{
            image = img;
        }

        Mat A,R,t;
        A = Mat::zeros(2,3,CV_64F);
        std::vector<Point2f> target(3),source(3);
        std::vector<Point> polygon;
        for(size_t i=0;i<triangles.size();i++){
            target[0] = target_shape[triangles[i][0]];
            target[1] = target_shape[triangles[i][1]];
            target[2] = target_shape[triangles[i][2]];

            source[0] = curr_shape[triangles[i][0]];
            source[1] = curr_shape[triangles[i][1]];
            source[2] = curr_shape[triangles[i][2]];

            Mat target_mtx = Mat(target).reshape(1)-1.0;
            Mat source_mtx = Mat(source).reshape(1)-1.0;
            Mat U = target_mtx.col(0);
            Mat V = target_mtx.col(1);
            Mat X = source_mtx.col(0);
            Mat Y = source_mtx.col(1);

            double denominator = (target[1].x-target[0].x)*(target[2].y-target[0].y)-
                                (target[1].y-target[0].y)*(target[2].x-target[0].x);
            // denominator = 1.0/denominator;

            A.at<double>(0) = ((target[2].y-target[0].y)*(source[1].x-source[0].x)-
                             (target[1].y-target[0].y)*(source[2].x-source[0].x))/denominator;
            A.at<double>(1) = ((target[1].x-target[0].x)*(source[2].x-source[0].x)-
                             (target[2].x-target[0].x)*(source[1].x-source[0].x))/denominator;
            A.at<double>(2) =X.at<float>(0) + ((V.at<float>(0) * (U.at<float>(2) - U.at<float>(0)) - U.at<float>(0)*(V.at<float>(2) - V.at<float>(0))) * (X.at<float>(1) - X.at<float>(0)) + (U.at<float>(0) * (V.at<float>(1) - V.at<float>(0)) - V.at<float>(0)*(U.at<float>(1) - U.at<float>(0))) * (X.at<float>(2) - X.at<float>(0))) / denominator;
            A.at<double>(3) =((V.at<float>(2) - V.at<float>(0)) * (Y.at<float>(1) - Y.at<float>(0)) - (V.at<float>(1) - V.at<float>(0)) * (Y.at<float>(2) - Y.at<float>(0))) / denominator;
            A.at<double>(4) = ((U.at<float>(1) - U.at<float>(0)) * (Y.at<float>(2) - Y.at<float>(0)) - (U.at<float>(2) - U.at<float>(0)) * (Y.at<float>(1) - Y.at<float>(0))) / denominator;
            A.at<double>(5) = Y.at<float>(0) + ((V.at<float>(0) * (U.at<float>(2) - U.at<float>(0)) - U.at<float>(0) * (V.at<float>(2) - V.at<float>(0))) * (Y.at<float>(1) - Y.at<float>(0)) + (U.at<float>(0) * (V.at<float>(1) - V.at<float>(0)) - V.at<float>(0)*(U.at<float>(1) - U.at<float>(0))) * (Y.at<float>(2) - Y.at<float>(0))) / denominator;

            // A = getAffineTransform(target,source);

            R=A.colRange(0,2);
            t=A.colRange(2,3);

            Mat pts_ori = Mat(textureIdx[i]).reshape(1);
            Mat pts = pts_ori.t(); //matlab
            Mat bx = pts_ori.col(0);
            Mat by = pts_ori.col(1);

            Mat base_ind = (by-1)*res.width+bx;

            Mat pts_f;
            pts.convertTo(pts_f,CV_64FC1);
            pts_f.push_back(Mat::ones(1,(int)textureIdx[i].size(),CV_64FC1));

            Mat trans = (A*pts_f).t();

            Mat T; trans.convertTo(T, CV_32S); // this rounding make the result a little bit different to matlab
            Mat mx = T.col(0);
            Mat my = T.col(1);

            Mat ind = (my-1)*image.cols+mx;
            int maxIdx = image.rows*image.cols;
            int idx;

            for(int k=0;k<ind.rows;k++){
                idx=ind.at<int>(k);
                if(idx<maxIdx){
                    warped.at<uchar>(base_ind.at<int>(k)) = image.at<uchar>(idx);
                }

            }

            warped.copyTo(warped2);
        }

        return warped2.clone();
    }

    template <class T>
    Mat FacemarkAAMImpl::getFeature(const Mat m, std::vector<int> map){
        std::vector<float> feat;
        Mat M = m.t();//matlab
        // #pragma omp parallel for
        for(size_t i=0;i<map.size();i++){
            feat.push_back((float)M.at<T>(map[i]));
        }
        return Mat(feat).clone();
    }

    void FacemarkAAMImpl::createMaskMapping(const Mat m1, const Mat m2, std::vector<int> & ind1, std::vector<int> & ind2, std::vector<int> & ind3){

        int cnt = 0, idx=0;

        ind1.clear();
        ind2.clear();
        ind3.clear();

        Mat mask = m1.t();//matlab
        Mat mask2 = m2.t();//matlab

        for(int i=0;i<mask.rows;i++){
            for(int j=0;j<mask.cols;j++){
                if(mask.at<uchar>(i,j)>0){
                    if(mask2.at<uchar>(i,j)>0){
                        ind2.push_back(idx);
                        ind3.push_back(cnt);
                    }

                    ind1.push_back(idx);

                    cnt +=1;
                }
                idx+=1;
            } // j
        } // i

    }

    void FacemarkAAMImpl::image_jacobian(const Mat gx, const Mat gy, const Mat Jx, const Mat Jy, Mat & G){

        Mat Gx = repeat(gx,1,Jx.cols);
        Mat Gy = repeat(gy,1,Jx.cols);

        Mat G1,G2;
        multiply(Gx,Jx,G1);
        multiply(Gy,Jy,G2);

        G=G1+G2;
    }

    void FacemarkAAMImpl::warpUpdate(std::vector<Point2f> & shape, Mat delta, std::vector<Point2f> s0, Mat S, Mat Q, std::vector<Vec3i> triangles,std::vector<std::vector<int> > Tp){
        std::vector<Point2f> new_shape;
        int nSimEig = 4;

        /*get dr, dp and compute ds0*/
        Mat dr = -Mat(delta, Range(0,nSimEig));
        Mat dp = -Mat(delta, Range(nSimEig, delta.rows));


        Mat ds0 = S*dp + Q*dr;
        Mat ds0_mat = Mat::zeros((int)s0.size(),2, CV_32FC1);
        Mat c0 = ds0_mat.col(0);
        Mat c1 = ds0_mat.col(1);
        Mat(ds0, Range(0,(int)s0.size())).copyTo(c0);
        Mat(ds0, Range((int)s0.size(),(int)s0.size()*2)).copyTo(c1);

        Mat s_new = computeWarpParts(shape,s0,ds0_mat, triangles, Tp);

        Mat diff =linearize(Mat(s_new - Mat(s0).reshape(1)));

        Mat r = Q.t()*diff;
        Mat p = S.t()*diff;

        Mat s = linearize(s0)  +S*p + Q*r;
        Mat(Mat(s.t()).reshape(0,2).t()).reshape(2).copyTo(shape);
    }

    Mat FacemarkAAMImpl::computeWarpParts(std::vector<Point2f> curr_shape,std::vector<Point2f> s0, Mat ds0, std::vector<Vec3i> triangles,std::vector<std::vector<int> > Tp){

        std::vector<Point2f> new_shape;
        std::vector<Point2f> ds = ds0.reshape(2);

        float mx,my;
        Mat A;
        std::vector<Point2f> target(3),source(3);
        std::vector<double> p(3);
        p[2] = 1;
        for(size_t i=0;i<s0.size();i++){
            p[0] = s0[i].x + ds[i].x;
            p[1] = s0[i].y + ds[i].y;

            std::vector<Point2f> v;
            std::vector<float>vx, vy;
            for(size_t j=0;j<Tp[i].size();j++){
                int idx = Tp[i][j];
                target[0] = s0[triangles[idx][0]];
                target[1] = s0[triangles[idx][1]];
                target[2] = s0[triangles[idx][2]];

                source[0] = curr_shape[triangles[idx][0]];
                source[1] = curr_shape[triangles[idx][1]];
                source[2] = curr_shape[triangles[idx][2]];

                A = getAffineTransform(target,source);

                Mat(A*Mat(p)).reshape(2).copyTo(v);
                vx.push_back(v[0].x);
                vy.push_back(v[0].y);
            }// j

            /*find the median*/
            size_t n = vx.size()/2;
            nth_element(vx.begin(), vx.begin()+n, vx.end());
            mx = vx[n];
            nth_element(vy.begin(), vy.begin()+n, vy.end());
            my = vy[n];

            new_shape.push_back(Point2f(mx,my));
        } // s0.size()

        return Mat(new_shape).reshape(1).clone();
    }

    void FacemarkAAMImpl::gradient(const Mat M, Mat & gx, Mat & gy){
        gx = Mat::zeros(M.size(),CV_32FC1);
        gy = Mat::zeros(M.size(),CV_32FC1);

        /*gx*/
        for(int i=0;i<M.rows;i++){
            for(int j=0;j<M.cols;j++){
                if(j>0 && j<M.cols-1){
                    gx.at<float>(i,j) = ((float)0.5)*(M.at<float>(i,j+1)-M.at<float>(i,j-1));
                }else if (j==0){
                    gx.at<float>(i,j) = M.at<float>(i,j+1)-M.at<float>(i,j);
                }else{
                    gx.at<float>(i,j) = M.at<float>(i,j)-M.at<float>(i,j-1);
                }

            }
        }

        /*gy*/
        for(int i=0;i<M.rows;i++){
            for(int j=0;j<M.cols;j++){
                if(i>0 && i<M.rows-1){
                    gy.at<float>(i,j) = ((float)0.5)*(M.at<float>(i+1,j)-M.at<float>(i-1,j));
                }else if (i==0){
                    gy.at<float>(i,j) = M.at<float>(i+1,j)-M.at<float>(i,j);
                }else{
                    gy.at<float>(i,j) = M.at<float>(i,j)-M.at<float>(i-1,j);
                }

            }
        }

    }

    void FacemarkAAMImpl::createWarpJacobian(Mat S, Mat Q, std::vector<Vec3i> triangles, Model::Texture & T, Mat & Wx_dp, Mat & Wy_dp, std::vector<std::vector<int> > & Tp){

        std::vector<Point2f> base_shape = T.base_shape;
        Rect resolution = T.resolution;

        std::vector<std::vector<int> >triangles_on_a_point;

        int npts = (int)base_shape.size();

        Mat dW_dxdyt ;
        /*get triangles for each point*/
        std::vector<int> trianglesIdx;
        triangles_on_a_point.resize(npts);
        for(int i=0;i<(int)triangles.size();i++){
            triangles_on_a_point[triangles[i][0]].push_back(i);
            triangles_on_a_point[triangles[i][1]].push_back(i);
            triangles_on_a_point[triangles[i][2]].push_back(i);
        }
        Tp = triangles_on_a_point;

        /*calculate dW_dxdy*/
        float v0x,v0y,v1x,v1y,v2x,v2y, denominator;
        for(int k=0;k<npts;k++){
            Mat acc = Mat::zeros(resolution.height, resolution.width, CV_32F);

            /*for each triangle on k-th point*/
            for(size_t i=0;i<triangles_on_a_point[k].size();i++){
                int tId = triangles_on_a_point[k][i];

                Vec3i v;
                if(triangles[tId][0]==k ){
                    v=Vec3i(triangles[tId][0],triangles[tId][1],triangles[tId][2]);
                }else if(triangles[tId][1]==k){
                    v=Vec3i(triangles[tId][1],triangles[tId][0],triangles[tId][2]);
                }else{
                    v=Vec3i(triangles[tId][2],triangles[tId][0],triangles[tId][1]);
                }

                v0x = base_shape[v[0]].x;
                v0y = base_shape[v[0]].y;
                v1x = base_shape[v[1]].x;
                v1y = base_shape[v[1]].y;
                v2x = base_shape[v[2]].x;
                v2y = base_shape[v[2]].y;

                denominator = (v1x-v0x)*(v2y-v0y)-(v1y-v0y)*(v2x-v0x);

                Mat pixels = Mat(T.textureIdx[tId]).reshape(1); // same, just different order
                Mat p;

                pixels.convertTo(p,CV_32F, 1.0,1.0); //matlab use offset
                Mat x = p.col(0);
                Mat y = p.col(1);

                Mat alpha = (x-v0x)*(v2y-v0y)-(y-v0y)*(v2x-v0x);
                Mat beta = (v1x-v0x)*(y-v0y)-(v1y-v0y)*(x-v0x);

                Mat res = 1.0 - alpha/denominator - beta/denominator; // same just different order

                /*remap to image form*/
                Mat dx = Mat::zeros(resolution.height, resolution.width, CV_32F);
                for(int j=0;j<res.rows;j++){
                    dx.at<float>((int)(y.at<float>(j)-1.0), (int)(x.at<float>(j)-1.0)) = res.at<float>(j); // matlab use offset
                };

                acc = acc+dx;
            }

            Mat vectorized = Mat(acc.t()).reshape(0,1);
            dW_dxdyt.push_back(vectorized.clone());

        }// k

        Mat dx_dp;
        hconcat(Q, S, dx_dp);

        Mat dW_dxdy = dW_dxdyt.t();
        Wx_dp = dW_dxdy* Mat(dx_dp,Range(0,npts));
        Wy_dp = dW_dxdy* Mat(dx_dp,Range(npts,2*npts));

    } //createWarpJacobian

} /* namespace face */
} /* namespace cv */
