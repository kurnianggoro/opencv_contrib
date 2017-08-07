/*----------------------------------------------
 * Usage:
 * facemark_lbf_fitting <video_name>
 *
 * example:
 * facemark_lbf_fitting myvideo.mp4
 *
 * note: do not forget to provide the LBF_MODEL and DETECTOR_MODEL
 * the model are available at opencv_contrib/modules/face/data/
 *--------------------------------------------------*/
 #define LBF_MODEL "../data/LBF.model"
 #define DETECTOR_MODEL "../data/haarcascade_frontalface_alt.xml"

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <ctime>

using namespace std;
using namespace cv;

bool myDetector( const Mat image, std::vector<Rect> & faces );

int main(int argc, char** argv ){
    FacemarkLBF::Params params;
    params.saved_file_name = LBF_MODEL;
    params.cascade_face = DETECTOR_MODEL;

    Ptr<Facemark> facemark = FacemarkLBF::create(params);
    facemark->setFaceDetector(myDetector);
    facemark->loadModel(params.saved_file_name.c_str());

    string filename = argv[1];
    VideoCapture capture(filename);
    Mat frame;

    if( !capture.isOpened() )
        throw "Error when reading video";

    Mat img;
    String text;
    char buff[255];
    double fittime;
    int nfaces;
    std::vector<Rect> rects,rects_scaled;
    std::vector<std::vector<Point2f> > landmarks;
    CascadeClassifier cc(params.cascade_face.c_str());
    namedWindow( "w", 1);
    for( ; ; )
    {
        capture >> frame;
        if(frame.empty())
            break;

        double __time__ = getTickCount();

        float scale = (400.0/frame.cols);
        resize(frame, img, Size(frame.cols*scale, frame.rows*scale));

        facemark->getFaces(img, rects);
        rects_scaled.clear();

        for(int j=0;j<rects.size();j++){
            rects_scaled.push_back(Rect(rects[j].x/scale,rects[j].y/scale,rects[j].width/scale,rects[j].height/scale));
        }
        rects = rects_scaled;
        fittime=0;
        nfaces = rects.size();
        if(rects.size()>0){
            double newtime = getTickCount();

            facemark->fit(frame, rects, landmarks);


            fittime = ((getTickCount() - newtime)/getTickFrequency());
            for(int j=0;j<rects.size();j++){
                landmarks[j] = Mat(Mat(landmarks[j]));
                facemark->drawPoints(frame, landmarks[j], Scalar(0,0,255));
            }
        }


        double fps = (getTickFrequency()/(getTickCount() - __time__));
        sprintf(buff, "faces: %i %03.2f fps, fit:%03.0f ms",nfaces,fps,fittime*1000);
        text = buff;
        putText(frame, text, Point(20,40), FONT_HERSHEY_PLAIN , 2.0,Scalar::all(255), 2, 8);

        imshow("w", frame);
        waitKey(1); // waits to display frame
    }
    waitKey(0); // key press to close window
}

CascadeClassifier face_cascade(DETECTOR_MODEL);
bool myDetector( const Mat image, std::vector<Rect> & faces ){
    Mat gray;
    faces.clear();

    if(image.channels()>1){
        cvtColor(image,gray,CV_BGR2GRAY);
    }else{
        gray = image.clone();
    }
    equalizeHist( gray, gray );

    face_cascade.detectMultiScale( gray, faces, 1.4, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  return true;
}
