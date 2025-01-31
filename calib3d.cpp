#include "calib3d.h"


void Fisheye_UndistortImage(Mat distorted, Mat undistorted, Mat k, Mat d) {
    cv::fisheye::undistortImage(*distorted, *undistorted, *k, *d);
}

void Fisheye_UndistortImageWithParams(Mat distorted, Mat undistorted, Mat k, Mat d, Mat knew, Size size) {
    cv::Size sz(size.width, size.height);
    cv::fisheye::undistortImage(*distorted, *undistorted, *k, *d, *knew, sz);
}

void InitUndistortRectifyMap(Mat cameraMatrix,Mat distCoeffs,Mat r,Mat newCameraMatrix,Size size,int m1type,Mat map1,Mat map2) {
    cv::Size sz(size.width, size.height);
    cv::initUndistortRectifyMap(*cameraMatrix,*distCoeffs,*r,*newCameraMatrix,sz,m1type,*map1,*map2);
}

Mat GetOptimalNewCameraMatrixWithParams(Mat cameraMatrix,Mat distCoeffs,Size size,double alpha,Size newImgSize,Rect* validPixROI,bool centerPrincipalPoint) {
    cv::Size sz(size.width, size.height);
    cv::Size newSize(newImgSize.width, newImgSize.height);
    cv::Rect rect(validPixROI->x,validPixROI->y,validPixROI->width,validPixROI->height);
    cv::Mat* mat = new cv::Mat(cv::getOptimalNewCameraMatrix(*cameraMatrix,*distCoeffs,sz,alpha,newSize,&rect,centerPrincipalPoint));
    validPixROI->x = rect.x;
    validPixROI->y = rect.y;
    validPixROI->width = rect.width;
    validPixROI->height = rect.height;
    return mat;
}

std::vector<cv::Mat> Mats2Vector(Mats* mats) {
    std::vector<cv::Mat> vec;
    for (int i = 0; i < mats->length; ++i) {
        vec.push_back(*mats->mats[i]);
    }
    return vec;
}

void Vector2Mats(std::vector<cv::Mat>& vec, Mats* mat) {
    int n = vec.size();
    if (mat->length < n) {
        Mat* old = mat->mats;
        mat->mats = (Mat*)realloc(old, n * sizeof(Mat));
        free(old);
    }
    for (int i = 0; i < n; i++) {
        mat->mats[i] = &vec[i];
    }
}

double CalibrateCameraSimple(Points3fArr objectPoints, Points2fArr imagePoints,
                             Size imageSize, Mat cameraMatrix, Mat distCoeffs,
                             Mats* rvecs, Mats* tvecs) {
    std::vector<std::vector<cv::Point3f>> oP;
    for (int i = 0; i < objectPoints.length; i++) {
        std::vector<cv::Point3f> line;
        for (int j = 0; j < objectPoints.data[i].length; j++) {
            auto p = objectPoints.data[i].points[j];
            line.push_back(cv::Point3f(p.x, p.y, p.z));
        }
        oP.push_back(line);
    }
    std::vector<std::vector<cv::Point2f>> iP;
    for (int i = 0; i < imagePoints.length; i++) {
        std::vector<cv::Point2f> line;
        for (int j = 0; j < imagePoints.data[i].length; j++) {
            auto p = imagePoints.data[i].points[j];
            line.push_back(cv::Point2f(p.x, p.y));
        }
        iP.push_back(line);
    }
    cv::Size sz(imageSize.width, imageSize.height);
    std::vector<cv::Mat> rs, ts;
    int flags = (cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_K1 | cv::CALIB_FIX_K2 |
                 cv::CALIB_FIX_K3 | cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5 | cv::CALIB_FIX_K6);
    cv::TermCriteria tc(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                        DBL_EPSILON);
    double err = cv::calibrateCamera(oP, iP, sz, rs, ts, *cameraMatrix, *distCoeffs, flags);
    Vector2Mats(rs, rvecs);
    Vector2Mats(ts, tvecs);
    return err;
}
