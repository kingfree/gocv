package gocv

/*
#include <stdlib.h>
#include "calib3d.h"
*/
import "C"
import "image"

// Calib is a wrapper around OpenCV's "Camera Calibration and 3D Reconstruction" of
// Fisheye Camera model
//
// For more details, please see:
// https://docs.opencv.org/trunk/db/d58/group__calib3d__fisheye.html

// CalibFlag value for calibration
type CalibFlag int32

const (
	// CalibUseIntrinsicGuess indicates that cameraMatrix contains valid initial values
	// of fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially
	// set to the image center ( imageSize is used), and focal distances are computed
	// in a least-squares fashion.
	CalibUseIntrinsicGuess CalibFlag = 1 << iota

	// CalibRecomputeExtrinsic indicates that extrinsic will be recomputed after each
	// iteration of intrinsic optimization.
	CalibRecomputeExtrinsic

	// CalibCheckCond indicates that the functions will check validity of condition number
	CalibCheckCond

	// CalibFixSkew indicates that skew coefficient (alpha) is set to zero and stay zero
	CalibFixSkew

	// CalibFixK1 indicates that selected distortion coefficients are set to zeros and stay zero
	CalibFixK1

	// CalibFixK2 indicates that selected distortion coefficients are set to zeros and stay zero
	CalibFixK2

	// CalibFixK3 indicates that selected distortion coefficients are set to zeros and stay zero
	CalibFixK3

	// CalibFixK4 indicates that selected distortion coefficients are set to zeros and stay zero
	CalibFixK4

	// CalibFixIntrinsic indicates that fix K1, K2? and D1, D2? so that only R, T matrices are estimated
	CalibFixIntrinsic

	// CalibFixPrincipalPoint indicates that the principal point is not changed during the global optimization.
	// It stays at the center or at a different location specified when CalibUseIntrinsicGuess is set too.
	CalibFixPrincipalPoint
)

// FisheyeUndistortImage transforms an image to compensate for fisheye lens distortion
func FisheyeUndistortImage(distorted Mat, undistorted *Mat, k, d Mat) {
	C.Fisheye_UndistortImage(distorted.Ptr(), undistorted.Ptr(), k.Ptr(), d.Ptr())
}

// FisheyeUndistortImageWithParams transforms an image to compensate for fisheye lens distortion with Knew matrix
func FisheyeUndistortImageWithParams(distorted Mat, undistorted *Mat, k, d, knew Mat, size image.Point) {
	sz := C.struct_Size{
		width:  C.int(size.X),
		height: C.int(size.Y),
	}
	C.Fisheye_UndistortImageWithParams(distorted.Ptr(), undistorted.Ptr(), k.Ptr(), d.Ptr(), knew.Ptr(), sz)
}

// InitUndistortRectifyMap computes the joint undistortion and rectification transformation and represents the result in the form of maps for remap
//
// For further details, please see:
// https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
//
func InitUndistortRectifyMap(cameraMatrix Mat, distCoeffs Mat, r Mat, newCameraMatrix Mat, size image.Point, m1type int, map1 Mat, map2 Mat) {
	sz := C.struct_Size{
		width:  C.int(size.X),
		height: C.int(size.Y),
	}
	C.InitUndistortRectifyMap(cameraMatrix.Ptr(), distCoeffs.Ptr(), r.Ptr(), newCameraMatrix.Ptr(), sz, C.int(m1type), map1.Ptr(), map2.Ptr())
}

// GetOptimalNewCameraMatrixWithParams computes and returns the optimal new camera matrix based on the free scaling parameter.
//
// For further details, please see:
// https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga7a6c4e032c97f03ba747966e6ad862b1
//
func GetOptimalNewCameraMatrixWithParams(cameraMatrix Mat, distCoeffs Mat, imageSize image.Point, alpha float64, newImgSize image.Point, centerPrincipalPoint bool) (Mat, image.Rectangle) {
	sz := C.struct_Size{
		width:  C.int(imageSize.X),
		height: C.int(imageSize.Y),
	}
	newSize := C.struct_Size{
		width:  C.int(newImgSize.X),
		height: C.int(newImgSize.Y),
	}
	rt := C.struct_Rect{}
	return newMat(C.GetOptimalNewCameraMatrixWithParams(cameraMatrix.Ptr(), distCoeffs.Ptr(), sz, C.double(alpha), newSize, &rt, C.bool(centerPrincipalPoint))), toRect(rt)
}

type Point2f struct {
	X, Y float64
}

type Point3f struct {
	X, Y, Z float64
}

// CalibrateCameraSimple finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.
//
// @return error, cameraMatrix, distCoeffs, rvecs, tvecs
//
// For further details, please see:
// https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga687a1ab946686f0d85ae0363b5af1d7b
//
func CalibrateCameraSimple(objects [][]Point3f, images [][]Point2f, size image.Point) (float64, Mat, Mat, []Mat, []Mat) {
	n := len(objects)
	m := len(objects[0])
	oPoints := C.Points3fArray_New(C.int(n), C.int(m))
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			point := C.Point3f{C.float(objects[i][j].X), C.float(objects[i][j].Y), C.float(objects[i][j].Z)}
			C.Points3fArr_Set(oPoints, C.int(i), C.int(j), point)
		}
	}
	n = len(images)
	m = len(images[0])
	iPoints := C.Points2fArray_New(C.int(n), C.int(m))
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			point := C.Point2f{C.float(images[i][j].X), C.float(images[i][j].Y)}
			C.Points2fArr_Set(iPoints, C.int(i), C.int(j), point)
		}
	}
	sz := C.Size{C.int(size.X), C.int(size.Y)}
	zero := C.Scalar{C.double(0), C.double(0), C.double(0), C.double(0)}
	cameraMatrix := C.Mat_NewWithSizeFromScalar(zero, 3, 3, MatTypeCV32F)
	distCoeffs := C.Mat_NewWithSizeFromScalar(zero, 1, 5, MatTypeCV32F)
	var rvecs, tvecs C.Mats
	err := float64(C.CalibrateCameraSimple(*oPoints, *iPoints, sz, cameraMatrix, distCoeffs, &rvecs, &tvecs))
	var Rs, ts []Mat
	n = int(rvecs.length)
	for i := 0; i < n; i++ {
		Rs = append(Rs, newMat(C.Mats_get(rvecs, C.int(i))))
		ts = append(ts, newMat(C.Mats_get(tvecs, C.int(i))))
	}
	return err, newMat(cameraMatrix), newMat(distCoeffs), Rs, ts
}
