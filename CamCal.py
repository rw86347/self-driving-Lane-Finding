import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class CamCal:
  # termination criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
  objp = np.zeros((9 * 6, 3), np.float32)
  objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

  # Arrays to store object points and image points from all the images.
  objpoints = []  # 3d point in real world space
  imgpoints = []  # 2d points in image plane.

  mtx = None
  dist = None
  rvecs = None
  tvecs = None
  calibrationDir = None
  roi = None

  def __init__(self, calibrationDirectory):
    self.calibrationDir = calibrationDirectory
    print(calibrationDirectory)
    h = 0
    w = 0
    # good tutorial
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    imageList = os.listdir(calibrationDirectory)
    for fname in imageList:
      img = cv2.imread(calibrationDirectory + "/" + fname)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      # Find the chess board corners
      ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

      # If found, add object points, image points (after refining them)
      if ret == True:
        self.objpoints.append(self.objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        self.imgpoints.append(corners2)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)
        h, w = img.shape[:2]

    # cv2.destroyAllWindows()
    ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
    self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
    print("calibrated")
    # self.showCorrectedImages()

  def showCorrectedImages(self):
    imageList = os.listdir(self.calibrationDir)
    for fname in imageList:
      img = cv2.imread(self.calibrationDir + "/" + fname)
      dst = self.correctImage(img)
      cv2.imshow('img', dst)
      cv2.waitKey(0)

  def correctImage(self, img):
    corrected = cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
    # crop the image
    x, y, w, h = self.roi
    dst = corrected[y:y + h, x:x + w]
    print("width = " + str(w))
    print("height = " + str(h))
    return dst