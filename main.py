# imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math
from CamCal import CamCal
from BinaryImage import BinaryImage
from PerspectiveChange import PerspectiveChange

class Lanes:
  # Calibrate Camera
  calEngine = CamCal('camera_cal')

  # load images or video
  useTestImages = False
  # videoName = "challenge_video.mp4"
  videoName = "project_video.mp4"
  staticImages = []
  nextStaticImage = 0
  cap = None
  out = None
  frame = None
  leftAveragePolyA = []
  leftAveragePolyB = []
  leftAveragePolyC = []
  rightAveragePolyA = []
  rightAveragePolyB = []
  rightAveragePolyC = []
  maxFramesPloyAverage = 10

  def getNextTestImage(self):
    if (self.nextStaticImage + 1) > len(self.staticImages):
      return None
    rv = self.staticImages[self.nextStaticImage]
    self.nextStaticImage = self.nextStaticImage + 1
    return rv

  def getNextVideoImage(self):
    if self.cap == None:
      self.cap = cv2.VideoCapture(self.videoName)
      fourcc = cv2.VideoWriter_fourcc(*'XVID')
      self.out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1212,629))
    ret, frame = self.cap.read()
    if ret == True:
      rv = frame
    else:
      return None
    return rv

  def getFrame(self):
    f = None
    if self.useTestImages == True:
      f = self.getNextTestImage()
    else:
      f = self.getNextVideoImage()
    return f

  def hist(self, img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0] // 2:, :]

    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)

    return histogram

  def fit_polynomial(self, binary_warped, warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img, laneWidth, ret = self.find_lane_pixels(binary_warped)

    left_y_eval = np.max(lefty)
    right_y_eval = np.max(righty)

    left_curverad = ((1 + (2 * leftx[0] * left_y_eval + leftx[1]) ** 2) ** 1.5) / np.absolute(2 * leftx[0])
    right_curverad = ((1 + (2 * rightx[0] * right_y_eval + rightx[1]) ** 2) ** 1.5) / np.absolute(2 * rightx[0])
    print("left curv:"+str(left_curverad)+"px right curv:"+str(right_curverad)+"px")


    # create return image with alpha channel
    # b_channel, g_channel, r_channel = cv2.split(warped)
    # alpha_channel = np.zeros(b_channel.shape, dtype=b_channel.dtype) * 50  # creating a dummy alpha channel image.
    # img_BGRA = cv2.merge((b_channel, g_channel, r_channel))
    returnImage= np.zeros_like(warped)

    left_fit = [0.0, 0.0, 0.0]
    right_fit = [0.0, 0.0, 0.0]
    if ret == True:
      # Fit a second order polynomial to each using `np.polyfit`
      left_fit = np.polyfit(lefty, leftx, 2)
      self.leftAveragePolyA.append(left_fit[0])
      self.leftAveragePolyB.append(left_fit[1])
      self.leftAveragePolyC.append(left_fit[2])
      right_fit = np.polyfit(righty, rightx, 2)
      self.rightAveragePolyA.append(right_fit[0])
      self.rightAveragePolyB.append(right_fit[1])
      self.rightAveragePolyC.append(right_fit[2])

      if len(self.rightAveragePolyA) > self.maxFramesPloyAverage:
        self.leftAveragePolyA = self.leftAveragePolyA[1:]
        self.leftAveragePolyB = self.leftAveragePolyB[1:]
        self.leftAveragePolyC = self.leftAveragePolyC[1:]
        self.rightAveragePolyA = self.rightAveragePolyA[1:]
        self.rightAveragePolyB = self.rightAveragePolyB[1:]
        self.rightAveragePolyC = self.rightAveragePolyC[1:]

    # now average the poly and put it back into the orig
    left_fit[0] = sum(self.leftAveragePolyA)/len(self.leftAveragePolyA)
    left_fit[1] = sum(self.leftAveragePolyB)/len(self.leftAveragePolyB)
    left_fit[2] = sum(self.leftAveragePolyC)/len(self.leftAveragePolyC)
    right_fit[0] = sum(self.rightAveragePolyA)/len(self.rightAveragePolyA)
    right_fit[1] = sum(self.rightAveragePolyB)/len(self.rightAveragePolyB)
    right_fit[2] = sum(self.rightAveragePolyC)/len(self.rightAveragePolyC)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # print("ploty")
    # print(ploty)
    try:
      left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
      right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
      # Avoids an error if `left` and `right_fit` are still none or incorrect
      print('The function failed to fit a line!')
      left_fitx = 1 * ploty ** 2 + 1 * ploty
      right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='black')
    # plt.plot(right_fitx, ploty, color='blue')

    for y in ploty:
      # print("y="+str(y)+" x="+str(left_fitx[int(y)]))
      radius = 10
      cv2.circle(returnImage, (int(left_fitx[int(y)]), int(y)), radius, (0, 255, 0, 255))
      cv2.circle(returnImage, (int(right_fitx[int(y)]), int(y)), radius, (0, 255, 0, 255))

    return out_img, returnImage, laneWidth

  def find_lane_pixels(self, binary_warped):
    laneWidthSum = 0
    laneWidthCount = 0
    leftScore = 0
    rightScore = 0
    minRequiredLaneScore = 5000
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 700

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
      # Identify window boundaries in x and y (and right and left)
      win_y_low = binary_warped.shape[0] - (window + 1) * window_height
      win_y_high = binary_warped.shape[0] - window * window_height
      win_xleft_low = leftx_current - margin
      win_xleft_high = leftx_current + margin
      win_xright_low = rightx_current - margin
      win_xright_high = rightx_current + margin

      # Draw the windows on the visualization image


      # Identify the nonzero pixels in x and y within the window #
      good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
      good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

      # Append these indices to the lists
      left_lane_inds.append(good_left_inds)
      right_lane_inds.append(good_right_inds)

      # If you found > minpix pixels, recenter next window on their mean position
      if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.putText(out_img, str(len(good_left_inds)), (win_xleft_low, win_y_low), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        leftScore += len(good_left_inds)
      if len(good_right_inds) > minpix:
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        cv2.putText(out_img, str(len(good_right_inds)), (win_xright_low, win_y_low), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        rightScore += len(good_right_inds)

      if len(good_left_inds) > minpix and len(good_right_inds) > minpix:
        dist = win_xright_low - win_xleft_low
        laneWidthSum += dist
        laneWidthCount += 1
        distCM = int(dist * (331/916))
        print ("distance = " + str(dist) + " cm = " + str(distCM))

      averageLaneWidth = 0
      if laneWidthCount < 0.001:
        averageLaneWidth = 99999
      else:
        averageLaneWidth = int(laneWidthSum/laneWidthCount)

      if len(good_left_inds) < minpix and len(good_right_inds) > minpix and laneWidthCount > 0:
        cv2.rectangle(out_img, (win_xright_low-averageLaneWidth, win_y_low), (win_xright_high-averageLaneWidth, win_y_high), (255, 0, 0), 2)

      if len(good_left_inds) > minpix and len(good_right_inds) < minpix and laneWidthCount > 0:
        cv2.rectangle(out_img, (win_xleft_low+averageLaneWidth, win_y_low), (win_xleft_high+averageLaneWidth, win_y_high), (255, 0, 0), 2)

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
      left_lane_inds = np.concatenate(left_lane_inds)
      right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
      # Avoids an error if the above is not implemented fully
      pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    print("left score: "+str(leftScore)+" right score: "+str(rightScore))
    if leftScore < minRequiredLaneScore and rightScore > minRequiredLaneScore:
      print("faking left lane")
      for i in leftx:
        i = rightx - averageLaneWidth

    if rightScore < minRequiredLaneScore and leftScore > minRequiredLaneScore:
      print("faking right lane")
      for i in rightx:
        i = leftx + averageLaneWidth

    if rightScore < minRequiredLaneScore and leftScore < minRequiredLaneScore:
      return leftx, lefty, rightx, righty, out_img, int(averageLaneWidth), False

    return leftx, lefty, rightx, righty, out_img, int(averageLaneWidth), True

  def run(self):
    # set testFames to true or false
    if self.useTestImages == True:
      imageList = os.listdir("test_images/")
      for file in imageList:
        image = mpimg.imread('test_images/'+file)
        self.staticImages.append(image)
      else:
        cap = cv2.VideoCapture(self.videoName)

    # loop through frames
    b = BinaryImage()
    p = PerspectiveChange()
    frame = self.getFrame()
    while type(frame) is np.ndarray:
      # print("inside of while loop")
      # apply distortion correction
      frame = self.calEngine.correctImage(frame)
      # crop using image warping
      warped = p.perspectiveChange(frame)

      # Create binary image
      # binary_warped = b.pipeline(warped, s_thresh=(150, 205), sx_thresh=(20, 40)) # orig pipeline(img, s_thresh=(170, 205), sx_thresh=(20, 50))
      binary_warped = b.abs_sobel_thresh(warped)
      # binary_warped = b.findYellowOnly(warped)
      # binary_warped = b.adaptiveThresholdingOfSaturation(warped)


      # determine the curvature and possition
      hist = self.hist(binary_warped)
      # plt.plot(hist)
      out_img, overlay, laneWidth = self.fit_polynomial(binary_warped, warped)

      # re-warp the image back
      overlay = p.unWarpImage(overlay)
      finalImage = cv2.addWeighted(frame, 0.8, overlay, 1.0, 0.0)
      font = cv2.FONT_HERSHEY_SIMPLEX
      # cv2.putText(finalImage, 'LaneWidth: '+str(laneWidth)+' cm', (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

      # write possition data back on to screen

      # Plot the result

      if self.useTestImages == False:
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('color', finalImage)
        cv2.imshow('birdseye', warped)
        cv2.waitKey(1)
        self.out.write(finalImage)
      else:
        f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(frame)
        ax1.set_title('Original Image', fontsize=40)

        ax2.imshow(warped)
        ax2.set_title('Warped Result', fontsize=40)

        ax3.imshow(binary_warped)
        ax3.set_title('Pipeline Change', fontsize=40)

        ax4.imshow(out_img)
        ax4.set_title('Polinomial Fit', fontsize=40)

        ax5.imshow(overlay)
        ax5.set_title('Line Overlay', fontsize=40)

        ax6.imshow(finalImage)
        ax6.set_title('Final', fontsize=40)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

      frame = self.getFrame()
    if self.useTestImages == False:
      self.cap.release()
      self.out.release()
    cv2.destroyAllWindows()

lineFinder = Lanes()
lineFinder.run()