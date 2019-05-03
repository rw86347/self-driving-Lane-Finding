import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class BinaryImage:

  def hls_select(self, img, thresh=(90, 255)):
      hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
      s_channel = hls[:,:,2]
      binary_output = np.zeros_like(s_channel)
      binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
      return binary_output


  def pipeline(self, img, s_thresh=(170, 205), sx_thresh=(20, 50)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    gray = cv2.cvtColor(color_binary, cv2.COLOR_BGR2GRAY)
    binary_output = np.zeros_like(gray)
    binary_output[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1
    # cv2.imshow("debug", gray)
    # cv2.waitKey(0)
    # plt.plot(binary_output)
    # plt.show()
    return binary_output

  def abs_sobel_thresh(self, img, orient='x', thresh_min=20, thresh_max=100):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
      abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
      abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

  def adaptiveThresholdingOfSaturation(self, img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    img = np.dstack((np.zeros_like(s_channel), s_channel, np.zeros_like(s_channel))) * 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

  def findYellowOnly(self, image):
    # Grab the x and y size and make a copy of the image
    ysize = image.shape[0]
    xsize = image.shape[1]
    color_select = np.copy(image)

    # Define color selection criteria
    ###### MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
    red_threshold = 255 - 0xFF
    green_threshold = 255 - 0xFF
    blue_threshold = 255 - 0xAB
    ######

    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    # Do a boolean or with the "|" character to identify
    # pixels below the thresholds
    thresholds = (image[:, :, 0] < rgb_threshold[0]) \
                 | (image[:, :, 1] < rgb_threshold[1]) \
                 | (image[:, :, 2] > rgb_threshold[2])
    color_select[thresholds] = [0, 0, 0]

    # Display the image
    plt.show(color_select)
    return color_select