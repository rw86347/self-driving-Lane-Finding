import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import numpy as np


def warp(frame):
  h, w = frame.shape[:2]
  h2 = h / 2 * 1.2
  w2 = w / 2
  lb = w2 * 0.1
  rb = w - lb
  lt = w2 * 0.8
  rt = w - lt
  bottomY = h * 0.9
  newWidth = rb - lb
  newHeight = h

  im_src = cv2.imread('test_images/test1.jpg')
  # Four corners of the book in source image
  pts_src = np.array([[lb, bottomY], [rb, bottomY], [rt, h2], [lt, h2]])
  pts =     np.array([[lb, bottomY], [rb, bottomY], [rt, h2], [lt, h2]], np.int32)
  cv2.polylines(frame, [pts], True, (0, 255, 255), 3)
  print("src")
  print(pts_src)

  pts_dst = np.array([[0, newHeight], [newWidth, newHeight], [newWidth, 0], [0, 0]])
  print("dst")
  print(pts_dst)

  # Calculate Homography
  h, status = cv2.findHomography(pts_src, pts_dst)

  # Warp source image to destination based on homography
  im_out = cv2.warpPerspective(im_src, h, (im_src.shape[1], im_src.shape[0]))

  # Display images
  # cv2.imshow("Source Image", frame)
  # cv2.imshow("Warped Source Image", im_out)
  # cv2.waitKey(0)
  return im_src

# img = cv2.imread('test_images/test1.jpg',0)
img = mpimg.imread('test_images/test1.jpg')
img = warp(img)
plt.imshow(img)
plt.show()

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(img,(5,5),0)
# ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plt.imshow(th3)
# plt.show()

