import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class PerspectiveChange:
  hToWarp = None
  hToUnwarp = None
  width = 0
  height = 0

  def crop(image_path, coords):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    """
    cropped_image = image_obj.crop(coords)
    return cropped_image

  def perspectiveChange(self, im_src):
    self.height, self.width = im_src.shape[:2]
    print(im_src.shape[:2])

    # Four corners of the book in source image
    topWidth = 0.034
    bottomWidth = 0.2
    offset = self.width*0.007
    array_src = [[self.width*(0.5-topWidth/2) + offset, self.height*0.60], # tl
             [self.width*(0.5+topWidth/2) + offset, self.height*0.60],     # tr
             [self.width*(1-bottomWidth/2) + offset, self.height*1.0],      # br
             [self.width*(bottomWidth/2) + offset, self.height*1.0]]      # bl
    pts_src =   np.array(array_src)
    pts_debug = np.array(array_src, np.int32)
    dst_pts = np.array([[self.width*0.05, 0],
                        [self.width*0.95, 0],
                        [self.width*0.95, 720],
                        [self.width*0.05, 720]])

    # Calculate Homography
    self.hToWarp, status = cv2.findHomography(pts_src, dst_pts)
    self.hToUnwarp, status = cv2.findHomography(dst_pts, pts_src)

    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, self.hToWarp, (im_src.shape[1], int(im_src.shape[0]*1.15)))
    # cv2.polylines(im_src, [pts_debug], True, (255, 255, 255), 2)

    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(im_src)
    # ax1.set_title('Original Image', fontsize=40)
    # ax2.imshow(im_out)
    # ax2.set_title('after', fontsize=40)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # plt.show()

    return im_out

  def unWarpImage(self, warpedImage):
      im_out = cv2.warpPerspective(warpedImage, self.hToUnwarp, (self.width, self.height))
      # plt.plot(im_out)
      # plt.show()
      return im_out

  def integretyScore(self, binaryImg):
    print("todo")



