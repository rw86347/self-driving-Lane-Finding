
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

Please see the images in the "output_images" directory

[image1]: ./output_images/Screenshot01 "image set number 1"
[image2]: ./output_images/Screenshot02 "image set number 2"
[image3]: ./output_images/Screenshot03 "image set number 3"
[video1]: ./output_images/output.avi "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration for my code is all contained in the class `CamCal.py`. 

The initialization routine for my camera calibration code take the directory for the calibation images.  I then save off the needed 
information for correcting images as member variables.  This allows me to quickly call the correctImage routine for each frame
in a video to fix the video up

It wasn't entirely obvious from the class tutorials how to correct the images from multiple chessboard pictures.  However, after a little 
web surfing I found it was as simple ast concatinating the data which I then gave to calibrateCamera and then to getOptimalNewCameraMatrix.

### Pipeline (single images)

The object BinaryImage.py has a multitude of different methods that I used to test different theories reguarding
pulling out the best data.  I think that this area of code is the weakest think in this type of lane finding.  I
would replace this whole section with a Convolutional Neural Net.  As I think there is more logic that we use to 
find lines in the road 

#### 1. Provide an example of a distortion-corrected image.

As you can see from all of the screenshot images I have corrected images. 
I like to call them drone images because it is like I am flying a drone ahead of the car
looking down on the lane markings

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Read the readme.txt in the images directory.  After doing image warping and pipeline/binaryImage 
manipulation I start doing statistics on the sliding window of data to look for additional bad data
If I had time I think this would be a good time to use pearson correlation to try and give a score 
to the data with in a window.  For example if you are looking at grass you will see a fairly even
distrobution of dots where a line marking will be tightly grouped.  In the end I opted for something
less time consuming.  I give each windown and lane a score.  If you don't score high enough the data
get tossed and replacement data is inserted based on the other lanes or based on other data in this lane.

Look at the code in main.py line 80 to see tossing of bad lane data. 

And if you look at main.py line 199-256 you will see where I am changing data when I think we have 
boggus histogram data.  This faked up data actually seems to work prety well.  However, the old addage
still holds true. Garbage in, Garbage out.  We really need to toss the pipeline section with a CNN.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

All of the perspective change code is in the PerspectiveChange.py.  Lines 27-37 is where I come up with the locations for the transform.  
Honestly I would never have been successful with this code had I not inserted line 45.  Because if you trapizoid is offset
left or right then the final image will be terrible.  I also learned on thing that was never taught in the class.  Line 45 is critcle
to the success of the project...

int(im_src.shape[0]*1.15)))

If you don't lengthen the height of the new image you will cut off some of the bottom of the lane and this will throw off your calculations.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In main.py line 71 you will see the function fit_polynomial.py.  This is largly based on the code 
give in the lectures.  However, you will notice that I have made changes.  One change that I would
like to make but didn't have time for was looking for wildly different curvatures.  If for example
the left lane is turning right and the right lane is turning left something definityly wrong.  At
this point I think we should look at confidance scores from each of the lanes and history.  This should
help us find which data is mostly likely wrong. 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In main.py lines 73-80 I calculated the radius of the curve.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

My screenshots contain each of the 6 steps in a single image.  You will notice
the last image of each screenshot has the plot laid out on the road.

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

I have an output.mov file in output_images that shows the lanes being drawn in a movie.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think the pipline gradiants is the weak link, and I think using hand written algorithms
is a bad idea to begin with.   We need to use a CNN because they are more agile.