In the screenshot image in this directory you will find 6 labeled images. I used this
window in my development.

1) Original Image

This isn't exactly the origional image this is the image after I corrected for the
lens distortion.

2) Warped Result

I found I got best results when my lines in this straight section of road pointed
directly up.  It help me to verify that I was warping the image correctly.  I also
wanted to make sure I got a little of the hood of the car in this image.  It was
my way of making sure I was capturing the whole trapizoid.

3) Pipline Change.

I messed around with lots of different pipelines vaiations.  They all had faults.
If you look in BinaryImage.py you can see all of the methods and different
combonations that I tried and played around with

4) Polinomial Fit

I made changes to and modification beyond what was taught in the class.  You will
notice that some boxes are red.  These boxes are derived from data associated with
the other opposing line and the current line.  You will also notice that I have a
number printed above each box.  I print and track a score for each window.  If a
window has too low of a score I will change the data much like I did for the red
box.

If a entire line gets too low of a score I will toss the data and start to make
assuptions based on the clear lane marking on the other side.

5) Line Overlay

This view does more than just print a polynomial, I have a smoothing function and
toss data that looks too random.  Crazy changes can be dangerous.  I would rather
to continue the action from the last frame if I have bad data.

6) final is just that the final image.