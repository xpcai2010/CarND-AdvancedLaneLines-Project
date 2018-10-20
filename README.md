## Writeup - Advanced Lane Finding Project

### Project Introduction
Advanced Lane Finding Project - the goal is to write a Python software pipeline to identify the drive lane boundaries in a video from a front-facing camera mounted on a car.

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[image7]: ./output_images/Camera_Calibration_Chessboard.jpg "Camera Calibration"
[image8]: ./output_images/Distortion_Corrected_Image.JPG "Distortion Correction Image"
[image9]: ./output_images/Combined_Binary_Image.JPG "Combined Binary Image"
[image10]: ./output_images/Perspective_Tranform_Image_1.JPG "Perspective Image"
[image11]: ./output_images/Transformed_and_Topdown_View_Binary_Image.JPG "Final Binary"
[image12]: ./output_images/lane_polyfit.JPG "Lane PolyFit"
[image13]: ./output_images/lane_found_image.JPG "Lane PolyFit"



[video1]: ./challenge_video_output.mp4 "Video"
[video2]: ./challenge_video_output.mp4.mp4 "Video"


### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second and third code cell as a function of `cal_undistort(img)` of the IPython notebook located in "./Advanced Lane Finding Project_XCai.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image7]

### Pipeline (single images)

#### 1. Below is an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

I used the function `cal_undistort(img)` and apply the distortion parameters: mtx(camera matrix) and dist(distortion coefficients) calculated from the previous camera calibration to the following image.

You can notice the difference at the left and right bottom corners in the image.

![alt text][image8]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color (one is based on HLS' S value and the other is based on HSV's V value) and gradient thresholds (Sobel X absolute value) to generate a binary image (thresholding steps are in function `color_hls_thresh(img, s_thresh=(0, 255))`, `color_hsv_thresh(img, v_thresh=(0, 255))` and `abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255)`). I also applied the region of interest function from the first project (`region_of_interest(img, vertices)`) to help me narrow the range of lane feature detection. At the end, I applied a mask combination (`mask_apply(img)`) with different masking combination and threshold aiming different images from the videos. We have two testing videos - project_video.mp4 and challenge_video.mp4. I have to make some adjustment in order to work both images.  Here's an example of my output for this step.  (note: this is not actually from one of the test images).

The detail on how I pick up the above lane feature detection is in my uploaded code: `Advanced Lane Finding Project_XCai.ipynb`.

![alt text][image9]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform(img)`. You can find the function in my uploaded code `Advanced Lane Finding Project_XCai.ipynb`. The `perspective_transform(img)` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[520, 500],
    [768.5, 500],
    [210, img.shape[0]],
    [1120,img.shape[0]]]
    )
dst = np.float32(
[[0 + x_offset, 0 + y_offset],
[img.shape[1]-x_offset, 0 + y_offset],
[0 + x_offset,img.shape[0]],
[img.shape[1]-x_offset,img.shape[0]]]
)
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 500      | 350, 450        |
| 768.5, 500      | 930, 450      |
| 210, 720     | 350, 720      |
| 1120, 720      | 930, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image10]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After I apply the distortion correction, color/gradient/ROI(region_of_interest) masking and perspective transformation.  We need to find the lane pixels and identify which one is belonging to the left lane and which one is belonging to the right lane.

![alt text][image11]

Then I followed the steps from the course:

###### a) Take a histogram along the columns in the lower half of the image.
###### b) Split the histogram into two sides, one for each lane line and detect 2 peaks as the centers of left and right lanes   
###### c) Set up windows and window hyperparameters
###### d) Iterate through nwindows to track curvature
###### e) Fit a polynomial after we have found all our pixels belonging to each line through the sliding window method

I did this in function `find_lane_pixels(binary_warped)` and `fit_polynomial(binary_warped)` in my uploaded code `Advanced Lane Finding Project_XCai.ipynb`.

![alt text][image12]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I implemented the function `measure_curvature_offset_real(ploty, leftx, rightx)`. With the function, we can measure the radius of curvature closest to the vehicle. Then in the function, defined conversions in x and y from pixels space to meters in order to get the real world space.  The final output for the video is the average curvature between the left and right lanes. Also the vehicle position is calculated by the difference between the lane center and the image center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in function `draw_detected_lane(image, binary_warped, left_fitx, right_fitx, ploty, Minv )` in my code.  Here is an example of my result on a test image:

![alt text][image13]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a one link for project video output [link to my video result](./project_video_output.mp4).
Here is another one for challenge video ouput [link to my video result](./challenge_video_output.mp4).

Here I'll talk about the approach I took. The video pipeline is similar to the above one described for the image lane detection. We're going to keep track of things like where the last several detections of the lane lines were and what the curvature was, so we can properly treat new detections. To do this, it's useful to define a Line() class to keep track of all the interesting parameters we measure from frame to frame.

Once we've found the lane lines in one frame of video, and we are reasonably confident they are actually the lines that we are looking for, we don't need to search blindly in the next frame. We can simply search within a window around the previous detection. It's a method function in the lane class - `find_lane_from_prior(self, binary_warped, out_img)`.   

I created a sanity check function by checking that they are separated by approximately the right distance horizontally. In this way, we will make sure that the detection makes sense. The sanity check function is `sanity_check(left_line, right_line, left_fitx, right_fitx, ploty)`. You can find it in my uploaded project code.

I also used smoothing method over the last n (n=20) frames of video to obtain a cleaner result.

Finally, I defined line drawing function `draw_line_polygon(out_img, left_line_pts, right_line_pts, color = (0, 255, 0))` to plot the detected lanes.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  

In the beginning, I had developed a lane detection feature, which had been working well for the first testing video - project_video.mp4. However, when I had applied the same pipeline to the second video - challenge_video.mp4, it did a poor job for the lane detection. Then I had to go back to retune the lane feature masking method. Also I optimized the sanity check code as well as smoothing method. At the end, the uploaded code `Advanced Lane Finding Project_XCai.ipynb` works pretty well for both videos. There is still room to improve if I get more time on the project. For example, by adding different lane detection feature for images with different brightness, saturation values.


#### 2. Where will your pipeline likely fail?  What could you do to make it more robust?

I think my pipeline likely to fail:
* when there is a transition between high and low brightness.
* going into the shadow.
* missing one lane or two.
* if there is a car in front of my vehicles
* different light condition

With deeper into the class, I think with new techniques (e.g. deep learning), I should be able to make my pipeline more robust.
