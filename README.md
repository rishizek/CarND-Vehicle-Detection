# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/sample_training_images.jpg
[image2]: ./output_images/hog_images_RGB.jpg
[image3]: ./output_images/hog_images_YCrCb.jpg
[image4]: ./output_images/sliding_windows.jpg
[image5]: ./output_images/improved_sliding_windows.jpg
[image6]: ./output_images/improved_sliding_windows.jpg
[image7]: ./output_images/sliding_windows_with_threshold.jpg
[image8]: ./output_images/sliding_windows_with_label_and_threshold.jpg
[video1]: ./output_video_20_07_scale13.mp4

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the section 3 of 
the IPython notebook (`vehicle_detection.ipynb`).  

I first analyzed the `vehicle` and `non-vehicle` sample images.  
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters 
including orientations, pixel per cell, and cells per block.
I applied HOG feature visualization to the above example in `RGB` and `YCrCb` color 
spaces as the former is one of the most basic color space and the latter demonstrated 
the highest classification accuracy at this project.

For both color spaces, when the images are converted to HOG feature images, 
we can observe the circular feature flows around car in car images whereas
more parallel feature flow in non-car images. It might be difficult for human 
to tell the difference between HOG images of `RGB` and `YCrCb` color spaces, but 
the channels of `YCrCb` HOG images might appear to capture more strong marks of object,
and this could be the reason that the classification accuracy of `YCrCb` usually 
bettr than that of `RGB` at this project.

Here is an example using the `RGB` color space and HOG parameters of 
`orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:
![alt text][image2]

Here is an example using the `YCrCb` color space and HOG parameters of 
`orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:
![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters in difference color spaces 
(`RGB`, `HSV`, `LUV`, `HLS`, `YUV`, `YCrCb`), specifically targeting to 
have the highest classification accuracy of the car and not-car images.
It turns out that the model with `YCrCb` hog features performs the best (99.61%),
and following `LUV` (99.44%), `YUV` (99.38%), `HSV` (99.27%), `HLS` (99.04%), 
`RGB` (98.37%). Note that I did not use holdout set to test 
accuracy because the purpose of this project is vehicle detection
and tracking of video, and not car classification of single image.
So the accuracy for holdout set, if tested, might be lower than the result
shown above. The codes for this analysis is
contained in the section 4 of the IPython notebook 
(`vehicle_detection.ipynb`).  


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM, logistic Regression and Random Forest using the above
mentioned hog features, and spatial and histogram features combined. 
Since there are always change that the different features have different
mean and variance, which usually make model difficult to learn. I normalized
features using `sklearn.StandardScaler()` function to improve performance.
Then, I split the deta set to training and test set to observe if model is not
over-fitting at the evaluation. 

I also trained the data set using three distinct 
classifier (`Linear SVM`, `Logistic Regression`, and `Random Forest`).
With the naive application with the default model parameters, Logistic Regression
performs the best (99.66%), following Linear SVM (99.61%), and Random
Forest (98.37%). In term of the evaluation speed, the performance follows 
Logistic Regression (0.02 second), Random Forest (0.06 second), and 
Linear SVM (0.45 second).  Although I stick with Linear SVM for this
project, the Logistic Regression could be another good choice for the vehicle
classifier. The codes for this analysis is
contained in the last cell in the section 4 of the IPython notebook 
(`vehicle_detection.ipynb`).  

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I tried to search a single car with sliding window of some overlaps, and also
restricted the image search range in the range that the cars can possibly exist 
(not in sky, for example). 
This restriction improves the computation speed as well as the vehicle 
detection accuracy. Although the classifier's accuracy is above 99%,
because of the various scaling of car in each window, there are many false positives 
and false negative when applied to the video images. The codes for this analysis is
contained in the section 5 of the IPython notebook  (`vehicle_detection.ipynb`).
Here is example images of how I applied
sliding windows to the test images and how well the model is able to identify car:

![alt text][image4]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Then I further improved the model with HOG sub-sampling window search,
which allows us to extract the HOG feature only once per frame and 
to sub-sample all of its overlaying windows. This improves computation 
speed further and reduces more false negatives. Although it could increase false 
positives, we can filter out them later as shown below.
The codes for this analysis is
contained in the section 6 of the IPython notebook (`vehicle_detection.ipynb`).

Here are some example images with heat map:

![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the heatmap created by the positions of positive detections 
in each frame of the video using `Tracker` class.
I then averaged lasat 20 frame of the heatmap and thresholded 
that heatmap to identify vehicle positions.  
I then used `scipy.ndimage.measurements.label()` to identify individual 
blobs in the averaged heatmap.  I then assumed each blob corresponded to a vehicle.  
I constructed bounding boxes to cover the area of each blob detected.  

The codes for `Tracker` class is
contained in the section 8 and those for the label box and threshold
in section 7 of the IPython notebook (`vehicle_detection.ipynb`).

Here's an example result showing the heatmap, 
the result of thresholding and applicaiton of 
`scipy.ndimage.measurements.label()` on the test images:

### Here are six frames and their corresponding heatmaps:

![alt text][image6]

### Here is the result of threshold on the integrated heatmap from all six images:
![alt text][image7]

The false positives are disappeared after the threshold. 
Note that for the purpose of explanation I applied the threshold to the 
heatmap of single frame here. However, when I applied to video, threshold
was applied to the the averaged heatmap of 20 viedo frames.

### Here the resulting bounding boxes and heatmap after threshold and `scipy.ndimage.measurements.label()` :
![alt text][image8]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I first created the vehicle and non-vehicle classifier, which is very accurate
by itself. I then used the sliding window search to extract a certain box from the
video frame to identify if there is a car in the box. Problems occur when the classifier
is fail to classify correctly. Although the classifier itself has over 99% accuracy when
applied to correctly scaled image, because the the scale of cars in the video frame 
varies significantly depending how far the target objects from the camera. 
The main two issues I faced in this project are as follow:
1. Temporal False Positive generated by not-Car object. The guard rails and trees are
occasionally identified as car and they caused false positives. 
To prevent this issue I used the heatmap to identify most
car likely objects, took the average of last 20 heatmaps, and then 
applied threshold to filter temporal false negative that only appears in a few video frames. 
After the application of this method, I was able to filter out all false 
positive for the project video successfully.

2. False Negative caused by scaling issue. At the first stage of my trials,
I was always missed a white car at certain period of video frames. I initially
suspected that my tracking method is not good enough and tried to tune model
by changing the average period of heatmap and threshold, but failed to 
improve the false negative. It turned out that it was not due to tracking method
but classier was keep failing to identify car at first place because of the inappropriate scale.
After adjusting the scaling factor properly, I was able to resolve this issue.

To improve the model, I can think of two approaches at the moment. First, 
by implementing more sophisticated tracking method, the false positive and
false negative rate could be reduced. Currently, I am tracking cars by simply
averaging heatmap for last 20 frames and then applying the threshold. However, if I am 
able to tack individual cars independently, it is much easier to interpolate
temporarily missing cars and remove non-car objects, and it will improve the model.
Secondly, there is high chance that by adjusting the scaling factor of 
window according to the position of window in the image, the model could be improved.
The main cause that the classifier fail is due to the incorrect scaling, 
because when the scaling is appropriate, the accuracy of the
classifier is over 99%. Fortunately, we know that the cars in the video
frame are large at the central bottom and getting small along with window moves to 
upper location of the video frame. Considering this, if we modified window scale depending
of the location of image, the accuracy of the classifier improves and as the 
result the vehicle detector will also be improved.
