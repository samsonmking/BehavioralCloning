# **Behavioral Cloning**

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report

[//]: # "Image References"
[image1]: ./examples/arch.png "Model Visualization"
[image2]: ./examples/center.jpg "Center Driving"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

- model.py containing the script to create and train the model
- drive.py for driving the car in autonomous mode
- model.h5 containing a trained convolution neural network
- README.md summarizing the results
- video.mp4 a video recording of my final result

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

This model was based off of the [NVIDIA end to end deep learning for self-driving cars](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) paper.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers between dense layers in order to reduce overfitting (model.py lines 78, 80).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 61). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 83).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I recorded the following scenarios to use as training data:

- 2 Laps of centered driving
- 1 Lap of cornering
- 1 Lap of recovery driving - moving back towards the center if the car got off the path

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

I began by implementing a very simple model with just one dense layer in order to verify I was able to import training data correctly and interact with the simulator.

After that, I began researching similar problems and decided to implement the model from [NVIDIA end to end deep learning for self-driving cars](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I made use of a Keras callbacks to save the model everytime performance improved on the validation to ensure the best performing model was captured. I then made use of another callback that would stop training if performance didn't imporove on the validation set after three epochs. This helps to prevent overfitting by stopping training if performance on the validation set is no longer improving.

After training with this approach, training would stop after only a few epochs (~5), as the model began to quickly overfit. To help combat this, I added dropout between the dense layers.

With dropout, I was able to train for more epochs, but there were still a couple scenarios where the car got dangerously close to the edge of the road. I augmented my training set by using data from the left and right cameras, and inverting the center camera image. See section 3 below for more details on this process. With the augmented data, the car was able to drive around the track smoothly.

#### 2. Final Model Architecture

| Layer (type)              | Output Shape        | Param # |
| ------------------------- | ------------------- | ------- |
| lambda_1 (Lambda)         | (None, 160, 320, 3) | 0       |
| cropping2d_1 (Cropping2D) | (None, 65, 320, 3)  | 0       |
| conv2d_1 (Conv2D)         | (None, 31, 158, 24) | 1824    |
| conv2d_2 (Conv2D)         | (None, 14, 77, 36)  | 21636   |
| conv2d_3 (Conv2D)         | (None, 5, 37, 48)   | 43248   |
| conv2d_4 (Conv2D)         | (None, 3, 35, 64)   | 27712   |
| conv2d_5 (Conv2D)         | (None, 1, 33, 64)   | 36928   |
| flatten_1 (Flatten)       | (None, 2112)        | 0       |
| dense_1 (Dense)           | (None, 100)         | 211300  |
| dropout_1 (Dropout)       | (None, 100)         | 0       |
| dense_2 (Dense)           | (None, 50)          | 5050    |
| dropout_2 (Dropout)       | (None, 50)          | 0       |
| dense_3 (Dense)           | (None, 10)          | 510     |
| dense_4 (Dense)           | (None, 1)           | 11      |

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it deviated from the center.

I then recorded a laps worth of cornering to avoid biasing straight driving.

To augment the data sat, I also flipped images and angles thinking that this would help prevent bias toward turning left instead of right. I also used the left and right camera images and applied an offset to the steering angle to offset the difference in camera angle. This can be seen in `model.py` lines 48 and 48.

I preprocessed this data by cropping the image 70 pixels from the top and 25 pixels from the bottom. This helped the model to focus on the road instead of the car hood and objects in the horizon (like trees).

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 17 as determined by the callbacks described above. I used an adam optimizer so that manually training the learning rate wasn't necessary.
