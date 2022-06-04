# Distracted Driver Detection

This project that is realized with machine learning, detects distracted drivers from an image that a user upload. 
The data that was used as training and testing data to build the model consists on a set of images, each taken in a car where the driver is doing some action, e.g. texting, talking on the phone, doing their makeup, etc.

The images are labeled following a set of 10 categories:

Class | Description 
--- | --- 
c0 | Safe driving 
c1 | Texting with right hand
c2 | Talking on the phone with right hand
c3 | Texting with left hand
c4 | Talking on the phone with left hand
c5 | Operating the radio
c6 | Drinking
c7 | Reaching behind
c8 | Hair and makeup
c9 | Talking to passenger

## Dependencies

* Python 3.8
* Keras 2.6.0
* matplotlib 3.3.3
* numpy 1.19.3

The dataset used is available on Kaggle (https://www.kaggle.com/c/state-farm-distracted-driver-detection).

