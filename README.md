# Sign-Language
## Problem Overview
  People with special-needs face a variety of different challenges and barriers that isolate them from their surroundings. Nowadays, several assistive technologies have been developed to reduce many of these barriers and simplify the communication between special-needs persons and the surrounding environment. However, few frameworks are presented to support them in the Arabic region either due to the lack of resources or the complexity of the Arabic language. The main goal of this work is to present a mobile-based framework that will help Arabic deaf people to communicate ‘on the go’ easily with virtually any one without the need of any specific devices or support from other people. 

# How it works
  We use the framework utilizes the power of cloud computing for the complex processing of the Arabic text and Videos. The video processing produced a Arabic text showing the corresponding Standard Arabic Language on the mobile handset of the deaf person.

# Hand tracking and facial landmarks
  - ### Run simple (Character level)
  ```
  $ python3 simple_test.py
  ```
  
  - ### Run (Character level - hand detection- facial landmarks)
      NOTE: Download [facial landmarks](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) model and put it in  <b>landmarks folder.
  ```
  $ python3 ASL_detection_landmark.py
  ```
  - ### Run (hand detection & tracking)
  ```
  $ python3 hand_detection_tracking.py
  ```
  

# Sample
- ## Text to Sign Language
  <img src="https://github.com/7AM7/Sign-Language/blob/master/videos/Text-to-sign-gif.gif" width="160" height="280" />
- ## Sign Language to Text
  <img src="https://github.com/7AM7/Sign-Language/blob/master/videos/sign-to-text-gif.gif" width="280" height="200" />

# Dataset
  - #### [SignsWorld Atlas](https://data.mendeley.com/datasets/y7pckrw6z2/1?fbclid=IwAR0ucbKGH9VdkzI2LccuTnk5wcoMQ0odAKSQkq6wKmG9-cvsDj4hwm9Rnb8)
    - Description of this data
      - A new dataset consists of 54,049 images of ArSL alphabets performed by more than 40 people for 32 standard Arabic signs and alphabets. The number of images per class differs from one class to another. Sample image of all Arabic Language Signs is also attached. The CSV file contains the Label of each corresponding Arabic Sign Language Image based on the image file name.
      - ### Note: [Google Colab](https://drive.google.com/open?id=18LIsB5eia_HQ342jxD3MWOVlWnit_yLL)

# Our Approach

# Performance/Accuracy

# Benchmark

# Future work
  - ### Build simple Machine Learning model with [SignsWorld Atlas](https://data.mendeley.com/datasets/y7pckrw6z2/1?fbclid=IwAR0ucbKGH9VdkzI2LccuTnk5wcoMQ0odAKSQkq6wKmG9-cvsDj4hwm9Rnb8).
  - ### Build simple Deep Learning model with [SignsWorld Atlas](https://data.mendeley.com/datasets/y7pckrw6z2/1?fbclid=IwAR0ucbKGH9VdkzI2LccuTnk5wcoMQ0odAKSQkq6wKmG9-cvsDj4hwm9Rnb8).

# Resources
- [Sign language recognition using scikit-learn](https://www.freecodecamp.org/news/weekend-projects-sign-language-and-static-gesture-recognition-using-scikit-learn-60813d600e79/?fbclid=IwAR12SNgtkL9rydJ2-n-cMtA-P2uK4b4OWde8GgEwXtedbw-sJHAERpJDlfE) is an introduction Sign language using ML and how it works.
- [Sign Language Recognition Datasets](http://facundoq.github.io/unlp/sign_language_datasets/?fbclid=IwAR2fBOoA97S_IiUgfLdaEVb3kKqld2quDk2_6oAEHDk4_gz22wnNWphJcQ4).
- [ArASL](https://www.sciencedirect.com/science/article/pii/S2352340919301283).
- [SignsWorld Atlas benchmark](https://www.sciencedirect.com/science/article/pii/S1319157814000548)
- [hand detection](https://github.com/victordibia/handtracking)
- [hand tracking](https://github.com/adipandas/multi-object-tracker)
