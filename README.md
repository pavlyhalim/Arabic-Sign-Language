# Sign-Language
## Problem Overview

People with special needs face various challenges and barriers that isolate them from their surroundings and challenge their opportunity to work, which is one of the Egyptian grand challenges. Nowadays, several assistive technologies have been developed to reduce many of these barriers and simplify the communication between special-needs persons and the surrounding environment. However, very few frameworks are presented to support them in the Arabic region either due to the lack of resources or the Arabic language's complexity. The purpose of the study is to present a framework that will help Arabic deaf people communicate 'on the go' easily, virtually, with anyone without any specific devices or support from other people. Framework has been used to utilize cloud computing power for the complex processing of Arabic text and Videos. The video processing produced an Arabic text showing the corresponding Standard Arabic Language on the mobile handset of the deaf person. After the prototype was tested, the results showed that it successfully met the design requirements: high accuracy, precision, and being artificially intelligent (by using modern AI algorithms, CNN, that provides good performance for video life tracking). In conclusion, the prototype proved to be easily applicable due to its easy operation.


# How it works
The overview of the proposed system with four stages: data acquisition, pre-processing, feature extraction, and recognition. The images and video frames were provided as the input to the system and the output was the predicted sign digit displayed in Arabic was augmented to add variations to the dataset and the model was trained using CNN. The model was saved and loaded with OpenCV to recognize the letters in real-time

# Feature extraction

The features from the digits were extracted using the CNN algorithm as shown in figure (3). The architecture of the proposed CNN model was configured similarly to the VGGNet. However, we used only six convolutional layers compared to VGGNet which has a minimum of 13 layers. The two consecutive convolutional layers were followed by the batch normalization for faster training convergence. As the input, 54,049 RGB images of size 64 × 64 × 3 pixels were fed into the model with a batch size of 32. The model was trained and tested with two different sizes of filters, the max-pooling of two strides, ReLU, and the SoftMax activation functions. At the end, the trained model was saved.

<img alt="image" src="https://user-images.githubusercontent.com/49916453/158060312-50b07841-25f3-4fd8-b8dc-a52d71454021.png">

# Image recognition

The trained model was loaded on a laptop using TensorFlow as backend, OpenCV to read video frames, Visual Studio Code, and Python as the editor and programming language, respectively. OpenCV captures real-time hand-shaped video frames from the signer and rescaled them into 64 × 64 × 3 pixels. The model successfully detects and predicts sign digits. The trained model was used to predict ARSL letter in real-time using webcam

<img alt="image" src="https://user-images.githubusercontent.com/49916453/158060280-fd2cc2d7-7f7d-4ffb-b4e2-b21afd07081c.png">

# Sign Language Recognition Using CNN

Four layers are there in a CNN, for the classification problems. The layers are convolution layers, pooling/subsampling layers, non-linear layers, and fully connected layers. 

# Convolutional Layer 

The convolution is a special operation that extracts different features of the input. The first it extracts low-level features like edges and corners. Then higher-level layers extract higher-level features as shown. For the process of 3D convolution in CNNs. The input is of size N x N x D and is convolved with the H kernels, each of them sized to k x k x D separately. Convolution of one input with one kernel produces one output feature, and with H kernels independently produces H features, respectively. Starts from top-left corner of the input, each kernel is moved from left to right. Once the top right corner reached, kernel is moved one element downward, and once again the kernel is moved from left to right, one element at a time. Process is done continuously until the kernel reaches the bottom-right corner. 

<img alt="image" src="https://user-images.githubusercontent.com/49916453/158060248-7471dde2-c4c5-44b3-83c0-5ba15496b8dd.png">

# Preprocessing Model

Main aim of pre-processing is an improvement of the image data that reduce unwanted deviation or enhances image features for further processing. Preprocessing is also referred as an attempt to capture the important pattern which express the uniqueness in data without noise or unwanted data which includes cropping, resizing and gray scaling. Cropping refers to the removal of the unwanted parts of an image to improve framing, accentuate subject matter or change aspect ratio as shown in figure (11). Resizing Images are resized to suit the space allocated or available. Resizing image are tips for keeping quality of original image. Changing the physical size affects the physical size but not the resolution

<img alt="image" src="https://user-images.githubusercontent.com/49916453/158060201-1215c67e-4168-4db2-9047-7621a2ea4240.png"> 

  
# Dataset

 - The Arabic Alphabets Sign Language Dataset (ArASL), A new dataset consists of 54,049 images of ArASL alphabets performed by more than 40 people for 32 standard Arabic signs and alphabets as shown in Figure (8). The number of images per class differs from one class to another. Sample image of all Arabic Language Signs. The dataset contains the Label of each corresponding Arabic Sign Language Image based on the image file name.

 - Latif, Ghazanfar; Alghazo, Jaafar; Mohammad, Nazeeruddin; AlKhalaf, Roaa; AlKhalaf, Rawan (2018), “Arabic Alphabets Sign Language Dataset (ArASL)”, Mendeley Data, V1, doi: 10.17632/y7pckrw6z2.1 


# Performance/Accuracy


# Future work
  - GANs stands for Generative Adversarial Networks. These are a type of generative model because they learn to copy the data distribution of the data you give it and therefore can generate novel images that look alike.  The reason why a GAN is named “adversarial”, is because it involves two competing networks (adversaries) as shown in Figure (16), that try to outwit each other. To improve our project prediction and the dataset quality we can use the GAN to create a new process and method before the CNN. The new process with using the GAN will help to create the dataset with a higher quality using real videos and photos to generate new samples for the Arabic sign language hand moves with the perfect quality and shape to improve the hand detection in the CNN processes and. Also, keeping the dataset continuously updated and improve the quality by type. 
 


# Literature cited:

  - The World Bank, & Plecher, H. (2020, October 13). Egypt Unemployment ratedata,chart.TheGlobalEconomy.Com.https://www.theglobaleconomy.com/Egypt/unemployment_rate/

  - Pigou L., Dieleman S., Kindermans PJ., Schrauwen B. (2015) Sign Language Recognition Using Convolutional Neural Networks. In: Agapito L., Bronstein M., Rother C. (eds) Computer Vision - ECCV 2014 Workshops. ECCV 2014. Lecture Notes in Computer Science, vol 8925. Springer, Cham, doi:10.1007/978-3-319-16178-5_40

  - Jie Huang, Wengang Zhou, Houqiang Li and Weiping Li, "Sign Language Recognition using 3D convolutional neural networks," 2015 IEEE International Conference on - - Multimedia and Expo (ICME), Turin, Italy, 2015, pp. 1-6, doi: 10.1109/ICME.2015.7177428. 

  - Latif, Ghazanfar; Alghazo, Jaafar; Mohammad, Nazeeruddin; AlKhalaf, Roaa; AlKhalaf, Rawan (2018), “Arabic Alphabets Sign Language Dataset (ArASL)”, Mendeley Data, V1, doi: 10.17632/y7pckrw6z2.1

  - Karma Wangchuk, Panomkhawn Riyamongkol, Rattapoom Waranusast, Real-time Bhutanese Sign Language digits recognition system using Convolutional Neural Network,ICT Express,2020,,ISSN 2405-9595, doi:10.1016/j.icte.2020.08.00

