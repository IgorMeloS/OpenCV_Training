# OpenCV

OpenCV is mandatory when the subject is computer vision. The most popular free computer vision library, OpenCV offers to us many applicability to the image preprocessing, transformation and analysis. Use OpenCV can help us to obtain better results, because before apply any model, we must have good data to train. Beyond the data preprocessing and analysis, OpenCV enable us to apply trained models and, build computer vision applications, as we will see in this project. Here, I present some examples how to use OpenCV starting from the basic concepts until deep applications. All examples here come from [PyImageSearch](https://www.pyimagesearch.com/), but with different images, in this way, we can explore more about the OpenCV tools. Let's go on ahead.

The OpenCV folder is composed by others folders, organized in the follow manner:

1. [Basics OpenCV](https://github.com/IgorMeloS/OpenCV_Training/tree/main/OpenCV/1%20-%20Basics%20OpenCV)

    This folder contains demonstrations of basic operations with OpenCV, as read image from disk or display it.  The files inside the folder are:

    * [Basics OpenCV](https://github.com/IgorMeloS/OpenCV_Training/blob/main/OpenCV/1%20-%20Basics%20OpenCV/Basics_OpenCV.ipynb)
      * Covered Skills
        * Image read
        * Image visualization
        * Image resize
        * Image rotate
        * Image smooth
        * Image draw
    * [image_transfromation](https://github.com/IgorMeloS/OpenCV_Training/blob/main/OpenCV/1%20-%20Basics%20OpenCV/image_transformation.ipynb)
      * Covered Skills
        * Translation
        * Rotation
        * Resizing
        * Flipping
        * Cropping
    * [image_transfromation2](https://github.com/IgorMeloS/OpenCV_Training/blob/main/OpenCV/1%20-%20Basics%20OpenCV/image_transformation2.ipynb)
      * Covered Skills
        * Image arithmetic
        * Bitwise aperation
        * Masking
        * Splitting and amerging channels
    * [image_histogram](https://github.com/IgorMeloS/OpenCV_Training/blob/main/OpenCV/1%20-%20Basics%20OpenCV/image_histogram.ipynb)
      * Coverd Skills
        * Image histogram
        * Image 2D histogram
        * Histogram equalization
        * Mask and histogram
    * [smoothing_blurring](https://github.com/IgorMeloS/OpenCV_Training/blob/main/OpenCV/1%20-%20Basics%20OpenCV/smoothing_blurring.ipynb)
      * Coverd Skills
        * Blurring with average method
        * Blurring with Gaussian method
        * Blurring with median method
        * Blurring with bilateral method
    * [threshold](https://github.com/IgorMeloS/OpenCV_Training/blob/main/OpenCV/1%20-%20Basics%20OpenCV/threshold.ipynb)
      * Covered Skills
        * Simple threshold (binary and inverse)
        * Adaptive threshold (mean and Gaussian)
        * Otsu and Riddler-Calvard method
    * [gradient_edgedetection](https://github.com/IgorMeloS/OpenCV_Training/blob/main/OpenCV/1%20-%20Basics%20OpenCV/gradient_edgedetection.ipynb)
      * Covered Skills
        * Gradient intensity (Laplacian and Sobel method)
        * Edge detection (Canny algorithm)
    * [contours](https://github.com/IgorMeloS/OpenCV_Training/blob/main/OpenCV/1%20-%20Basics%20OpenCV/contours.ipynb)
      * Covered Skills
        * Find contours
        * Draw contours

2. [Counting objects](https://github.com/IgorMeloS/OpenCV_Training/tree/main/OpenCV/2%20-%20Counting%20objects)

    Simple application to count object in an image using OpenCV.
    - Covered Skills
      - convert images to gray scale
      - edge detection
      - thresholding
      - finding, counting, and drawing contours
      - conducting erosion and dilation
      - mask
3. [Deep Learning face-detection with openCV](https://github.com/IgorMeloS/OpenCV_Training/tree/main/OpenCV/3%20-%20Deep%20Learning%20face-detection%20with%20openCV)

    Face detection with OpenCV and pre-trained Deep Neural Network. OpenCV library offers to us an elegant function to load pre-trained neural networks and make predictions. We must to download the weights of the dnn to load with OpenCV. In this project, we make face detection for [some images](https://github.com/IgorMeloS/OpenCV_Training/blob/main/OpenCV/3%20-%20Deep%20Learning%20face-detection%20with%20openCV/face_detector.ipynb), but it's also possible to make face recognition with your webcam by this [file](https://github.com/IgorMeloS/OpenCV_Training/blob/main/OpenCV/3%20-%20Deep%20Learning%20face-detection%20with%20openCV/face_detector_video.py).
    - Covered Skills
      - OpenCV dnn function
      - Create blobs for image transformation with dnn function
      - Make face recognition with OpenCV (drawing rectangles and printing the confidence of the prediction)
4. [OpenCV Age Detection](https://github.com/IgorMeloS/OpenCV_Training/tree/main/OpenCV/4%20-%20OpenCV%20Age%20Detection)
    
    Age detection with OpenCV requires two pre-trained models. One for face detection and another for age detection. This project presents face detection for images, and also for video stream.
    - Coverd Skills
      - OpenCV dnn function
      - Create blobs for image transformation
      - Extraction of Region of Interest (ROI)
      - Prediction inside of ROI (age detection)
5. [OpenCV Face Recognition](https://github.com/IgorMeloS/OpenCV_Training/tree/main/OpenCV/5%20-%20Face%20Recognition)
    
    Face recognition is another task that we can do with OpenCV. Previously, we’ve deployed a simple face detection, but this is not enough to recognize faces.
    With the face detection, we can grab the region of interest (ROI). From the ROI we can apply a CNN model to encode the faces into a 128-d features vector.  As soon as we have the features vector, we can classify each face using a Machine Learning classification model. In this folder, we found two others folders. The first folder is an approach using OpenCV (dnn module to create the feature vector) and a Linear regression model. In the second folder, we deploy the face recognition using the Face Recognition library, this library is shortcut of the dlib library. The method is similar to the first, the difference here is, dlib takes care about image preprocessing, as face alignment, for example. We detect and recognize faces of three Brazilian singers, Chico Buarque, Gilberto Gil and Caetano Veloso.
    
    - [Face Recognition with OpenCV](https://github.com/IgorMeloS/OpenCV_Training/tree/main/OpenCV/5%20-%20Face%20Recognition/Face%20Recognition%20with%20OpenCV)
    - [Face Recognition with dlib and facenet](https://github.com/IgorMeloS/OpenCV_Training/tree/main/OpenCV/5%20-%20Face%20Recognition/Face%20Recognition%20with%20dlib%20and%20facenet)
