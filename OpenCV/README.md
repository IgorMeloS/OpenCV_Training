# OpenCV

OpenCV is mandatory when the subject is computer vision. The most popular free computer vision library, OpenCV offers to us many applicability to the image preprocessing, transformation and analysis. Use OpenCV can help us to obtain better results, because before apply any model, we must have a good data to train our model. Here, I present some examples how to use OpenCV starting from the basic concepts until deep applications.

The OpenCV folder is composed by others folders, organized in the follow manner:

1. [Basics OpenCV](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/OpenCV/1%20-%20Basics%20OpenC)

    This folder contains demonstrations of basic operations with OpenCV, as read image from disk or display it.  The files inside the folder are:

    * [Basics OpenCV](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/OpenCV/1%20-%20Basics%20OpenCV/Basics_OpenCV.ipynb)
      * Covered Skills
        * Image read
        * Image visualization
        * Image resize
        * Image rotate
        * Image smooth
        * Image draw
    * [image_transfromation](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/OpenCV/1%20-%20Basics%20OpenCV/image_transformation.ipynb)
      * Covered Skills
        * Translation
        * Rotation
        * Resizing
        * Flipping
        * Cropping
    * [image_transfromation2](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/OpenCV/1%20-%20Basics%20OpenCV/image_transformation2.ipynb)
      * Covered Skills
        * Image arithmetic
        * Bitwise aperation
        * Masking
        * Splitting and amerging channels
    * [image_histogram](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/OpenCV/1%20-%20Basics%20OpenCV/image_histogram.ipynb)
      * Coverd Skills
        * Image histogram
        * Image 2D histogram
        * Histogram equalization
        * Mask and histogram
    * [smoothing_blurring](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/OpenCV/1%20-%20Basics%20OpenCV/smoothing_blurring.ipynb)
      * Coverd Skills
        * Blurring with average method
        * Blurring with Gaussian method
        * Blurring with median method
        * Blurring with bilateral method
    * [threshold](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/OpenCV/1%20-%20Basics%20OpenCV/threshold.ipynb)
      * Covered Skills
        * Simple threshold (binary and inverse)
        * Adaptive threshold (mean and Gaussian)
        * Otsu and Riddler-Calvard method
    * [gradient_edgedetection](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/OpenCV/1%20-%20Basics%20OpenCV/gradient_edgedetection.ipynb)
      * Covered Skills
        * Gradient intensity (Laplacian and Sobel method)
        * Edge detection (Canny algorithm)
    * [contours](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/OpenCV/1%20-%20Basics%20OpenCV/contours.ipynb)
      * Covered Skills
        * Find contours
        * Draw contours

2. [Counting objects](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/OpenCV/2%20-%20Counting%20objects/counting_objects.ipynb)

    Simple application to count object in an image using OpenCV.
    - Covered Skills
      - convert images to gray scale
      - edge detection
      - thresholding
      - finding, counting, and drawing contours
      - conducting erosion and dilation
      - mask
3. [Deep Learning face-detection with openCV](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/OpenCV)

    Face detection with OpenCV and a pre-trained Deep Neural Network. OpenCV library offers to us an elegant function to load pre-trained neural networks and make predictions. We must to download the weights of the dnn to load with OpenCV. In this project, we make face detection for [some images](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/OpenCV/3%20-%20Deep%20Learning%20face-detection%20with%20openCV/face_detector.ipynb), but it's also possible to make face recognition with your webcam by this [file](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/OpenCV/3%20-%20Deep%20Learning%20face-detection%20with%20openCV/face_detector_video.py).
    - Covered Skills
      - OpenCV dnn function
      - Create blobs from an image with dnn function
      - Make face recognition with OpenCV (drawing rectangles and printing the confidence of the prediction)
