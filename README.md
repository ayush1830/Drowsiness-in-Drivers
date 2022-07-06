                                              User manual
Code
Available at the following Google drive link -
https://drive.google.com/drive/folders/1k4AnHUjps8oRYKfwT7htuyfKUx9L_4ex?usp=sh
aring
The folder driver-safety contains:
1. main.py - our code with functions for face detection, checks drowsiness of eyes,
yawning and head pose estimation
2. alarm1.wav - the alarm tone we have used throughout our code
3. classes.txt - contains information about the 80 classes of COCO objects that YOLO
is capable of detecting, which we used in mobile phone detection
4. facial_landmarks_68markup.jpg - image which shows the 68 landmarks features
as defined in dlib
5. fun.py - contains the code for the YOLO detection part which basically includes
the Darknet 53 CNN model
6. lowlight_enhancement.ipynb - contains the Colab notebook used for image
enhancement
7. yolov3-320.cfg, yovlov3-320.weights - supporting files for YOLO
8. shape_predictor_68_face_landmarks.dat - supporting file for using face detector
from dlib
9. Zero-DCE-1-master - the code for the model, which we have modified a bit to fit
our use - folder was very large so we have shared the drive link instead
(https://drive.google.com/drive/folders/1rh6fD54TZnqjSD8V2klDfefyJnf9P3Hb?us
p=sharing)
10. Our project report - WSD project report - Group 2.pdf (available at the link
https://docs.google.com/document/d/16JOZADSwwwf3aY7QRI6OFmbiv9wMVbB
93cc_XoGbjaQ/edit?usp=sharing)
11. Our final presentation, which summarizes all we have done - Driver Safety
(Final).pdf (available at the link
https://docs.google.com/document/d/1NWUZKcfsl3yF_rP2qzy2pLfN7XSqWZqFKi
QebApti4o/edit?usp=sharing)
12. final_video.mp4 - demonstration video
13. User manual - A user manual for the running the entire code
14. requirements.txt - which contains all the libraries needed to be installed
15. Performance check.pdf - our accuracy check of the model
16. test_result.xlsx - which contains details about the runs we did on the modelFOR NORMAL LIGHT
1. Install PyCharm or VS code
(https://www.toolsqa.com/blogs/install-visual-studio-code/,
https://www.guru99.com/how-to-install-python.html)
2. Install opencv by typing the following command in the terminal:
pip install opencv-python
3. Install numpy
pip install numpy
4. Install dlib library with the help of the following steps:
The following link can be directly referred to in case of vs code:
https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-
10-57348ba1117f
Then install cmake using the following terminal command:
pip install cmake
And then dlib using the following command
pip install dlib
5. Download the shape face predictor of dlib from
https://github.com/davisking/dlib-models and extract the dat file
6. Install pygame - this is for playing the alarm.This can be done with the help of the
following command in the terminal:
pip install pygame
7. Install tensorflow with the help of the following command:
pip install tensorflowUse the following command in case errors come up:
pip install tensorflow == 2.2
8. In our drive, there is a file named fun.py. Please download that directly, it
contains the CNN model that will work for mobile phone detection.
9. Install wget, math and scipy libraries - with
pip install wget
pip install math
pip install scipy
10. Open this link to download the yolo3.weights file -
https://pjreddie.com/media/files/yolov3.weights
This is also available in the given drive link
11. Run main.py from the drive link
While using the cv2.VideoCapture(), use 0 argument to use the system’s own webcam. In
case you want to use an external camera, use argument 1. Install Droidcam client in
order to use the secondary camera
Running the low light enhancer on colab
1. Open Google Colab
2. The reason we are using colab for this is because it offers a free GPU driver with
limited usage (over time).
3. Open the file lowlight_enhancement.ipynb on Colab
4. Run each cell, and keep the video you want to enhance in the Zero-DCE_code
folder
5. Give the name as the argument in the VideoCapture() method, or the path to the
file
6. Run that cell
7. You will get the enhanced video in the same folder as the start
8. Be sure to run the penultimate cell to initialize the values each time
