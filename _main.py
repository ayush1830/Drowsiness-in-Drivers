import cv2
import dlib
import math
from scipy.spatial import distance
from pygame import mixer
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization
)
from tensorflow.keras.regularizers import l2
import wget
import fun
from fun import (YoloV3,load_darknet_weights,draw_outputs,DarknetConv,DarknetResidual,DarknetBlock,Darknet,YoloConv,YoloOutput,yolo_boxes,yolo_nms,weights_download,calculate_eye,upperlayer_calculate_lips)
cap = cv2.VideoCapture(1)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_COMPLEX
font_scale = 0.5
sleep_score1=0
mixer.init()
alarm = mixer.Sound('alarm1.wav')
sleep_score2=0
frame_count=0
Initial_nose_tip_X=[]
count_X=0
average_nose_X = 0
yolo = YoloV3()
load_darknet_weights(yolo,'yolov3-320.weights')

while True:
    _, frame = cap.read()
    height,width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)
    img = img / 255
    class_names = [c.strip() for c in open("classes.txt").readlines()]
    boxes, scores, classes, nums = yolo(img)
    count=0
    for i in range(nums[0]):
        if int(classes[0][i] == 0):
            count +=1
        if int(classes[0][i] == 67):
            
            alarm.play()
    frame = draw_outputs(frame, (boxes, scores, classes, nums), class_names)
    faces = hog_face_detector(gray)
    for face in faces:
        X1=face.left()
        Y1=face.top()
        X2=face.right()
        Y2=face.bottom()
        cv2.rectangle(frame,(X1,Y1),(X2,Y2),(0,255,0),3)
        face_landmarks = dlib_facelandmark(gray,face)
        for n in range(0,68):
                X=face_landmarks.part(n).x
                Y=face_landmarks.part(n).y
                cv2.circle(frame,(X,Y),3,(255,0,0),-1)

        face_landmarks = dlib_facelandmark(gray,face)
        X=face_landmarks.part(34).x
        Y=face_landmarks.part(34).y
        if(len(Initial_nose_tip_X) < 10):
                Initial_nose_tip_X.append(X)
        if(len(Initial_nose_tip_X)==10):
                average_nose_X = int(sum(Initial_nose_tip_X)/10)
        leftEye = []
        rightEye = []

        if(X>(average_nose_X + 70)):
                count_X=count_X+1
        if(X<(average_nose_X - 70)):
                count_X=count_X+1
        if(X<(average_nose_X+70) and X>(average_nose_X - 70)):
                count_X=count_X-1
        if(count_X<0):
                count_X=0
        if(count_X > 40):
                alarm.play()
        if(count_X < 40):
                alarm.stop()
        
        
        cv2.putText(frame,'Pose Error:'+str(count_X),(width-250,height-160), font, 1,(0,0,255),1,cv2.LINE_AA)

        for n in range(36,42):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                leftEye.append((x,y))
                next_point = n+1
                if n == 41:
                        next_point = 36
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame,(x,y),(x2,y2),(123,255,0),1)

        for n in range(42,48):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                rightEye.append((x,y))
                next_point = n+1
                if n == 47:
                        next_point = 42
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame,(x,y),(x2,y2),(123,255,0),1)        
        upperlips = []
        for n in range(48,60):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                upperlips.append((x,y))
                next_point = n+1
                if n == 59:
                        next_point = 48
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame,(x,y),(x2,y2),(0,255,123),1)
        left_eye = calculate_eye(leftEye)
        right_eye = calculate_eye(rightEye)
        EYE = (left_eye+right_eye)/2
        EYE = round(EYE,2)
        upper_lips = upperlayer_calculate_lips(upperlips)
        LIPS = upper_lips
        LIPS = round(LIPS,2)
        if EYE < 0.24:
                sleep_score1=sleep_score1+1
                cv2.putText(frame,"Closed eyes",(10,height-20), font, 1,(255,0,0),1,cv2.LINE_AA)
        else:
                sleep_score1=sleep_score1-1
                cv2.putText(frame,"Open eyes",(10,height-20), font, 1,(255,0,0),1,cv2.LINE_AA)
        
        if LIPS > .8:
                sleep_score2=sleep_score2+1
                cv2.putText(frame,"Yawning",(10,height-80), font, 1,(255,0,0),1,cv2.LINE_AA)
                if (frame_count==0):
                        frame_count=frame_count+1   #it will initiate the counting of total number of frames once we will start yawning
                
        else:
                cv2.putText(frame,"Not Yawning",(10,height-80), font, 1,(0,255,255),1,cv2.LINE_AA)
        
        if(sleep_score1<0):
                sleep_score1=0

        cv2.putText(frame,'blinking:'+str(sleep_score1),(width-200,height-20), font, 1,(0,0,255),1,cv2.LINE_AA)
        if(sleep_score1>40):
                #person is feeling sleepy so we beep the alarm
                cv2.putText(frame, "ALERT!", (10,50), font, 2, (0, 0, 255), 4)
                try:
                    alarm.play()

                except:  # isplaying = False
                    pass
                
         
        cv2.putText(frame,'Yawning:'+str(sleep_score2),(width-200,height-80), font, 1,(0,0,255),1,cv2.LINE_AA)

        if(sleep_score2)>50:         #For demonstration purpose it is 50 
                cv2.putText(frame, "ALERT!", (10,50), font, 2, (0, 0, 255), 4)

                try:
                    alarm.play()
                except:  
                    pass
        
        if(frame_count>0):
                frame_count=frame_count+1

        if (frame_count > 54000):          #54000=30*60*30
                sleep_score2=0
                frame_count=0
        

    cv2.imshow("Drowsiness Detector",frame)
    key = cv2.waitKey(1)

    if (key==ord('d')):
            alarm.stop()
            sleep_score1=0
            sleep_score2=0
            count_X=0
            

    if (key == 27):
        break

cap.release()
cv2.destroyAllWindows()