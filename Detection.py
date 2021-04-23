#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer


# In[3]:


mixer.init()
sound=mixer.Sound("/Users/yagmurkahya/Desktop/DrowsyDetection/alarm.wav")
#sound.play()


# In[4]:


face = cv2.CascadeClassifier(f'{os.getcwd()}/cascades/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(f'{os.getcwd()}/cascades/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(f'{os.getcwd()}/cascades/haarcascade_righteye_2splits.xml')


# In[5]:


model = load_model(f'{os.getcwd()}/son_model.h5')
path = os.getcwd()

# Capture accesses the video feed. The "0" is the number of your video device, in case you have multiple.
cap = cv2.VideoCapture(0)
if cap.isOpened() == True:
    print("Video stream open.")
else:
    print("Problem opening video stream.")

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# starting value for closed-eyes inferences
score = 0
threshold = 6
thicc = 2
rpred = [99]
lpred = [99]


# In[6]:


while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2]
    
    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Haar Cascade object detection in OpenCV to gray frame
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)
    
    # draw black bars top and bottom
    cv2.rectangle(frame, (0,height-50) , (width,height) , (0,0,0) , thickness=cv2.FILLED )
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
        
        
    for (x,y,w,h) in right_eye:
        r_eye = frame[y:y+h,x:x+w]
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye = r_eye/255
        r_eye =  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        break
        
        
        
    for (x,y,w,h) in left_eye:
        l_eye = frame[y:y+h,x:x+w]
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye = l_eye/255
        l_eye =l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        break
        
        
        
    if(rpred[0]==0 and lpred[0]==0):
        score += 1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        # prevent a runaway score beyond threshold
        if score > threshold + 1:
            score = threshold
    else:
        score -= 1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    # SCORE HANDLING
    # print current score to screen
    if(score < 0):
        score = 0   
    cv2.putText(frame,'Drowsiness Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    # threshold exceedance
    if(score>threshold):
        # save a frame when threshold exceeded and play sound
        cv2.imwrite(os.path.join(path,'closed_eyes_screencap.jpg'),frame)
        try:
            sound.play()
        except:  # isplaying = False
            pass
        
        # add red box as warning signal and make box thicker
        if(thicc<16):
            thicc += 2
        # make box thinner again, to give it a pulsating appearance
        else:
            thicc -= 2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame, (0,0), (width,height), (0,0,255), thickness=thicc)
        
    # draw frame with all the stuff with have added
    cv2.imshow('frame',frame)
    
    # break the infinite loop when pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close stream capture and close window
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




