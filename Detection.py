#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer


# In[2]:


mixer.init()
sound=mixer.Sound("/Users/yagmurkahya/Desktop/DrowsyDetection/alarm.wav")
#sound.play()


# In[3]:


face = cv2.CascadeClassifier(f'{os.getcwd()}/cascades/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(f'{os.getcwd()}/cascades/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(f'{os.getcwd()}/cascades/haarcascade_righteye_2splits.xml')


# In[4]:


model = load_model(f'{os.getcwd()}/son_model.h5')
path = os.getcwd()


cap = cv2.VideoCapture(1)
if cap.isOpened() == True:
    print("Video açıldı.")
else:
    print("Problem")
cap.set(cv2.CAP_PROP_FRAME_WIDTH,300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 350)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL


score = 0
sınır = 10
cizgi = 2
rpred = [99]
lpred = [99]


# In[5]:


while(True):
   
    
    ret, frame = cap.read()
    height,width = frame.shape[:2]
    
    
  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)
    
    
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
        cv2.putText(frame,"Closed",(7,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        
        
        if score > sınır + 1:
            score = sınır
    else:
        score -= 1
        cv2.putText(frame,"Open",(7,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
   

    if(score < 0):
        score = 0   
    cv2.putText(frame,'Drowsiness Score:'+str(score),(80,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    
    if(score>sınır):
        
        cv2.imwrite(os.path.join(path,'closed_eyes.jpg'),frame)
        try:
            sound.play()
        except:  
            pass
        
        
        if(cizgi<16):
            cizgi += 2
        
        else:
            cizgi -= 2
            if(cizgi<2):
                cizgi=2
        cv2.rectangle(frame, (0,0), (width,height), (0,0,255), thickness=thicc)
        
  
    cv2.imshow('frame',frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




