#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model


# In[2]:


def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), 
              shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):
    return gen.flow_from_directory(dir,batch_size=batch_size,
                                   shuffle=shuffle,color_mode='grayscale',
                                   class_mode=class_mode,target_size=target_size)


# In[3]:


BS= 32
TS=(24,24)
train_batch= generator(f"{os.getcwd()}/TrainDataset",shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator(f"{os.getcwd()}/TestDataset",shuffle=True, batch_size=BS,target_size=TS)
stepsPerEpoch= len(train_batch.classes)//BS
validationSteps = len(valid_batch.classes)//BS
print("Steps per epoch: {}\nValidation steps: {}".format(stepsPerEpoch,validationSteps))


# In[4]:


model = Sequential([
    
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),
    
    
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
    
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
    Dropout(0.25),
    
    
    Flatten(),
    
  
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.summary()


# In[5]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[9]:


model.fit(train_batch, 
                    validation_data=valid_batch, 
                    epochs=15, steps_per_epoch=stepsPerEpoch ,validation_steps=validationSteps)


# In[ ]:





# In[10]:


model.save("/Users/yagmurkahya/Desktop/DrowsyDetection/son_model.h5",overwrite=True)


# In[ ]:




