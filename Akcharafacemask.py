# -*- coding: utf-8 -*-
"""
Created on 27 Mar 21

@author: Anitha
"""


import numpy as np
import keras
import keras.backend as k
from keras.layers import Conv2D,MaxPooling2D,SpatialDropout2D,Flatten,Dropout,Dense
from keras.models import Sequential,load_model
from keras.optimizers import Adam
from keras.preprocessing import image
import cv2
import datetime
import time


# UNCOMMENT THE FOLLOWING CODE TO TRAIN THE CNN FROM SCRATCH

# BUILDING MODEL TO CLASSIFY BETWEEN MASK AND NO MASK

# IMPLEMENTING LIVE DETECTION OF FACE MASK

mymodel=load_model('mymodel.h5')

cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    _,img=cap.read()
    face=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
    for(x,y,w,h) in face:
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite('temp.jpg',face_img)
        test_image=image.load_img('temp.jpg',target_size=(150,150,3))
        test_image=image.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        #test_image=np.argmax(mymodel.predict(test_image), axis=-1)
        pred=mymodel.predict_classes(test_image)[0][0]
        #pred=np.argmax(model.predict(), axis=-1)
        print("Prediction:", pred)
        if pred==1:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(img,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(img,'Thank you for wearing Mask',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
            
        datet=str(datetime.datetime.now())
        cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        pred=str(pred)
        cv2.putText(img,pred,(40,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        time.sleep(3)
          
    cv2.imshow('Akchara Facemask Detector',img)
    
    if cv2.waitKey(1)==ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        quit()
        break
    
cap.release()
cv2.destroyAllWindows()
exit()
quit()
