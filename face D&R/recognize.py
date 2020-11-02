# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:58:30 2020

@author: tripprakhar
"""


import numpy as np
import cv2
import pickle


face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

labels ={}
with open("labels.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}


recognizer.read("trainner.yml") 
cap = cv2.VideoCapture(0)

def change_res(cap,width, height):
    cap.set(3, width)
    cap.set(4, height)
    
change_res(cap,640,480)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = frame[y:y+h, x:x+w]
        #img_item = "my-image.png"
        #cv2.imwrite(img_item,roi_gray)
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45:# and conf<=85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color=(255,255,255)
            stroke=1
            cv2.putText(frame,name , (x,y), font ,1,color,stroke, cv2.LINE_AA)
            
        #img_item="7.png"
        #cv2.imwrite(img_item, roi_color)
        
        color=(255,0,0) #BGR
        stroke = 2
        widthxend=x+w;
        heightyend = y+h
        cv2.rectangle(frame,(x,y),(widthxend,heightyend),color, stroke)
        
        
        
        
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()