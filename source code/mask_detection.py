#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from joblib import dump,load


# In[3]:


face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')


# In[4]:


TRAIN=np.load('train.npy')
TEST=np.load('test.npy')


# In[5]:


x_train,x_test,y_train,y_test=train_test_split(TRAIN,TEST,test_size=0.30)

#print(classification_report(y_test,y_pred))
# In[6]:


op=SVC()
op.fit(x_train,y_train)
y_pred=op.predict(x_test)
accuracy_score(y_test,y_pred)


# In[7]:


print(classification_report(y_test,y_pred))


# In[8]:


model=op
dump(op,'modell.joblib')
model_joblib=load('modell.joblib')


# In[9]:


with open('model_pickle','wb') as f:
    pickle.dump(op,f)
with open('model_pickle','rb') as f:
    model=pickle.load(f)


# In[13]:


lol=1
font=cv2.FONT_HERSHEY_COMPLEX
d={1:'MASK',0:'No Mask'}
output='example.avi'
fps=30
codec='MJPG'
fourcc=cv2.VideoWriter_fourcc(*codec)
writer=None
(h,w)=None,None
zeros=None


# In[14]:


vs = cv2.VideoCapture(0)

while True:
    flag, img = vs.read()
    if flag:
        img = imutils.resize(img, width=1000)
        img=cv2.flip(img,1)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(30, 30))
        a,b=0,0
        for x,y,w,h in faces:  
            frame=cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),4)
            face=img[y:y+h,x:x+w,:]
            face=cv2.resize(face,(100,100))
            face=face.reshape(1,-1)
            pred=model.predict(face)[0]
            n=int(pred)
            if(n==1): b+=1
            else: a+=1
            m=d[n]
            if n==0: cv2.putText(img,m,(x,y),font,1,(0,0,255),2)
            if n==1: cv2.putText(img,m,(x,y),font,1,(0,255,0),2)
        #print(a,b,len(faces))
        (i,j)=img.shape[:2]
        if writer is None:
            writer=cv2.VideoWriter(output,fourcc,fps,(j,i),True)
        cv2.rectangle(img,(0,0),(200,60),(255,255,255),-1)
        cv2.putText(img,"no of face detected: " +str(len(faces)),(0,25),font,0.5,(255,0,0),1)
        cv2.putText(img," Wearing MASK : " +str(b),(0,40),font,0.5,(0,255,0),1)
        cv2.putText(img,"Not Wearing MASK: " +str(a),(0,55),font,0.5,(0,0,255),1)
        output=np.zeros((i,j,3),dtype="uint8")
        output[0:i,0:j]=img
        writer.write(output)
        cv2.imshow('hell0_world',output)
        if cv2.waitKey(lol) & 0xFF==27: break
    else: break
cv2.destroyAllWindows()
vs.release()
writer.release()


# In[ ]:




