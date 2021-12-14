import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl

X, y = fetch_openml('mnist_784', version = 1 , return_X_y=True)
print(pd.Series(y).value_counts())
classes = ['0','1','2','3','4','5','6','7','8','9']
nclasses = len(classes)

#Setting an HTTPS Context to fetch data from OpenML
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

x_train , x_test , y_train  , y_test = train_test_split(X,y , random_state = 9 , train_size=7500 , test_size = 2500)
x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0
clf = LogisticRegression(solver='saga' , multi_class="multinomial").fit(x_train_scaled  , y_train)

ypred = clf.predict(x_test_scaled)
a = accuracy_score(y_test , ypred)

print("The accuracy is : ", a)

cap = cv2.VideoCapture(0)
while(True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape()
        upperleft = ( int(width/2 - 56) , int(height/2 - 56) )
        bottomright = ( int(width/2 + 56) , int(height/2 + 56) ) 
        cv2.rectangle(gray , upperleft , bottomright , (255,0,0) , 2 )

        #roi = Region Of Interest/focus area
        roi = gray[upperleft [1]: bottomright[1] , upperleft[0]: bottomright[0] ]
        
        #Converting cv2 image to pil format so that the interpreter understands
        impil = Image.fromarray(roi )
        imgbw = impil.convert("L")
        imgbw_resize  = imgbw.resize((28,28) , Image.ANTIALIAS )
        # inverting the image
        img_invert = PIL.ImageOps.invert(imgbw_resize)
        pixel_filter = 20 
        #converting to scalar quantity
        min_pixel = np.percentile(img_invert , pixel_filter  )
        #using clip to limit the values between 0,255
        img_scaled = np.clip(img_invert - min_pixel , 0,255)
        max_pixel = np.max(img_invert)

        #converting into an array
        imgScaled = np.asarray(img_scaled)/max_pixel
        
        #creating a test sample and making a prediction
        test_sample = np.array(imgScaled).reshape(1,784)
        test_pred = clf.predict(test_sample)
        print("predicted class is : ", test_pred )
 
        # Display the resulting frame
        cv2.imshow("frame" , gray)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break

    except Exception as e :
        pass

cap.release()
cv2.destroyAllWindows()