import sys
import os
import numpy as np
import copy
import cv2

# install: pip install --upgrade arabic-reshaper
import arabic_reshaper

# install: pip install python-bidi
from bidi.algorithm import get_display

# install: pip install Pillow
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import tensorflow as tf 
model = tf.keras.models.load_model('models/asl_model.h5', compile=False)

def process_image(img):
    img = cv2.resize(img, (64, 64))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, 64 , 64 , 3))
    img = img.astype('float32') / 255.
    return img

cap = cv2.VideoCapture(0)

frame_counter = 0

sequence = ''
fontFile = "fonts/Sahel.ttf"
font = ImageFont.truetype(fontFile, 70)
categories=[
["ain",'ع'],
["al","ال"],
["aleff",'أ'],
["bb",'ب'],
["dal",'د'],
["dha",'ط'],
["dhad","ض"],
["fa","ف"],
["gaaf",'جف'],
["ghain",'غ'],
["ha",'ه'],
["haa",'ه'],
["jeem",'ج'],
["kaaf",'ك'],
["la",'لا'],
["laam",'ل'],
["meem",'م'],
["nun","ن"],
["ra",'ر'],
["saad",'ص'],
["seen",'س'],
["sheen","ش"],
["ta",'ت'],
["taa",'ط'],
["thaa","ث"],
["thal","ذ"],
["toot",' ت'],
["waw",'و'],
["ya","ى"],
["yaa","ي"],
["zay",'ز']]
while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    
    if ret:
        x1, y1, x2, y2 = 150, 150, 400, 400
        img_cropped = img[y1:y2, x1:x2]

        image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
        
        a = cv2.waitKey(1)
        if frame_counter % 5 == 0:
            score = 0
            res = ''
            try:
                proba = model.predict(process_image(img_cropped))[0]
                mx = np.argmax(proba)

                score = proba[mx] * 100
                res = categories[mx][0]
                sequence = categories[mx][1]
            except:
                continue 

        reshaped_text = arabic_reshaper.reshape(sequence)   
        bidi_text = get_display(reshaped_text)    

        frame_counter += 1
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 300), bidi_text, (0,0,0), font=font)
        img = np.array(img_pil)
        #cv2.putText(img, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
        cv2.putText(img, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.imshow("Simple Test", img)

        
        if a == 27: # when `esc` is pressed
            break

# Following line should... <-- This should work fine now
cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()
