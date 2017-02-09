# -*- coding: utf-8 -*-
import picamera
import picamera.array
import cv2
import sys
import os
from datetime import datetime
name=sys.argv[1]
cascade_path="/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml"

with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as stream:
        camera.resolution=(320,240)
        #camera.resolution=(50,50)

        while True:
            camera.capture(stream,'bgr',use_video_port=True)
            cascade=cv2.CascadeClassifier(cascade_path)
            gray=cv2.cvtColor(stream.array,cv2.COLOR_BGR2GRAY)
            facerect=cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=2,minSize=(30,30),maxSize=(150,150))
            """
            if len(facerect)>0:
                for rect in facerect:
                    cv2.rectangle(stream.array,tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]),(0,0,255),thickness=2)
            """
            #cv2.imshow('frame',stream.array)
            
            if len(facerect)>0:
                dir_path="/home/mikihiro/Desktop/openCV/book/createData/"+name
                if os.path.isdir(dir_path)==False:
                    os.mkdir(dir_path)
                new_image_path=dir_path+"/"+name+datetime.now().strftime("%Y%m%d-%H%M%S")+".jpg"
                cv2.imwrite(new_image_path,stream.array)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            stream.seek(0)
            stream.truncate()

        cv2.destroyAllWindows()
