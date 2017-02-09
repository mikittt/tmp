import cv2
import sys
import os
import re
import sys
name=sys.argv[1]
cascade_path="/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml"
cascade=cv2.CascadeClassifier(cascade_path)
image_directory=os.listdir("./"+name)
for images in image_directory:
    image_path="./"+name+"/"+images
    image=cv2.imread(image_path)
    if (image is None):
        print("cannot open "+image_path+"\n\n\n")
        continue
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    facerect=cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=2,minSize=(30,30),maxSize=(150,150))

    new_dir_path="./"+name+"_face"
    if os.path.isdir(new_dir_path)==False:
        os.mkdir(new_dir_path)
    new_image_path=new_dir_path+"/"+re.sub(".jpg","",images)
    for i,rect in enumerate(facerect):
        x=rect[0]
        y=rect[1]
        width=rect[2]
        height=rect[3]
        dst=image[y:y+height,x:x+width]
        dst_path=new_image_path+"_"+str(i)+".jpg"
        cv2.imwrite(dst_path,dst)
