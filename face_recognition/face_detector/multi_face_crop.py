import sys
import os
import dlib
import glob
import cv2
import re
import numpy as np
from PIL import Image
import time
import multiprocessing
from multiprocessing import Process

core=multiprocessing.cpu_count()

rotation_angle=[-45,45]

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
file_list=glob.glob(os.path.join(faces_folder_path, "*.jpg"))
file_count=len(file_list)//core+1
def face_crop(file_list):
    for f in file_list:
        print("Processing file: {}".format(f))
        img = cv2.imread(f)
        dets = detector(img, 1)
        if len(dets)==0:
            for angle in rotation_angle:
                pad_image_size=2000
                size=img.shape
                pad_center=[img.shape[1]//2,img.shape[0]//2]
                xs=np.clip(np.arange(pad_image_size**2,dtype=np.int32)%pad_image_size-pad_image_size//2+pad_center[0],0,size[1]-1)
                ys=np.clip(np.arange(pad_image_size**2,dtype=np.int32)//pad_image_size-pad_image_size//2+pad_center[1],0,size[0]-1)
                img2=img[ys,xs,:].copy().reshape(pad_image_size,pad_image_size,3)
                img2=np.asarray(Image.fromarray(img2).rotate(angle)).astype("uint8")
                dets = detector(img2, 1)
                if len(dets)>0:
                    img=img2
                    break
        if len(dets)>0:
            print("Number of faces detected: {}".format(len(dets)))
            for k, d in enumerate(dets):

                shape = predictor(img, d)
                # plot face features
                """
                for i in range(0,68):
                center=(shape.part(i).x,shape.part(i).y)
                cv2.circle(img,center,1,(0,0,255),-1)
                """
                bottom=np.array((shape.part(8).x,shape.part(8).y))
                right=np.array((shape.part(16).x,shape.part(16).y))
                left=np.array((shape.part(0).x,shape.part(0).y))
                vart=np.array([-right[1]+left[1],right[0]-left[0]])*np.cross(right-left,bottom-left)/np.linalg.norm(right-left)**2
                ld=left+vart
                rd=ld+right-left
                ld_side=bottom+vart*0.1+(ld-bottom)*1.4
                rd_side=bottom+vart*0.1+(rd-bottom)*1.4
                base=np.linalg.norm(ld_side-rd_side)
                ru_side=((ld_side-rd_side)/base)[::-1]*np.array([-1,1])*base+rd_side
                lu_side=ru_side+ld_side-rd_side
                center=(ld_side+ru_side)/2
                theta=180*np.arcsin(np.cross((rd_side-ld_side)/base,np.array([1,0])))/np.pi
                im_size = int(max(rd_side[1],ld_side[1]))-int(min(ru_side[1],lu_side[1]))
                xs = np.clip(np.arange(im_size**2,dtype=np.int64)%im_size+int(min(lu_side[0],ld_side[0])),0,img.shape[1]-1)
                ys = np.clip(np.arange(im_size**2,dtype=np.int64)//im_size+int(min(ru_side[1],lu_side[1])),0,img.shape[0]-1)
                img2 = img[ys,xs,:].copy().reshape(im_size,im_size,3)
                PIL_im=Image.fromarray(img2[:,:,::-1])
                PIL_im=PIL_im.rotate(-theta)
                _,length=PIL_im.size
                base = int(base)
                PIL_im=PIL_im.crop((length//2-base//2,length//2-base//2,length//2+base//2,length//2+base//2))
                dir_path="detected/"
                if os.path.isdir(dir_path)==False:
                    os.mkdir(dir_path)

                PIL_im.save(dir_path+re.sub(".jpg","",f.split('/')[-1])+str(k)+".jpg")
            
            
                # draw face detection
                #cv2.line(img,tuple(rd_side.astype("int64")),tuple(ld_side.astype("int64")),(0,0,255),2)
                #cv2.line(img,tuple(ld_side.astype("int64")),tuple(lu_side.astype("int64")),(0,0,255),2)
                #cv2.line(img,tuple(lu_side.astype("int64")),tuple(ru_side.astype("int64")),(0,0,255),2)
                #cv2.line(img,tuple(ru_side.astype("int64")),tuple(rd_side.astype("int64")),(0,0,255),2)
            
                #point face parts
                
                """
                for i in range(42,48):
                center=(shape.part(i).x,shape.part(i).y)
                cv2.circle(img,center,1,(0,0,255),-1)
                """
    #save image
    #cv2.imwrite("detected/"+re.sub(".jpg","",f)+"_detected.jpg",img)
jobs=[]
for i in range(core):
    job=Process(target=face_crop,args=(file_list[i*file_count:(i+1)*file_count],)) 
    jobs.append(job)
    job.start()

[job.join() for job in jobs]
