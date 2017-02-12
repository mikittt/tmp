import os
import numpy as np
import glob
import shutil

dir_list=np.arange(100,203)

for i in dir_list:
    file_list=glob.glob("*/"+str(i)+"*.jpg")
    if os.path.isdir(str(i))==False:
        os.mkdir(str(i))
    for part in file_list:
        shutil.move(part,str(i)+"/")
