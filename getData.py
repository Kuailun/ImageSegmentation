import os
import nibabel as nib
import scipy.misc
import numpy as np

def transfer(dir,save):
    dirs=os.listdir(dir)
    ret=[]
    for i in range(len(dirs)):
        dd=os.listdir(dir+"\\"+dirs[i])
        rett=[]
        for j in range(len(dd)):
            if(j!=1):
                if not (os.path.exists(save+"\\"+dirs[i]+"\\"+dd[j]+"\\")):
                    os.makedirs(save+"\\"+dirs[i]+"\\"+dd[j]+"\\")
                Convert(dir,save,"\\"+dirs[i]+"\\",dd[j])
                rett.append(dir+"\\"+dirs[i]+"\\"+dd[j]+"\\"+dd[j]+".jpg")
        ret.append(rett)
    return ret

def Convert(dir,save,path,name):
    niipath=dir+path+"\\"+name+"\\"+name+".nii"
    img = nib.load(niipath)
    width,height,layer=img.shape
    for i in range(layer):
        imgS=img.dataobj[:,:,i]+23
        img_max=np.amax(imgS)
        if(img_max!=0):
            imgS=(imgS/img_max*255)//1
        save_path=save+path+name+"\\"+name+"-"+str(i)+".jpg"
        scipy.misc.imsave(save_path,imgS)

competition_name=os.getcwd()+"\\data\\ISLES 2018\\TRAINING"
saveDir=os.getcwd()+"\\imageData\\"
ids_train=transfer(competition_name,saveDir)
