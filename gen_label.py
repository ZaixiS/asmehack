from skimage import io
import numpy as np
import os
import pandas as pd

dir='./Hackathon_3dBuild/MPM_Layer0013/'
images=[]
labels=[]
for temp in range(len(os.listdir(dir))):
    name='%sframe%05d.bmp'%(dir,temp+1)
    images.append('frame%05d'%(temp+1))
    labels.append(np.sum(io.imread(name)>150))
df=pd.DataFrame({'image':images,'label':labels})
df.to_csv('MPM_Layer0013.csv',index=0)