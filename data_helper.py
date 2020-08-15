from skimage import io
import pandas as pd
import numpy as np
import os
import random

# data folder
data_dir='layer13'
if not os.path.exists('./%s'%(data_dir)): os.mkdir('./%s'%(data_dir))
if not os.path.exists('./%s/train'%(data_dir)): os.mkdir('./%s/train'%(data_dir))
if not os.path.exists('./%s/val'%(data_dir)): os.mkdir('./%s/val'%(data_dir))
# resize factor [0,1]
resize=0.2

# read label
y_dir='MPM_Layer0013'
labels=[]
for temp in range(len(os.listdir('./Hackathon_3dBuild/%s/'%(y_dir)))):
    name='./Hackathon_3dBuild/%s/frame%05d.bmp'%(y_dir,temp+1)
    labels.append(np.sum(io.imread(name)>150))

# read command
x_dir='CMD_Layer0013'
data_df=pd.read_csv('./Hackathon_3dBuild/BuildCommand/%s.csv'%(x_dir),header=None)
left=data_df[0].min()
right=data_df[0].max()
down=data_df[1].min()
up=data_df[1].max()
mag=1000
width=int((right-left)*mag*resize)+1
height=int((up-down)*mag*resize)+1
power=np.zeros([width,height])
speed=np.zeros([width,height])
temporal=np.zeros([width,height])

current_order=0
locations=[]
for line in data_df.iterrows():
    current_order+=1
    current_x=int((line[1][0]-left)*mag*resize)
    current_y=int((line[1][1]-down)*mag*resize)
    power[current_x,current_y]=line[1][2]
    speed[current_x,current_y]=line[1][3]
    temporal[current_x,current_y]=current_order
    if line[1][4]==2: locations.append([current_x,current_y])

power_min=np.min(power)
power_max=np.max(power)
speed_min=np.min(speed)
speed_max=np.max(speed)
crop_x=224
crop_y=224

# exponential scale for training
cooling_alpha=0.01
# log scale for visualization
clock_alpha=50
# use for training
method='TRAIN'
# use for visualization
# method='VIS'

photo_id=0
train_label=[]
val_label=[]
for temp in locations:
    print(photo_id,end='\t')
    center_x,center_y=temp
    power_crop=np.zeros([crop_x,crop_y])
    speed_crop=np.zeros([crop_x,crop_y])
    temporal_crop=np.zeros([crop_x,crop_y])
    for x in range(center_x-int(crop_x*0.5),center_x+int(crop_x*0.5)):
        for y in range(center_y-int(crop_y*0.5),center_y+int(crop_y*0.5)):
            # consider locations in the window with time stamp earlier than center
            if x<0 or x>=width: continue
            if y<0 or y>=height: continue
            if temporal[x,y]==0 or temporal[x,y]>temporal[center_x,center_y]: continue
            # power crop
            original_power=(power[x,y]-power_min)/(power_max-power_min)
            power_crop[x+int(crop_x*0.5)-center_x,y+int(crop_y*0.5)-center_y]=int(original_power*255)
            # speed crop
            original_speed=(speed[x,y]-power_min)/(speed_max-speed_min)
            speed_crop[x+int(crop_x*0.5)-center_x,y+int(crop_y*0.5)-center_y]=int(original_speed*255)
            # temporal crop
            elapsed_time=temporal[center_x,center_y]-temporal[x,y]
            if method=='TRAIN':
                temporal_value=int(np.exp(-elapsed_time*cooling_alpha)*255)
            else:
                temporal_value=max(255-int(np.log(elapsed_time+1)*clock_alpha),0)
                temporal_crop[x+int(crop_x*0.5)-center_x,y+int(crop_y*0.5)-center_y]=temporal_value
    rgb_image=np.stack((power_crop.astype(np.uint8),speed_crop.astype(np.uint8),temporal_crop.astype(np.uint8)),axis=-1)
    if random.random()<0.8:
        # train
        io.imsave('./%s/train/%d.png'%(data_dir,len(train_label)),rgb_image,check_contrast=False)
        train_label.append(labels[photo_id])
    else:
        # val
        io.imsave('./%s/val/%d.png'%(data_dir,len(val_label)),rgb_image,check_contrast=False)
        val_label.append(labels[photo_id])
    photo_id+=1
pd.DataFrame({'image':np.arange(len(train_label)),'label':train_label}).to_csv('./%s/train.csv'%(data_dir),index=0)
pd.DataFrame({'image':np.arange(len(val_label)),'label':val_label}).to_csv('./%s/val.csv'%(data_dir),index=0)