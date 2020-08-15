from skimage import io
import pandas as pd
import numpy as np
import os
# target csv
target_name='CMD_Layer0013'
if not os.path.exists('./%s'%(target_name)): os.mkdir('./%s'%(target_name))
# resize factor [0,1]
resize=0.3

df=pd.read_csv('./Hackathon_3dBuild/BuildCommand/%s.csv'%(target_name),header=None)
left=df[0].min()
right=df[0].max()
down=df[1].min()
up=df[1].max()
mag=1000
width=int((right-left)*mag*resize)+1
height=int((up-down)*mag*resize)+1
power=np.zeros([width,height])
speed=np.zeros([width,height])
temporal=np.zeros([width,height])

current_order=0
locations=[]
for line in df.iterrows():
    current_order+=1
    current_x=int((line[1][0]-left)*mag*resize)
    current_y=int((line[1][1]-down)*mag*resize)
    power[current_x,current_y]=line[1][2]
    speed[current_x,current_y]=line[1][3]
    temporal[current_x,current_y]=current_order
    if line[1][4]==2: locations.append([current_x,current_y])
power_min=np.min(power)
power_max=np.max(power)
speed_min=np.min(power)
speed_max=np.max(power)
crop_x=224
crop_y=224
cooling_factor=0
clock_scale=0.001

photo_id=0
for temp in locations:
    photo_id+=1
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
            elapsed_time=temporal[x,y]-temporal[center_x,center_y]
            original_power=(power[x,y]-power_min)/(power_max-power_min)
            cooled_power=original_power*np.exp(elapsed_time*cooling_factor)
            power_crop[x+int(crop_x*0.5)-center_x,y+int(crop_y*0.5)-center_y]=int(cooled_power*255)
            # speed crop
            original_speed=(speed[x,y]-power_min)/(speed_max-speed_min)
            speed_crop[x+int(crop_x*0.5)-center_x,y+int(crop_y*0.5)-center_y]=int(original_speed*255)
            # temporal crop
            temporal_crop[x+int(crop_x*0.5)-center_x,y+int(crop_y*0.5)-center_y]=max(255+int(elapsed_time*255*clock_scale),0)
    io.imsave('./%s/%d.png'%(target_name,photo_id),
              np.stack((power_crop.astype(np.uint8),
                        speed_crop.astype(np.uint8),
                        temporal_crop.astype(np.uint8)),
                        axis=-1),
              check_contrast=False)
    # break