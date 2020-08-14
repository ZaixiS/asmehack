from skimage import io
import pandas as pd
import numpy as np
df=pd.read_csv('./Hackathon_3dBuild/BuildCommand/CMD_Layer0013.csv',header=None)
left=df[0].min()
right=df[0].max()
down=df[1].min()
up=df[1].max()
mag=1000
width=int((right-left)*mag)+1
height=int((up-down)*mag)+1
thermal=np.zeros([width,height])
order=np.zeros([width,height])

current_order=0
photos=[]
for line in df.iterrows():
    current_order+=1
    current_x=int((line[1][0]-left)*mag)
    current_y=int((line[1][1]-down)*mag)
    if line[1][4]==2: photos.append([current_x,current_y])
    thermal[current_x,current_y]=line[1][2]
    order[current_x,current_y]=current_order

crop_x=224
crop_y=224
thermal_min=np.min(thermal)
thermal_max=np.max(thermal)
cooling_factor=0
photo_id=0
for temp in photos[3:]:
    photo_id+=1
    print(photo_id)
    center_x,center_y=temp
    crop=np.zeros([crop_x,crop_y])
    for x in range(center_x-int(crop_x*0.5),center_x+int(crop_x*0.5)):
        for y in range(center_y-int(crop_y*0.5),center_y+int(crop_y*0.5)):
            # in scope and time order earlier than center
            if x<0 or x>=width: continue
            if y<0 or y>=height: continue
            if order[x,y]==0 or order[x,y]>order[center_x,center_y]: continue
            elapsed_time=order[x,y]-order[center_x,center_y]
            original_thermal=(thermal[x,y]-thermal_min)/(thermal_max-thermal_min)
            cooled_thermal=original_thermal*np.exp(elapsed_time*cooling_factor)
            crop[x+int(crop_x*0.5)-center_x,y+int(crop_y*0.5)-center_y]=int(cooled_thermal*255)
    io.imsave('./photo_crop/%d.png'%(photo_id),crop.astype(np.uint8))