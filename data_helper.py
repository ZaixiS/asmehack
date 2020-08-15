from skimage import io
import pandas as pd
import numpy as np
import os
import random

def parse_layer(num,resize=0.2,crop_x=224,crop_y=224,cooling_alpha=0.001,clock_alpha=50,method='TRAIN',verbose=False,mode='both'):
    # create directory
    print('Layer: %d'%(num))
    layer='L%04d'%(num)
    layer_dir='./%s/%s'%(work,layer)
    train_dir='./%s/%s/train'%(work,layer)
    val_dir='./%s/%s/val'%(work,layer)
    if not os.path.exists(layer_dir): os.mkdir(layer_dir)
    if not os.path.exists(train_dir): os.mkdir(train_dir)
    if not os.path.exists(val_dir): os.mkdir(val_dir)

    if mode!='image':
        # read label
        pixel=[]
        category=[]
        mpm_dir='./%s/MPM/Layer%03dto%03d/%s/'%(home,(num-1)//50*50+1,(num-1)//50*50+50,layer)
        for temp in range(len(os.listdir(mpm_dir))):
            name='%sframe_%04d.bmp'%(mpm_dir,temp+1)
            pixel.append(np.sum(io.imread(name)>150))
        threshold=np.array([10,
                            np.quantile(pixel,.25),
                            np.quantile(pixel,.75)])
        category=[np.sum(temp>threshold) for temp in pixel]

    if mode!='label':
        # read cmd
        cmd=pd.read_csv('./%s/Command_Part1/XYPT_Part01_%s.csv'%(home,layer),header=None)
        left=cmd[0].min()
        right=cmd[0].max()
        down=cmd[1].min()
        up=cmd[1].max()
        mag=1000
        width=int((right-left)*mag*resize)+1
        height=int((up-down)*mag*resize)+1
        power=np.zeros([width,height])
        speed=np.zeros([width,height])
        temporal=np.zeros([width,height])

        current_order=0
        locations=[]
        first_row=cmd[0:1]
        history_x=[first_row[0][0]]*3
        history_y=[first_row[1][0]]*3
        speed_max=-10000
        speed_min=10000
        for line in cmd.iterrows():
            raw_x=line[1][0]
            raw_y=line[1][1]
            history_x.pop()
            history_x.insert(0,raw_x)
            history_y.pop()
            history_y.insert(0,raw_y)
            # calculate speed
            last_speed=((history_x[2]-history_x[0])*(history_x[2]-history_x[0])+(history_y[2]-history_y[0])*(history_y[2]-history_y[0]))**0.5/2e-5
            if current_order>0:
                speed[current_x,current_y]=last_speed
                speed_max=max(last_speed,speed_max)
                speed_min=min(last_speed,speed_min)
            current_order+=1
            current_x=int((raw_x-left)*mag*resize)
            current_y=int((raw_y-down)*mag*resize)
            power[current_x,current_y]=line[1][2]
            temporal[current_x,current_y]=current_order
            if line[1][3]==2: locations.append([current_x,current_y])
        power_min=np.min(power)
        power_max=np.max(power)

    if mode=='label':
        # label only
        num_train=int(0.8*len(pixel))
        train_pixel=pixel[:num_train]
        val_pixel=pixel[num_train:]
        train_category=category[:num_train]
        val_category=category[num_train:]
        pd.DataFrame({'image':np.arange(len(train_pixel)),'pixel':train_pixel,'category':train_category}).to_csv('./%s/%s/train.csv'%(work,layer),index=0)
        pd.DataFrame({'image':np.arange(len(val_pixel)),'pixel':val_pixel,'category':val_category}).to_csv('./%s/%s/val.csv'%(work,layer),index=0)
        print('Done')
        return
    # generate photos
    photo_id=0
    train_pixel=[]
    train_category=[]
    val_pixel=[]
    val_category=[]
    num_train=int(0.8*len(locations))
    for temp in locations:
        if verbose: print(photo_id,end='\t')
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
                original_speed=(speed[x,y]-speed_min)/(speed_max-speed_min)
                speed_crop[x+int(crop_x*0.5)-center_x,y+int(crop_y*0.5)-center_y]=int(original_speed*255)
                # temporal crop
                elapsed_time=temporal[center_x,center_y]-temporal[x,y]
                if method=='TRAIN':
                    temporal_value=int(np.exp(-elapsed_time*cooling_alpha)*255)
                else:
                    temporal_value=max(255-int(np.log(elapsed_time+1)*clock_alpha),0)
                temporal_crop[x+int(crop_x*0.5)-center_x,y+int(crop_y*0.5)-center_y]=temporal_value
        rgb_image=np.stack((power_crop.astype(np.uint8),speed_crop.astype(np.uint8),temporal_crop.astype(np.uint8)),axis=-1)
        if mode=='image':
            # generate images only (fixed train val split)
            if photo_id<num_train:
                io.imsave('./%s/%s/train/%d.png'%(work,layer,photo_id),rgb_image,check_contrast=False)
            else:
                io.imsave('./%s/%s/val/%d.png'%(work,layer,photo_id-num_train),rgb_image,check_contrast=False)
        else:
            if random.random()<0.8:
                # train
                io.imsave('./%s/%s/train/%d.png'%(work,layer,len(train_pixel)),rgb_image,check_contrast=False)
                train_pixel.append(pixel[photo_id])
                train_category.append(category[photo_id])
            else:
                # val
                io.imsave('./%s/%s/val/%d.png'%(work,layer,len(val_pixel)),rgb_image,check_contrast=False)
                val_pixel.append(pixel[photo_id])
                val_category.append(category[photo_id])
        photo_id+=1
    if mode=='both':
        pd.DataFrame({'image':np.arange(len(train_pixel)),'pixel':train_pixel,'category':train_category}).to_csv('./%s/%s/train.csv'%(work,layer),index=0)
        pd.DataFrame({'image':np.arange(len(val_pixel)),'pixel':val_pixel,'category':val_category}).to_csv('./%s/%s/val.csv'%(work,layer),index=0)
    print('Done')

home='problem2'
work='layer_data'
if not os.path.exists('./%s'%(work)): os.mkdir('./%s'%(work))
# parse_layer(1,mode='image')
# for layer in range(1,151):
    # parse_layer(layer,mode='image')