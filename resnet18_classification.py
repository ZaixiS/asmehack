from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from skimage import io, transform

import copy
import os.path
import glob
from torch.utils.data import Dataset, DataLoader
import pandas as pd
plt.ion()   # interactive mode
from matplotlib.pyplot import imshow

 
# Load Data
# ---------
# Load in pre-processed images and labels in training set and validation set. The pre-processed image contains the information of command line, and the lables are the melt pool types (zero, small, normal and large).


# Writing a custermized dataset for my data
class CommendDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir , label_dir  , phase, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
#         self.num_image = glob.glob(os.path.join(root_dir,'*.png'))
        # if phase == 'train':
        #     self.num_image = 80000
        # else:
        #     self.num_image = 20000
        self.df_label = pd.read_csv(os.path.join(label_dir, phase + '.csv'))
        self.num_image=self.df_label.shape[0]
        self.phase = phase

    def __len__(self):
#         return len(self.num_image)
        return self.num_image

    def __getitem__(self, idx):
        lab = self.df_label.iloc[idx]['category']
        # if self.phase == 'val':
        #     idx = idx+1999
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                str(idx)+'.png')
        # image = io.imread(img_name)
        # image = image[20:40,20:40,:].transpose((2, 0, 1))
        
        image = io.imread(img_name).transpose((2, 0, 1))

        if self.transform:
            image = self.transform(image)
        
        return image,lab



# Data augmentation and normalization for training
data_transforms = transforms.Compose([
        transforms.CenterCrop(40)
    ])
data_dir = './layers/'
label_dir = './layers/'
image_datasets = {x: CommendDataset(root_dir = os.path.join(data_dir, x),label_dir =label_dir,transform=None ,phase = x) for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100,
                                             shuffle=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
# Training the model
# ------------------
# 
 
# ### Define the classification model


def train_model_class(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            epo_count = 0
            for inputs, labels in dataloaders[phase]:
                epo_count += 1
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.float32)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    labels = labels.long()
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)    
#                 epoch_loss += loss * inputs.size(0)
#                 if (epo_count % 10) == 0:
#                     num_sample = inputs.size(0)*epo_count
#                     cur_RMSE = epoch_loss/num_sample
#                     cur_RMSE = cur_RMSE.sqrt()
#                     print('{} number of samples: {} RMSE: {:.4f}'.format(phase, num_sample, cur_RMSE))
#                     if phase == 'val' and cur_RMSE < best_acc:
#                         best_acc = cur_RMSE
#                         best_model_wts = copy.deepcopy(model.state_dict()) 
#                         print('Best performace, saving model...')
#                         torch.save(best_model_wts,'./state/model.p')
#                 # statistics
#                 # running_loss += loss.item() * inputs.size(0)
#                 # running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'val' and epoch_acc < best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts,'./state/classifaction.p')
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

 
# ### Define Visualizing functions


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format('name'))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

 
# ### Define the CNN model


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        # our dimension [3,40,40]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 20, 20]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 10, 10]
            nn.ZeroPad2d(1), # [64,12,12]
    

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 12, 12]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 6, 6]
            nn.ZeroPad2d(1), # [128,8,8]
    

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 4, 4]
 

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 2, 2]
            
#             nn.Conv2d(512, 512, 3, 1, 1), # [512, 4, 4]
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2, 0),       # [512, 2, 2]
        )
        self.cnn = nn.DataParallel(self.cnn)
        self.fc = nn.Sequential(
            nn.Linear(512*2*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

 
# Running the model
# ----------------------
# 
# 
# 


num_ca = 4
model_use = 'resnet18'

if model_use == 'resnet18': 
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs,num_ca)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

elif model_use == 'alexnet': 
    model_ft = models.alexnet(pretrained=True)
    #num_ftrs = model_ft.classifier[-1].in_features
    #model_ft.classifier[-1] = nn.Linear(num_ftrs,num_ca)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    
elif model_use == 'diy':
    model_ft = Classifier()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-5, momentum=0.9)
    

model_ft = model_ft.to(device)


criterion = nn.CrossEntropyLoss()
 

# Observe that all parameters are being optimized

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

 
# ### Running Model


model_ft = train_model_class(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                             num_epochs=20)



# visualize_model(model_ft)

 
# ConvNet as fixed feature extractor
# ----------------------------------
# 
# Here, we need to freeze all the network except the final layer. We need
# to set ``requires_grad == False`` to freeze the parameters so that the
# gradients are not computed in ``backward()``.
# 
# You can read more about this in the documentation
# `here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
# 
# 
# 


model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 1)

model_conv = model_conv.to(device)

criterion = nn.MSELoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

 
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
# 
# On CPU this will take about half the time compared to previous scenario.
# This is expected as gradients don't need to be computed for most of the
# network. However, forward does need to be computed.
# 
# 
# 


model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=50)



visualize_model(model_conv)

plt.ioff()
plt.show()

 
# Further Learning
# -----------------
# 
# If you would like to learn more about the applications of transfer learning,
# checkout our `Quantized Transfer Learning for Computer Vision Tutorial <https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html>`_.
# 
# 
# 
# 

