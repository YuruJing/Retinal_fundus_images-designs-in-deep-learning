#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install torchmetrics # add that into docker 


# In[1]:


## import libraries
import torchmetrics as tm
import numpy as np
import pandas as pd
import torch
import torch.hub
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms

## import auxilliary functions and models
from Extract_data_utils_js import extract_images, get_mean_sd, train_test_split, Custom_Dataset,googlenet_train_valid, train_valid 
import inception


# # Load and Combine the dataset

# In[2]:


ukbb_images = '/data/alzeye/ukbb/crop_50/'
ukbb_csv    = '/data/alzeye/ukbb/MERGED_UKBB_FUNDUS_final.csv'


# In[61]:


# read data.csv
df = pd.read_csv(ukbb_csv)
important_columns = ["id", "laterality", "file_name",
                     "first_value|assess_age","all_values|gender_f0_m1", "spherical_equivalent_L", "spherical_equivalent_R"]


# In[60]:


df['all_values|gender_f0_m1'].isnull().any().sum()


# In[57]:


df.info(verbose=True)


# In[3]:


get_ipython().system('nvidia-smi')


# In[85]:


# left_eye 
clean_left_df = df.dropna(subset = ['spherical_equivalent_L'])
#left_eye = extract_images(clean_left_df, important_columns, 'L', ukbb_images)
# read left_eye_images
left_eye_targets = list()
left_gender = list()

with open(r'L_crop_50.txt', 'r') as f:
    left_eye = f.readlines()
    left_eye = [line[:-1] for line in left_eye]
    for sample_path in tqdm(left_eye):
        left_eye_targets.append(clean_left_df[clean_left_df['file_name']==sample_path].iloc[0]['spherical_equivalent_L'])
        #left_gender.append(clean_left_df[clean_left_df['file_name']==sample_path].iloc[0]['all_values|gender_f0_m1'])
print(len(left_eye))
print(len(left_eye_targets))
#print(len(left_gender))                                                                        


# In[86]:


# right_eye 
clean_right_df = df.dropna(subset = ['spherical_equivalent_R'])
#right_eye = extract_images(clean_right_df, important_columns, 'R', ukbb_images)
# read right_eye_images
right_eye_targets = list()
right_gender = list()
with open(r'R_crop_50.txt', 'r') as f:
    right_eye = f.readlines()
    right_eye = [line[:-1] for line in right_eye]

for sample_path in tqdm(right_eye):
    right_eye_targets.append(clean_right_df[clean_right_df['file_name']==sample_path].iloc[0]['spherical_equivalent_R'])
    #right_gender.append(clean_right_df[clean_right_df['file_name']==sample_path].iloc[0]['all_values|gender_f0_m1'])
    # right_eye_targets.append(self.data[self.data['file_name'] == sample_path][self.target_SE].to_numpy())
print(len(right_eye))
print(len(right_eye_targets))
#print(len(right_gender)) 


# In[87]:


len(left_eye)


# In[88]:


merge_df1 = {'file_name':left_eye, 'gender':left_gender, 'spherical_equivalent': left_eye_targets}
merge_df1 = pd.DataFrame(merge_df1)
merge_df1


# In[89]:


merge_df2 = {'file_name':right_eye, 'gender':right_gender, 'spherical_equivalent': right_eye_targets}
merge_df2 = pd.DataFrame(merge_df2)
merge_df2


# In[90]:


# conmbine the full eye merged data
full_df = pd.concat([merge_df1, merge_df2])
full_df


# In[5]:


# save
full_df.to_csv('all_crop50_eyes.csv', index=False)


# In[4]:


data = pd.read_csv('all_crop50_eyes.csv')
data


# In[93]:


data.info()


# In[8]:


data[data['gender']==0.0]


# In[9]:


data[data['gender']==1.0]


# In[5]:


full_eye = list(data['file_name'])
full_target = list(data['spherical_equivalent'])


# In[6]:


# without considering the gender
full_data = Custom_Dataset(ukbb_images, full_eye, full_target)


# In[8]:


# combine left and right eye data
left_data = Custom_Dataset(ukbb_images, left_eye, left_eye_targets)
right_data = Custom_Dataset(ukbb_images, right_eye, right_eye_targets)
full_data = torch.utils.data.ConcatDataset([left_data, right_data])


# In[7]:


# get the mean and std for batchsize 128
mean, std = get_mean_sd(full_data, 128) #only works on number of workers = 0
mean, std


# In[8]:


# normalize the images and train_test_split the data  # add data augmentation later
transf = transforms.Compose([ transforms.Resize(224),
                              #transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=mean, std=std)
                            ])
train_set, test_set = train_test_split(full_data, test_ratio = 0.1, train_trans=transf, test_trans=transf, workers=0)#full_data


# ## Google Net

# ### batch_size= 128, test_ratio=0.1, without data augmentation, learning rate 0.00001

# In[23]:


use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
model7 = inception.GoogLeNet(aux_logits=True, num_classes=1).to(device)
googlenet_train_valid(model7, device, 'MAE', train_set, test_set, 128 , 0.00001,100,128)


# ### batch_size= 128, test_ratio=0.1, without data augmentation, learning rate 0.0001

# In[17]:


use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
model1 = inception.GoogLeNet(aux_logits=True, num_classes=1).to(device)
googlenet_train_valid(model1, device, 'MAE', train_set, test_set, 128 , 0.0001,100,128)


# In[18]:


# save the model 
torch.save(model1.state_dict(), "/data/alzeye/yuru/GoogleNet_crop50_final128.pt")


# ### batch_size= 128, test_ratio=0.2, without data augmentation, learning rate 0.0001

# In[20]:


train_set, test_set = train_test_split(full_data, test_ratio = 0.2, trans=transf, workers=0)#full_data
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
model3 = inception.GoogLeNet(aux_logits=True, num_classes=1).to(device)
googlenet_train_valid(model3, device, 'MAE', train_set, test_set, 128 , 0.0001,100,128)


# ### batch_size= 128, test_ratio=0.1, without data augmentation, learning rate 0.001

# In[11]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
model10 = inception.GoogLeNet(aux_logits=True, num_classes=1).to(device)
googlenet_train_valid(model10, device, 'MAE', train_set, test_set, 128 , 0.001,100,128)


# In[12]:


# save the model 
torch.save(model10.state_dict(), "/data/alzeye/yuru/GoogleNet_crop50_final128v3.pt")


# ### batch_size= 128, test_ratio=0.1, without data augmentation, learning rate 0.01

# In[22]:


train_set, test_set = train_test_split(full_data, test_ratio = 0.1, trans=transf, workers=0)#full_data
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
model5 = inception.GoogLeNet(aux_logits=True, num_classes=1).to(device)
googlenet_train_valid(model5, device, 'MAE', train_set, test_set, 128 , 0.01,100,128)


# In[25]:


# save the model 
torch.save(model5.state_dict(), "/data/alzeye/yuru/GoogleNet_crop50_final128v2.pt")


# ### batch_size= 128, test_ratio=0.1, without data augmentation, learning rate 0.1

# In[24]:


use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
model8 = inception.GoogLeNet(aux_logits=True, num_classes=1).to(device)
googlenet_train_valid(model8, device, 'MAE', train_set, test_set, 128 , 0.1,100,128)


# ### batch_size= 64, test_ratio=0.1, without data augmentation, learning rate 0.0001

# In[13]:


# get the mean and std for batchsize 64
mean, std = get_mean_sd(full_data, 64) #only works on number of workers = 0
mean, std


# In[14]:


# normalize the images and train_test_split the data  # add data augmentation later
transf = transforms.Compose([ transforms.Resize(224),
                              #transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=mean, std=std)
                            ])
train_set, test_set = train_test_split(full_data, test_ratio = 0.1, trans=transf, workers=0)#full_data


# In[15]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
model9 = inception.GoogLeNet(aux_logits=True, num_classes=1).to(device)
googlenet_train_valid(model9, device, 'MAE', train_set, test_set, 64 , 0.0001,100,64)


# In[16]:


# save the model 
torch.save(model9.state_dict(), "/data/alzeye/yuru/GoogleNet_crop50_final64v1.pt")


# ### batch_size= 64, test_ratio=0.1, without data augmentation, learning rate 0.001

# In[15]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
model = inception.GoogLeNet(aux_logits=True, num_classes=1).to(device)
googlenet_train_valid(model, device, 'MAE', train_set, test_set, 64, 0.001,100,64)


# In[16]:


# save the model 
torch.save(model.state_dict(), "/data/alzeye/yuru/GoogleNet_crop50_final64v2.pt")


# ### batch_size= 32, test_ratio=0.1, without data augmentation, learning rate 0.001

# In[9]:


# get the mean and std for batchsize 32
mean, std = get_mean_sd(full_data, 32) #only works on number of workers = 0
mean, std


# In[10]:


# normalize the images and train_test_split the data  # add data augmentation later
transf = transforms.Compose([ transforms.Resize(224),
                              #transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=mean, std=std)
                            ])
train_set, test_set = train_test_split(full_data, test_ratio = 0.1, trans=transf, workers=0)#full_data


# In[11]:


use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
model1 = inception.GoogLeNet(aux_logits=True, num_classes=1).to(device)
googlenet_train_valid(model1, device, 'MAE', train_set, test_set, 32, 0.001,100,32)


# In[12]:


# save the model 
torch.save(model1.state_dict(), "/data/alzeye/yuru/GoogleNet_crop50_final32v1.pt")


# ### batch_size= 16, test_ratio=0.1, without data augmentation, learning rate 0.001

# In[8]:


# get the mean and std for batchsize 16
mean, std = get_mean_sd(full_data, 16) #only works on number of workers = 0
mean, std


# In[9]:


# normalize the images and train_test_split the data  # add data augmentation later
transf = transforms.Compose([ transforms.Resize(224),
                              #transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=mean, std=std)
                            ])
train_set, test_set = train_test_split(full_data, test_ratio = 0.1, trans=transf, workers=0)#full_data


# In[12]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
model2 = inception.GoogLeNet(aux_logits=True, num_classes=1).to(device)
googlenet_train_valid(model2, device, 'MAE', train_set, test_set, 16, 0.001,50,16)


# In[13]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
model2 = inception.GoogLeNet(aux_logits=True, num_classes=1).to(device)
googlenet_train_valid(model2, device, 'MAE', train_set, test_set, 16, 0.001,50,16)


# In[14]:


# save the model 
torch.save(model2.state_dict(), "/data/alzeye/yuru/GoogleNet_crop50_final16v1.pt")


# ## VGG-16, ResNet, Densenet, EfficientNet, Convnext, (Pre-trained False)

# In[9]:


from torchvision.models import resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, densenet161, DenseNet161_Weights, efficientnet_v2_l, EfficientNet_V2_L_Weights, convnext_large, ConvNeXt_Large_Weights, convnext_small, ConvNeXt_Small_Weights, vgg16_bn, VGG16_BN_Weights


# ### batch_size= 128, test_ratio=0.1, without data augmentation, learning rate 0.01

# In[10]:


# get the mean and std for batchsize 128
mean, std = get_mean_sd(full_data, 128) #only works on number of workers = 0
mean, std


# In[11]:


# normalize the images and train_test_split the data  # add data augmentation later
transf = transforms.Compose([ transforms.Resize(224),
                              #transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=mean, std=std)
                            ])
train_set, test_set = train_test_split(full_data, test_ratio = 0.1, trans=transf, workers=0)#full_data


# #### resnet50

# In[20]:


use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
resnet = resnet50(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, 1)
resnet = resnet.to(device)
train_valid(resnet, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[21]:


# save the model 
torch.save(resnet.state_dict(), "/data/alzeye/yuru/Resnet_crop50_128.pt")


# #### desnet161

# In[16]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
desnet = densenet161(weights=None)
desnet.classifier = nn.Linear(desnet.classifier.in_features, 1)
desnet = desnet.to(device)
train_valid(desnet, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[17]:


# save the model 
torch.save(desnet.state_dict(), "/data/alzeye/yuru/Densenet_crop50_128.pt")


# #### EffecientNet

# In[ ]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
effectnet = efficientnet_v2_l(weights=None)
effectnet.classifier[1] = nn.Linear(effectnet.classifier[1].in_features, 1)
effectnet = effectnet.to(device)
train_valid(effectnet, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[ ]:


# save the model 
torch.save(effectnet.state_dict(), "/data/alzeye/yuru/Effectnet_crop50_128.pt")


# #### ConvxNet_small

# In[ ]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet1 = convnext_small(weights=None)
convnet1.classifier[2] = nn.Linear(convnet1.classifier[2].in_features, 1)
convnet1 = convnet1.to(device)
train_valid(convnet1, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[ ]:





# #### VGG16

# In[ ]:





# In[ ]:





# #### MobileNet-SSD

# In[11]:


import Model
#import torch.nn.functional as f


# ### Learning rate 0.0001

# In[11]:


# learning rate 0.01 - range from 6-7 mse, mae 1.7
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
model1 = Model.MobileNetSSD(1).to(device)
train_valid(model1, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# ### Learning rate 0.00001

# In[12]:


## learning rate 0.000001 is too small (make mse 6-7, mae 1.7)
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
model3 = Model.MobileNetSSD(1).to(device)
train_valid(model3, device, 'MAE', train_set, test_set, 128, 0.00001,100,128)


# ## VGG-16, ResNet, Densenet, EfficientNet, Convnext, (Pre-trained True)

# #### resnet50 (pretrained)

# In[18]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
resnet1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
resnet1.fc = nn.Linear(resnet1.fc.in_features, 1)
resnet1 = resnet1.to(device)
train_valid(resnet1, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[19]:


# save the model 
torch.save(resnet1.state_dict(), "/data/alzeye/yuru/Resnet_trained_crop50_128.pt")


# ### Learning rate = 0.0001

# In[ ]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
resnet1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
resnet1.fc = nn.Linear(resnet1.fc.in_features, 1)
resnet1 = resnet1.to(device)
train_valid(resnet1, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[ ]:





# #### densese161 (pretrained)

# In[13]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
desnet1 = densenet161(weights=DenseNet161_Weights)
desnet1.classifier = nn.Linear(desnet1.classifier.in_features, 1)
desnet1 = desnet1.to(device)
train_valid(desnet1, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[ ]:


# save the model 
torch.save(desnet1.state_dict(), "/data/alzeye/yuru/Densenet_trained_crop50_128.pt")


# ### Learning rate 0.0001

# In[ ]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
desnet2 = densenet161(weights=DenseNet161_Weights)
desnet2.classifier = nn.Linear(desnet2.classifier.in_features, 1)
desnet2 = desnet2.to(device)
train_valid(desnet2, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# ### Learning rate 0.000001

# In[ ]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
desnet3 = densenet161(weights=DenseNet161_Weights)
desnet3.classifier = nn.Linear(desnet3.classifier.in_features, 1)
desnet3 = desnet3.to(device)
train_valid(desnet3, device, 'MAE', train_set, test_set, 128, 0.000001,100,128)


# #### EfficientNet_v2_l (pretrained)

# In[12]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
effectnet1 = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights)
effectnet1.classifier[1] = nn.Linear(effectnet1.classifier[1].in_features, 1)
effectnet1 = effectnet1.to(device)
train_valid(effectnet1, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[13]:


# save the model 
torch.save(effectnet1.state_dict(), "/data/alzeye/yuru/Effectnet_trained_crop50_128.pt")


# ### Learning rate = 0.0001

# In[16]:


use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
effectnet2 = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights)
effectnet2.classifier[1] = nn.Linear(effectnet2.classifier[1].in_features, 1)
effectnet2 = effectnet2.to(device)
train_valid(effectnet2, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# ### Learning rate = 0.000001

# In[17]:


use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
effectnet3 = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights)
effectnet3.classifier[1] = nn.Linear(effectnet2.classifier[1].in_features, 1)
effectnet3 = effectnet3.to(device)
train_valid(effectnet3, device, 'MAE', train_set, test_set, 128, 0.000001,100,128)


# #### convnext_small (pretrained)

# ### Learning rate =0.01

# In[12]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet1 = convnext_small(weights=ConvNeXt_Small_Weights)
convnet1.classifier[2] = nn.Linear(convnet1.classifier[2].in_features, 1)
convnet1 = convnet1.to(device)
train_valid(convnet1, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# ### Learning rate =0.0001

# In[17]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet2 = convnext_small(weights=ConvNeXt_Small_Weights)
convnet2.classifier[2] = nn.Linear(convnet2.classifier[2].in_features, 1)
convnet2 = convnet2.to(device)
train_valid(convnet2, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[13]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet2 = convnext_small(weights=ConvNeXt_Small_Weights)
convnet2.classifier[2] = nn.Linear(convnet2.classifier[2].in_features, 1)
convnet2 = convnet2.to(device)
train_valid(convnet2, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# ### Learning rate = 0.000001

# In[10]:


use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet3 = convnext_small(weights=ConvNeXt_Small_Weights)
convnet3.classifier[2] = nn.Linear(convnet3.classifier[2].in_features, 1)
convnet3 = convnet3.to(device)
train_valid(convnet3, device, 'MAE', train_set, test_set, 128, 0.000001,100,128)


# ### Learning rate = 0.00001

# In[11]:


use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet4 = convnext_small(weights=ConvNeXt_Small_Weights)
convnet4.classifier[2] = nn.Linear(convnet4.classifier[2].in_features, 1)
convnet4 = convnet4.to(device)
train_valid(convnet4, device, 'MAE', train_set, test_set, 128, 0.00001,100,128)


# In[14]:


# save the model 
torch.save(convnet2.state_dict(), "/data/alzeye/yuru/Convnet_trained_crop50_128.pt")


# #### ConvNext Large (pretrained)

# In[124]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet4 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet4.classifier[2] = nn.Linear(convnet4.classifier[2].in_features, 1)
convnet4 = convnet4.to(device)
train_valid(convnet4, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# ### Test different last layers structure

# In[11]:


# 2 fc layers
import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet.classifier[2] =nn.Sequential(OrderedDict([('fc1',nn.Linear(1536, 200)),
                                                 ('fc2', nn.Linear(200,1))]))
convnet = convnet.to(device)
train_valid(convnet, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[12]:


# 3 fc layers
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet1 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet1.classifier[2] =nn.Sequential(OrderedDict([('fc1',nn.Linear(1536, 500)),
                                                 ('fc2', nn.Linear(500, 200)),
                                                 ('fc3', nn.Linear(200, 1))]))
convnet1 = convnet1.to(device)
train_valid(convnet1, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[16]:


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


# In[17]:


# conv + 1 fc
import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet2 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet2.classifier[1] = nn.Sequential(OrderedDict([('conv',nn.Conv2d(1536, 1536, kernel_size=(7, 7),stride=(1, 1), padding=(3, 3), groups=1536))]))#nn.Linear(196608, 1)
convnet2.classifier[2] =Identity()#nn.Sequential(OrderedDict([('fc',nn.Linear(1,1))]))
convnet2 = convnet2.to(device)
train_valid(convnet2, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# ### test multiple attention layers 

# In[53]:


# combine attention + contencate features + dropout 
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet13 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet13.features[1][1].block[5] = nn.Sequential(OrderedDict([('se', SELayer(12)),
                                                  ('linear', nn.Linear(768, 192))]))
# convnet13.features[3][0].block[5] = nn.Sequential(OrderedDict([('se', SELayer(6)),
#                                                   ('linear', nn.Linear(1536, 384))]))
convnet13.features[3][2].block[5] = nn.Sequential(OrderedDict([('se', SELayer(6)),
                                                  ('linear', nn.Linear(1536, 384))]))

# convnet13.features[5][1].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet13.features[5][3].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
convnet13.features[5][5].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][7].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet13.features[5][9].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][11].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet13.features[5][13].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
convnet13.features[5][15].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][17].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet13.features[5][19].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
convnet13.features[5][21].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][23].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet13.features[5][25].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[7][0].block[5] = nn.Sequential(OrderedDict([('se', SELayer(6)),
#                                                   ('linear', nn.Linear(6144, 1536))]))
convnet13.classifier[1] = nn.Sequential(OrderedDict([
                                          ('se', SELayer(1)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet13.classifier[2] = nn.Sequential(OrderedDict([
                                          ('fc1', nn.Linear(1536,500)),
                                          ('fc2',nn.Linear(500,200)),
                                           ('fc3', nn.Linear(200,1))]))
convnet13 = convnet13.to(device)
train_valid(convnet13, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[62]:


# Multiple se layer + 3 fc
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet13 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet13.features[1][1].block[5] = nn.Sequential(OrderedDict([('se', SELayer(12)),
                                                  ('linear', nn.Linear(768, 192))]))
# convnet13.features[3][0].block[5] = nn.Sequential(OrderedDict([('se', SELayer(6)),
#                                                   ('linear', nn.Linear(1536, 384))]))
convnet13.features[3][2].block[5] = nn.Sequential(OrderedDict([('se', SELayer(6)),
                                                  ('linear', nn.Linear(1536, 384))]))

# convnet13.features[5][1].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet13.features[5][3].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
convnet13.features[5][5].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][7].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet13.features[5][9].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][11].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet13.features[5][13].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
convnet13.features[5][15].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][17].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet13.features[5][19].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
convnet13.features[5][21].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][23].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet13.features[5][25].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[7][0].block[5] = nn.Sequential(OrderedDict([('se', SELayer(6)),
#                                                   ('linear', nn.Linear(6144, 1536))]))
# convnet13.classifier[1] = nn.Sequential(OrderedDict([
#                                           ('se', SELayer(1536)),
#                                           ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet13.classifier[2] = nn.Sequential(OrderedDict([
                                          ('fc1', nn.Linear(1536,500)),
                                          ('fc2',nn.Linear(500,200)),
                                           ('fc3', nn.Linear(200,1))]))
convnet13 = convnet13.to(device)
train_valid(convnet13, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[63]:


# multiple bceblock + 3 fc layers
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet1 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet1.features[1][1].block[5] = nn.Sequential(OrderedDict([('se', ECABlock(12)),
                                                  ('linear', nn.Linear(768, 192))]))
# convnet13.features[3][0].block[5] = nn.Sequential(OrderedDict([('se', SELayer(6)),
#                                                   ('linear', nn.Linear(1536, 384))]))
convnet1.features[3][2].block[5] = nn.Sequential(OrderedDict([('se', ECABlock(6)),
                                                  ('linear', nn.Linear(1536, 384))]))

# convnet13.features[5][1].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet1.features[5][3].block[5] = nn.Sequential(OrderedDict([('se', ECABlock(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
convnet1.features[5][5].block[5] = nn.Sequential(OrderedDict([('se', ECABlock(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][7].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet1.features[5][9].block[5] = nn.Sequential(OrderedDict([('se', ECABlock(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][11].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet1.features[5][13].block[5] = nn.Sequential(OrderedDict([('se', ECABlock(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
convnet1.features[5][15].block[5] = nn.Sequential(OrderedDict([('se', ECABlock(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][17].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet1.features[5][19].block[5] = nn.Sequential(OrderedDict([('se', ECABlock(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
convnet1.features[5][21].block[5] = nn.Sequential(OrderedDict([('se', ECABlock(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][23].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet1.features[5][25].block[5] = nn.Sequential(OrderedDict([('se', ECABlock(3)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[7][0].block[5] = nn.Sequential(OrderedDict([('se', SELayer(6)),
#                                                   ('linear', nn.Linear(6144, 1536))]))
# convnet13.classifier[1] = nn.Sequential(OrderedDict([
#                                           ('se', SELayer(1536)),
#                                           ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet1.classifier[2] = nn.Sequential(OrderedDict([
                                          ('fc1', nn.Linear(1536,500)),
                                          ('fc2',nn.Linear(500,200)),
                                           ('fc3', nn.Linear(200,1))]))
convnet1 = convnet1.to(device)
train_valid(convnet1, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[ ]:


# multiple mixture bca and selayer + 3 fc layers


# In[19]:


# multiple cbamblock (fc) + 3 fc layers
import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet2 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet2.features[1][1].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("FC", 7, 12, 16)),
                                                  ('linear', nn.Linear(768, 192))]))
# convnet13.features[3][0].block[5] = nn.Sequential(OrderedDict([('se', SELayer(6)),
#                                                   ('linear', nn.Linear(1536, 384))]))
convnet2.features[3][2].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("FC", 7, 6, 16)),
                                                  ('linear', nn.Linear(1536, 384))]))

# convnet13.features[5][1].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet2.features[5][3].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("FC", 7, 3, 16)),
                                                  ('linear', nn.Linear(3072, 768))]))
convnet2.features[5][5].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("FC", 7, 3, 16)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][7].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet2.features[5][9].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("FC", 7, 3, 16)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][11].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet2.features[5][13].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("FC", 7, 3, 16)),
                                                  ('linear', nn.Linear(3072, 768))]))
convnet2.features[5][15].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("FC", 7, 3, 16)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][17].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet2.features[5][19].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("FC", 7, 3, 16)),
                                                  ('linear', nn.Linear(3072, 768))]))
convnet2.features[5][21].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("FC", 7, 3, 16)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][23].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet2.features[5][25].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("FC", 7, 3, 16)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[7][0].block[5] = nn.Sequential(OrderedDict([('se', SELayer(6)),
#                                                   ('linear', nn.Linear(6144, 1536))]))
# convnet13.classifier[1] = nn.Sequential(OrderedDict([
#                                           ('se', SELayer(1536)),
#                                           ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet2.classifier[2] = nn.Sequential(OrderedDict([
                                          ('fc1', nn.Linear(1536,500)),
                                          ('fc2',nn.Linear(500,200)),
                                           ('fc3', nn.Linear(200,1))]))
convnet2 = convnet2.to(device)
train_valid(convnet2, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[15]:


# multiple cbamblock (conv) + 3 fc layers
import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet2 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet2.features[1][1].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("Conv", 7, 12, 16, 2, 1)),
                                                  ('linear', nn.Linear(768, 192))]))
# convnet13.features[3][0].block[5] = nn.Sequential(OrderedDict([('se', SELayer(6)),
#                                                   ('linear', nn.Linear(1536, 384))]))
convnet2.features[3][2].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("Conv", 7, 6, 16, 2, 1)),
                                                  ('linear', nn.Linear(1536, 384))]))

# convnet13.features[5][1].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet2.features[5][3].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("Conv", 7, 3, 16, 2, 1)),
                                                  ('linear', nn.Linear(3072, 768))]))
convnet2.features[5][5].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("Conv", 7, 3, 16, 2, 1)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][7].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet2.features[5][9].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("Conv", 7, 3, 16, 2, 1)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][11].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet2.features[5][13].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("Conv", 7, 3, 16, 2, 1)),
                                                  ('linear', nn.Linear(3072, 768))]))
convnet2.features[5][15].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("Conv", 7, 3, 16, 2, 1)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][17].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet2.features[5][19].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("Conv", 7, 3, 16, 2, 1)),
                                                  ('linear', nn.Linear(3072, 768))]))
convnet2.features[5][21].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("Conv", 7, 3, 16, 2, 1)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[5][23].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
convnet2.features[5][25].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock("Conv", 7, 3, 16, 2, 1)),
                                                  ('linear', nn.Linear(3072, 768))]))
# convnet13.features[7][0].block[5] = nn.Sequential(OrderedDict([('se', SELayer(6)),
#                                                   ('linear', nn.Linear(6144, 1536))]))
# convnet13.classifier[1] = nn.Sequential(OrderedDict([
#                                           ('se', SELayer(1536)),
#                                           ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet2.classifier[2] = nn.Sequential(OrderedDict([
                                          ('fc1', nn.Linear(1536,500)),
                                          ('fc2',nn.Linear(500,200)),
                                           ('fc3', nn.Linear(200,1))]))
convnet2 = convnet2.to(device)
train_valid(convnet2, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[21]:


# multiple cbamblock_v3 (fc) + 3 fc layers
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet2 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet2.features[1][1].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock_v3("FC", 7, 12, 16)),
                                                  ('linear', nn.Linear(768, 192))]))
# convnet13.features[3][0].block[5] = nn.Sequential(OrderedDict([('se', SELayer(6)),
#                                                   ('linear', nn.Linear(1536, 384))]))
# convnet2.features[3][2].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock_v3("FC", 7, 12, 16)),
#                                                   ('linear', nn.Linear(1536, 384))]))

# convnet13.features[5][1].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
#                                                   ('linear', nn.Linear(3072, 768))]))
# convnet2.features[5][3].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock_v3("FC", 7, 6, 16)),
#                                                   ('linear', nn.Linear(3072, 768))]))
# convnet2.features[5][5].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock_v3("FC", 7, 6, 16)),
#                                                   ('linear', nn.Linear(3072, 768))]))
# # convnet13.features[5][7].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
# #                                                   ('linear', nn.Linear(3072, 768))]))
# convnet2.features[5][9].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock_v3("FC", 7, 6, 16)),
#                                                   ('linear', nn.Linear(3072, 768))]))
# # convnet13.features[5][11].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
# #                                                   ('linear', nn.Linear(3072, 768))]))
# convnet2.features[5][13].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock_v3("FC", 7, 6, 16)),
#                                                   ('linear', nn.Linear(3072, 768))]))
# convnet2.features[5][15].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock_v3("FC", 7, 6, 16)),
#                                                   ('linear', nn.Linear(3072, 768))]))
# # convnet13.features[5][17].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
# #                                                   ('linear', nn.Linear(3072, 768))]))
# convnet2.features[5][19].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock_v3("FC", 7, 5, 16)),
#                                                   ('linear', nn.Linear(3072, 768))]))
# convnet2.features[5][21].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock_v3("FC", 7, 6, 16)),
#                                                   ('linear', nn.Linear(3072, 768))]))
# # convnet13.features[5][23].block[5] = nn.Sequential(OrderedDict([('se', SELayer(3)),
# #                                                   ('linear', nn.Linear(3072, 768))]))
# convnet2.features[5][25].block[5] = nn.Sequential(OrderedDict([('combine',CBAMBlock_v3("FC", 7, 6, 16)),
#                                                   ('linear', nn.Linear(3072, 768))]))
# convnet13.features[7][0].block[5] = nn.Sequential(OrderedDict([('se', SELayer(6)),
#                                                   ('linear', nn.Linear(6144, 1536))]))
# convnet13.classifier[1] = nn.Sequential(OrderedDict([
#                                           ('se', SELayer(1536)),
#                                           ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet2.classifier[2] = nn.Sequential(OrderedDict([
                                          ('fc1', nn.Linear(1536,500)),
                                          ('fc2',nn.Linear(500,200)),
                                           ('fc3', nn.Linear(200,1))]))
convnet2 = convnet2.to(device)
train_valid(convnet2, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[ ]:


# multiple cbamblock_v3 (conv) + 3 fc layers


# In[ ]:


# multiple cbamblock_v4 (fc) + 3 fc layers


# In[ ]:


# multiple cbamblock_v4 (conv) + 3 fc layers


# ### Ensembling resnet50 and ConvNext

# In[ ]:





# In[ ]:


### Femal e VS Male 


# In[43]:


convnet13


# In[40]:


convnet13


# In[25]:


convnet13


# ### test different attention layer

# In[15]:


# se attention
import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet.classifier[1] = nn.Sequential(OrderedDict([
                                          ('se', SELayer(1536))]))
                                          #('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(1536, 1))]))
convnet = convnet.to(device)
train_valid(convnet, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[16]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet2 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet2.classifier[1] = nn.Sequential(OrderedDict([
                                          ('se', SELayer(1536)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet2.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(1536, 1))]))
convnet2 = convnet2.to(device)
train_valid(convnet2, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[17]:


# bce block attention 
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet1 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet1.classifier[1] = nn.Sequential(OrderedDict([
                                          ('se', ECABlock(1536))]))
convnet1.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(1536, 1))]))
convnet1 = convnet1.to(device)
train_valid(convnet1, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[18]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet4 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet4.classifier[1] = nn.Sequential(OrderedDict([
                                          ('se', ECABlock(1536)),
                                           ('pooling', nn.AdaptiveAvgPool2d(output_size=1))]))
convnet4.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(1536, 1))]))
convnet4 = convnet4.to(device)
train_valid(convnet2, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[16]:


# spatial attention
import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet3 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet3.classifier[1] = nn.Sequential(OrderedDict([
                                          ('spatial', Spatial_Attention_Module(7))]))
convnet3.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(1536, 1))]))
convnet3 = convnet3.to(device)
train_valid(convnet3, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[17]:


# combine attention 
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet5 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet5.classifier[1] = nn.Sequential(OrderedDict([
                                          ('combine',CBAMBlock("FC", 7, 1536, 16))]))
convnet5.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(1536, 1))]))
convnet5 = convnet5.to(device)
train_valid(convnet5, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[25]:


# concatenate 2 attention
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet6 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet6.classifier[1] = nn.Sequential(OrderedDict([
                                          ('combine',CBAMBlock_v2("FC", 7, 1536, 16))]))
convnet6.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(3072, 1))]))
convnet6 = convnet6.to(device)
train_valid(convnet6, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[27]:


# concatenate 2 attention + dropout
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet7 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet7.classifier[1] = nn.Sequential(OrderedDict([
                                          ('combine',CBAMBlock_v3("FC", 7, 1536, 16))]))
convnet7.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(3072, 1))]))
convnet7 = convnet7.to(device)
train_valid(convnet7, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[31]:


# combine attention + contencate features 
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet8 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet8.classifier[1] = nn.Sequential(OrderedDict([
                                          ('combine',CBAMBlock_v4("FC", 7, 1536, 16))]))
convnet8.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(4608, 1))]))
convnet8 = convnet8.to(device)
train_valid(convnet8, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[33]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet9 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet9.classifier[1] = nn.Sequential(OrderedDict([
                                          ('combine',CBAMBlock_v5("FC", 7, 1536, 16))]))
convnet9.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(3072, 1))]))
convnet9 = convnet9.to(device)
train_valid(convnet9, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[39]:


# combine attention + contencate features + dropout 
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet9 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet9.classifier[1] = nn.Sequential(OrderedDict([
                                          ('combine',CBAMBlock_v6("FC", 7, 1536, 16))]))
convnet9.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(4608, 1))]))
convnet9 = convnet9.to(device)
train_valid(convnet9, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[41]:


# combine attention + contencate features + dropout 
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet10 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet10.classifier[1] = nn.Sequential(OrderedDict([
                                          ('combine',CBAMBlock_v7("FC", 7, 1536, 16))]))
convnet10.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(3072, 1))]))
convnet10 = convnet10.to(device)
train_valid(convnet10, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[45]:


torch.cuda.empty_cache()


# In[15]:


import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet11 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet11.classifier[1] = nn.Sequential(OrderedDict([
                                          ('combine',CBAMBlock_v8("FC", 7, 1536, 16))]))
convnet11.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(3072, 1))]))
convnet11 = convnet11.to(device)
train_valid(convnet11, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[17]:


# combine attention + contencate features + dropout 
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet12 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet12.classifier[1] = nn.Sequential(OrderedDict([
                                          ('combine',CBAMBlock_v9("FC", 7, 1536, 16))]))
convnet12.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(4608, 1))]))
convnet12 = convnet12.to(device)
train_valid(convnet12, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[ ]:


use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet14 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet14.classifier[1] = nn.Sequential(OrderedDict([
                                          ('combine',CBAMBlock_v10("FC", 7, 1536, 16))]))
convnet14.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(3072, 1))]))
convnet14 = convnet14.to(device)
train_valid(convnet14, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[ ]:





# ### add multiple attention layers

# In[ ]:


# concatenate 2 attention


# In[ ]:


# combine attention + contencate features


# In[ ]:


# combine attention + contencate features + dropout 


# In[21]:


## combine 2 attention 
class CBAMBlock_v2(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None):
        super(CBAMBlock_v2, self).__init__()
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = SELayer(channel = channels, reduction= ratio)#Channel_Attention_Module_FC(channels = channels, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and gamma != None and b != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = ECABlock(channels = channels, gamma = gamma, b = b)#Channel_Attention_Module_Conv(channels = channels, gamma = gamma, b = b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = Spatial_Attention_Module(k = spatial_attention_kernel_size)

    def forward(self, x):
        channel_features= self.channel_attention_block(x)
        spatial_features = self.spatial_attention_block(x)
        out = torch.cat((channel_features, spatial_features), 1)
        
        return out


# In[16]:


## combine 2 attention 
class CBAMBlock_v3(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, dropout_rate: float = 0.2, gamma: int = None, b: int = None):
        super(CBAMBlock_v3, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = SELayer(channel = channels, reduction= ratio)#Channel_Attention_Module_FC(channels = channels, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and gamma != None and b != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = ECABlock(channels = channels, gamma = gamma, b = b)#Channel_Attention_Module_Conv(channels = channels, gamma = gamma, b = b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = Spatial_Attention_Module(k = spatial_attention_kernel_size)

    def forward(self, x):
        channel_features= self.channel_attention_block(x)
        spatial_features = self.spatial_attention_block(x)
        out = torch.cat((channel_features, spatial_features), 1)
        # add dropout 
        self.dropout(out)
        
        return out


# In[28]:


## combine 2 attention 
class CBAMBlock_v4(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None):
        super(CBAMBlock_v4, self).__init__()
        self.dropout = nn.Dropout(0.25)
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = SELayer(channel = channels, reduction= ratio)#Channel_Attention_Module_FC(channels = channels, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and gamma != None and b != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = ECABlock(channels = channels, gamma = gamma, b = b)#Channel_Attention_Module_Conv(channels = channels, gamma = gamma, b = b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = Spatial_Attention_Module(k = spatial_attention_kernel_size)

    def forward(self, x):
        channel_features= self.channel_attention_block(x)
        spatial_features = self.spatial_attention_block(x)
        out = torch.cat((x, channel_features, spatial_features), 1)
        # add dropout
        #self.dropout(out)
        
        return out


# In[32]:


## combine 2 attention 
class CBAMBlock_v5(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None):
        super(CBAMBlock_v5, self).__init__()
        self.dropout = nn.Dropout(0.25)
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = SELayer(channel = channels, reduction= ratio)#Channel_Attention_Module_FC(channels = channels, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and gamma != None and b != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = ECABlock(channels = channels, gamma = gamma, b = b)#Channel_Attention_Module_Conv(channels = channels, gamma = gamma, b = b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = Spatial_Attention_Module(k = spatial_attention_kernel_size)

    def forward(self, x):
        channel_features= self.channel_attention_block(x)
        #spatial_features = self.spatial_attention_block(x)
        out = torch.cat((x, channel_features), 1)
        return out


# In[37]:


## combine 2 attention 
class CBAMBlock_v6(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None):
        super(CBAMBlock_v6, self).__init__()
        self.dropout = nn.Dropout(0.25)
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = SELayer(channel = channels, reduction= ratio)#Channel_Attention_Module_FC(channels = channels, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and gamma != None and b != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = ECABlock(channels = channels, gamma = gamma, b = b)#Channel_Attention_Module_Conv(channels = channels, gamma = gamma, b = b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = Spatial_Attention_Module(k = spatial_attention_kernel_size)

    def forward(self, x):
        channel_features= self.channel_attention_block(x)
        spatial_features = self.spatial_attention_block(x)
        out = torch.cat((x, channel_features, spatial_features), 1)
        # add dropout
        self.dropout(out)
        return out


# In[40]:


## combine 2 attention 
class CBAMBlock_v7(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None):
        super(CBAMBlock_v7, self).__init__()
        self.dropout = nn.Dropout(0.25)
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = SELayer(channel = channels, reduction= ratio)#Channel_Attention_Module_FC(channels = channels, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and gamma != None and b != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = ECABlock(channels = channels, gamma = gamma, b = b)#Channel_Attention_Module_Conv(channels = channels, gamma = gamma, b = b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = Spatial_Attention_Module(k = spatial_attention_kernel_size)

    def forward(self, x):
        channel_features= self.channel_attention_block(x)
        #spatial_features = self.spatial_attention_block(x)
        out = torch.cat((x, channel_features), 1)
        # add dropout
        self.dropout(out)
        return out


# In[10]:


## combine 2 attention 
class CBAMBlock_v8(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None):
        super(CBAMBlock_v8, self).__init__()
        self.dropout = nn.Dropout(0.25)
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = SELayer(channel = channels, reduction= ratio)#Channel_Attention_Module_FC(channels = channels, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and gamma != None and b != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = ECABlock(channels = channels, gamma = gamma, b = b)#Channel_Attention_Module_Conv(channels = channels, gamma = gamma, b = b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = Spatial_Attention_Module(k = spatial_attention_kernel_size)

    def forward(self, x):
        channel_features= self.channel_attention_block(x)
        #spatial_features = self.spatial_attention_block(x)
        self.dropout(channel_features)
        out = torch.cat((x, channel_features), 1)
        # add dropout
        #self.dropout(out)
        return out


# In[16]:


## combine 2 attention 
class CBAMBlock_v9(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None):
        super(CBAMBlock_v9, self).__init__()
        self.dropout = nn.Dropout(0.25)
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = SELayer(channel = channels, reduction= ratio)#Channel_Attention_Module_FC(channels = channels, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and gamma != None and b != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = ECABlock(channels = channels, gamma = gamma, b = b)#Channel_Attention_Module_Conv(channels = channels, gamma = gamma, b = b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = Spatial_Attention_Module(k = spatial_attention_kernel_size)

    def forward(self, x):
        channel_features= self.channel_attention_block(x)
        spatial_features = self.spatial_attention_block(x)
        new_features = torch.cat((channel_features, spatial_features), 1)
        self.dropout(new_features)
        out = torch.cat((x, new_features), 1)
        # add dropout
        #self.dropout(out)
        return out


# In[18]:


## combine 2 attention 
class CBAMBlock_v10(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None):
        super(CBAMBlock_v10, self).__init__()
        self.dropout = nn.Dropout(0.25)
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = SELayer(channel = channels, reduction= ratio)#Channel_Attention_Module_FC(channels = channels, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and gamma != None and b != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = ECABlock(channels = channels, gamma = gamma, b = b)#Channel_Attention_Module_Conv(channels = channels, gamma = gamma, b = b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = Spatial_Attention_Module(k = spatial_attention_kernel_size)

    def forward(self, x):
        v = self.channel_attention_block(x)
        v = self.spatial_attention_block(v)
        new_features = self.dropout(v)
        out = torch.cat((x, new_features), 1)
        # add dropout
        #self.dropout(out)
        return out


# In[123]:


print(convnet4)


# #### Vgg16 (pretrained)

# ### Learning rate 0.0001

# In[14]:


## learning rate 0.01 is too bad (cannot proper converge) 7-6 mse
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
vgg2 = vgg16_bn(weights=VGG16_BN_Weights)
vgg2.classifier[6] = nn.Linear(vgg2.classifier[6].in_features, 1)
vgg2 = vgg2.to(device)
train_valid(vgg2, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# ### Learning rate 0.00001 

# In[16]:


# learning rate 0.00001 and 0.000001 is too small - results around 3 for mse, mae 1.3
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
vgg4 = vgg16_bn(weights=VGG16_BN_Weights)
vgg4.classifier[6] = nn.Linear(vgg4.classifier[6].in_features, 1)
vgg4 = vgg4.to(device)
train_valid(vgg4, device, 'MAE', train_set, test_set, 128, 0.00001,100,128)


# In[17]:


# save the model 
torch.save(vgg2.state_dict(), "/data/alzeye/yuru/VGG16_trained_crop50_128.pt")


# In[44]:


print(resnet1)


# ## Add Attention Layer 

# #### SE-resnet50 alone

# In[9]:


from Model import se_resnet50
import Model


# In[14]:


se_resnet = se_resnet50(pretrained=False)
print(se_resnet)


# In[12]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet = se_resnet50(pretrained=False).to(device)
train_valid(se_resnet, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[10]:


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# In[11]:


import math
class ECABlock(nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return x * v


# In[12]:


## spatial attention
class Spatial_Attention_Module(nn.Module):
    def __init__(self, k: int):
        super(Spatial_Attention_Module, self).__init__()
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max
        # In order to keep the size of the front and rear images consistent
        # with calculate, k = 1 + 2p, k denote kernel_size, and p denote padding number
        # so, when p = 1 -> k = 3; p = 2 -> k = 5; p = 3 -> k = 7, it works. when p = 4 -> k = 9, it is too big to use in network
        assert k in [3, 5, 7], "kernel size = 1 + 2 * padding, so kernel size must be 3, 5, 7"
        self.conv = nn.Conv2d(2, 1, kernel_size = (k, k), stride = (1, 1), padding = ((k - 1) // 2, (k - 1) // 2),
                              bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # compress the C channel to 1 and keep the dimensions
        avg_x = self.avg_pooling(x, dim = 1, keepdim = True)
        max_x, _ = self.max_pooling(x, dim = 1, keepdim = True)
        v = self.conv(torch.cat((max_x, avg_x), dim = 1))
        v = self.sigmoid(v)
        return x * v


# In[13]:


## combine 2 attention 
class CBAMBlock(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None):
        super(CBAMBlock, self).__init__()
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = SELayer(channel = channels, reduction= ratio)#Channel_Attention_Module_FC(channels = channels, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and gamma != None and b != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = ECABlock(channels = channels, gamma = gamma, b = b)#Channel_Attention_Module_Conv(channels = channels, gamma = gamma, b = b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = Spatial_Attention_Module(k = spatial_attention_kernel_size)

    def forward(self, x):
        x = self.channel_attention_block(x)
        x = self.spatial_attention_block(x)
        return x


# In[ ]:


if __name__ == "__main__":
    feature_maps = torch.randn((8, 54, 32, 32))
    model = CBAMBlock("FC", 5, channels = 54, ratio = 9)
    model(feature_maps)
    model = CBAMBlock("Conv", 5, channels = 54, gamma = 2, b = 1)
    model(feature_maps)


# In[13]:


# save the model 
torch.save(se_resnet.state_dict(), "/data/alzeye/yuru/SEResnet_crop50_128.pt")


# ### se-resnet pretrained 

# ### soft-attention layer with 2 fc layers

# In[22]:


# 1 soft attention layer with 2 fully connected layers
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
se_resnet.avgpool = nn.Sequential(OrderedDict([('se', SELayer(2048)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,1))]))

se_resnet = se_resnet.to(device)
train_valid(se_resnet, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[39]:


# 1 soft attention layer with 2 fully connected layers (resnet 34)
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet34 = resnet34(weights=ResNet34_Weights)
se_resnet34.avgpool = nn.Sequential(OrderedDict([('se', SELayer(512)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet34.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(512,1024)),
                                          ('fc2',nn.Linear(1024,1))]))


se_resnet34 = se_resnet34.to(device)
train_valid(se_resnet34, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[37]:


print(se_resnet34)


# In[15]:


# 1 soft attention layer with 2 fully connected layers (resnet 34), learning rate 0.0001
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet34_2 = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
se_resnet34_2.avgpool = nn.Sequential(OrderedDict([('se', SELayer(512)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet34_2.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(512,1024)),
                                          ('fc2',nn.Linear(1024,1))]))

se_resnet34_2 = se_resnet34_2.to(device)
train_valid(se_resnet34_2, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# ### change activiation function (RELU -> Leaky RELU)

# In[16]:





# In[38]:


## multiple attention layers with resnet 34 
# 1 soft attention layer with 2 fully connected layers (resnet 34), learning rate 0.0001
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet34_3 = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
se_resnet34_3.relu = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                           ('se', SELayer(64)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_3.layer1[0].relu = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                           ('se', SELayer(64)),
                                           ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_3.layer1[1].relu = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                          ('se', SELayer(64)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_3.layer1[2].relu = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                          ('se', SELayer(64)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_3.layer2[0].relu = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                          ('se', SELayer(128)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_3.layer2[1].relu = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                          ('se', SELayer(128)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_3.layer2[2].relu = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                          ('se', SELayer(128)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_3.layer2[3].relu = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                          ('se', SELayer(128)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_3.layer3[0].relu = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                          ('se', SELayer(256)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_3.layer3[1].relu = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                          ('se', SELayer(256)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_3.layer3[2].relu = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                          ('se', SELayer(256)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_3.layer3[3].relu = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                          ('se', SELayer(256)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_3.layer3[4].relu = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                          ('se', SELayer(256)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_3.layer3[5].relu = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                          ('se', SELayer(256)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_3.layer4[0].relu = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                          ('se', SELayer(512)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_3.layer4[1].relu = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                          ('se', SELayer(512)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_3.layer4[2].relu = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                          ('se', SELayer(512)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))

se_resnet34_3.avgpool = nn.Sequential(OrderedDict([('act', nn.ReLU(inplace=True)),
                                          ('se', SELayer(512)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet34_3.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(512,1024)),
                                             ('fc2',nn.Linear(1024,1))]))

se_resnet34_3 = se_resnet34_3.to(device)
train_valid(se_resnet34_3, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[40]:


## LeakyReLU
## multiple attention layers with resnet 34 
# 1 soft attention layer with 2 fully connected layers (resnet 34), learning rate 0.0001
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet34_4 = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
se_resnet34_4.relu = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(64)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_4.layer1[0].relu = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(64)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_4.layer1[1].relu = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(64)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_4.layer1[2].relu = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(64)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_4.layer2[0].relu = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(128)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_4.layer2[1].relu = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(128)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_4.layer2[2].relu = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(128)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_4.layer2[3].relu = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(128)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_4.layer3[0].relu = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(256)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_4.layer3[1].relu = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(256)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_4.layer3[2].relu = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(256)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_4.layer3[3].relu = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(256)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_4.layer3[4].relu = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(256)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_4.layer3[5].relu = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(256)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_4.layer4[0].relu = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(512)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_4.layer4[1].relu = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(512)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
se_resnet34_4.layer4[2].relu = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(512)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))

se_resnet34_4.avgpool = nn.Sequential(OrderedDict([('act', nn.LeakyReLU(inplace=True)),
                                          ('se', SELayer(512)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet34_4.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(512,1024)),
                                          ('fc2',nn.Linear(1024,1))]))

se_resnet34_4 = se_resnet34_4.to(device)
train_valid(se_resnet34_4, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# ### Change all ReLU function to Leaky ReLU

# In[ ]:





# In[24]:


## Convnext + one attention layer
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
convnet.classifier[1] = nn.Sequential(OrderedDict([
                                          ('se', SELayer(768)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(768, 1))]))
convnet = convnet.to(device)
train_valid(convnet, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[ ]:


## large convnet 


# In[19]:


convnet1 = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)


# In[20]:


convnet1


# ### ECANet layer with 2 fc layers

# In[28]:


# 1 ecanet layer with 2 fully connected layers
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
se_resnet1.avgpool = nn.Sequential(OrderedDict([('se', ECABlock(2048)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet1.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,1))]))


#print(se_resnet1)
se_resnet1 = se_resnet1.to(device)
train_valid(se_resnet1, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[17]:


## Convnext + one attention layer
import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet1 = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
convnet1.classifier[1] = nn.Sequential(OrderedDict([
                                          ('se', ECABlock(768)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet1.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(768, 1))]))
convnet1 = convnet1.to(device)
train_valid(convnet1, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[15]:


## Convnext + one attention layer
import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet1 = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
convnet1.classifier[1] = nn.Sequential(OrderedDict([
                                          ('se', ECABlock(768)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet1.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(768, 1))]))
convnet1 = convnet1.to(device)
train_valid(convnet1, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# ### Spatial layer with 2 fc layers

# In[31]:


## k =3 
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
sp_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
sp_resnet.avgpool = nn.Sequential(OrderedDict([('spatial', Spatial_Attention_Module(3)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
sp_resnet.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,1))]))


#print(sp_resnet)
sp_resnet = sp_resnet.to(device)
train_valid(sp_resnet, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[16]:


## k =5
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
sp_resnet1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
sp_resnet1.avgpool = nn.Sequential(OrderedDict([('spatial', Spatial_Attention_Module(5)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
sp_resnet1.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,1))]))


#print(sp_resnet)
sp_resnet1 = sp_resnet1.to(device)
train_valid(sp_resnet1, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[17]:


## k = 7
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
sp_resnet2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
sp_resnet2.avgpool = nn.Sequential(OrderedDict([('spatial', Spatial_Attention_Module(7)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
sp_resnet2.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,1))]))


#print(sp_resnet)
sp_resnet2 = sp_resnet2.to(device)
train_valid(sp_resnet2, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[14]:


import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet2 = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
convnet2.classifier[1] = nn.Sequential(OrderedDict([
                                          ('spatial', Spatial_Attention_Module(7)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet2.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(768, 1))]))
convnet2 = convnet2.to(device)
train_valid(convnet2, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# ### Combine Spatial and se layer with 2 fc layers

# In[22]:


## se layer (fc) + spatial 
from collections import OrderedDict
import torchmetrics
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
com_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
com_resnet.avgpool = nn.Sequential(OrderedDict([('combine', CBAMBlock("FC", 7, 2048, 16)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
com_resnet.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,1))]))


com_resnet = com_resnet.to(device)
train_valid(com_resnet, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[119]:


## add multiple attention layers
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
com_resnet3 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)


# In[120]:


com_resnet3.layer1[0].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
                                          ('combine',CBAMBlock_v2("FC", 7, 64, 16))]))
# com_resnet3.layer1[1].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
#                                           ('combine',CBAMBlock_v2("FC", 7, 256, 16))]))

# com_resnet3.layer1[2].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
#                                           ('combine',CBAMBlock_v2("FC", 7, 256, 16))]))
# com_resnet3.layer2[0].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
#                                           ('combine',CBAMBlock_v2("FC", 7, 128, 16))]))
# com_resnet3.layer2[1].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
#                                           ('combine',CBAMBlock_v2("FC", 7, 128, 16))]))
# com_resnet3.layer2[2].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
#                                          ('combine',CBAMBlock_v2("FC", 7, 128, 16))]))
# com_resnet3.layer2[3].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
#                                           ('combine',CBAMBlock_v2("FC", 7, 128, 16))]))
# com_resnet3.layer3[0].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
#                                           ('combine',CBAMBlock_v2("FC", 7, 256, 16))]))
# com_resnet3.layer3[1].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
#                                          ('combine',CBAMBlock_v2("FC", 7, 256, 16))]))
# com_resnet3.layer3[2].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
#                                         ('combine',CBAMBlock_v2("FC", 7, 256, 16))]))
# com_resnet3.layer3[3].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
#                                         ('combine',CBAMBlock_v2("FC", 7, 256, 16))]))
# com_resnet3.layer3[4].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
#                                          ('combine',CBAMBlock_v2("FC", 7, 256, 16))]))
# com_resnet3.layer3[5].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
#                                         ('combine',CBAMBlock_v2("FC", 7, 256, 16))]))
# com_resnet3.layer4[0].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
#                                         ('combine',CBAMBlock_v2("FC", 7, 512, 16))]))
# com_resnet3.layer4[1].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
#                                        ('combine',CBAMBlock_v2("FC", 7, 512, 16))]))
# com_resnet3.layer4[2].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
#                                         ('combine',CBAMBlock_v2("FC", 7, 512, 16))]))
# com_resnet3.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(512,1024)),
#                                           ('fc2',nn.Linear(1024,1))]))


# In[116]:


t = torch.ones((128,64,13,13))
t.shape


# In[117]:


model = CBAMBlock_v2("FC", 7, 64, 16)


# In[118]:


model(t)


# In[72]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet3 = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
convnet3.classifier[1] = nn.Sequential(OrderedDict([
                                          ('combine', CBAMBlock("FC", 7, 2048, 16)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet3.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(768, 1))]))
convnet3 = convnet3.to(device)
train_valid(convnet3, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[ ]:





# ### Concat Spatial and se layer with 2 fc layers

# In[115]:


class CBAMBlock_v2(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None):
        super(CBAMBlock_v2, self).__init__()
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = SELayer(channel = channels, reduction = ratio)#Channel_Attention_Module_FC(channels = channels, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and gamma != None and b != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = ECABlock(channels = channels, gamma = gamma, b = b)#Channel_Attention_Module_Conv(channels = channels, gamma = gamma, b = b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = Spatial_Attention_Module(k = spatial_attention_kernel_size)

    def forward(self, x):
        out = self.channel_attention_block(x)
        out = self.spatial_attention_block(out)
        #final = x*out
        return x*out


# In[33]:


use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
com_resnet1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
com_resnet1.avgpool = nn.Sequential(OrderedDict([('combine', CBAMBlock_v2("FC", 7, 2048, 16)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
com_resnet1.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,1))]))


#print(com_resnet1)
com_resnet1 = com_resnet1.to(device)
train_valid(com_resnet1, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[ ]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
convnet.classifier[1] = nn.Sequential(OrderedDict([
                                          ('spatial', CBAMBlock_v2("FC", 5, channel = 54, reduction = 16)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet.classifier[2] = nn.Sequential(OrderedDict([('fc1',nn.Linear(768, 1000)),
                                                  ('fc2',nn.Linear(1000, 1))]))                                   
                                          


# In[17]:


## add multiple attention layers
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
com_resnet2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
print(com_resnet2)
#com_resnet1.avgpool = nn.Sequential(OrderedDict([('spatial', CBAMBlock_v2("FC", 5, channels = 54, ratio = 16)),
                                          #('pool',nn.AdaptiveAvgPool2d(output_size=1))]))


# In[21]:


com_resnet2.layer1[0].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
                                          ('combine',CBAMBlock_v2("FC", 5, channels = 54, ratio = 16))]))
com_resnet2.layer1[1].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
                                          ('combine',CBAMBlock_v2("FC", 5, channels = 54, ratio = 16))]))

com_resnet2.layer1[2].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
                                          ('combine',CBAMBlock_v2("FC", 5, channels = 54, ratio = 16))]))
com_resnet2.layer2[0].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
                                          ('combine',CBAMBlock_v2("FC", 5, channels = 54, ratio = 16))]))
com_resnet2.layer2[1].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
                                          ('combine',CBAMBlock_v2("FC", 5, channels = 54, ratio = 16))]))
com_resnet2.layer2[2].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
                                          ('combine',CBAMBlock_v2("FC", 5, channels = 54, ratio = 16))]))
com_resnet2.layer2[3].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
                                          ('combine',CBAMBlock_v2("FC", 5, channels = 54, ratio = 16))]))
com_resnet2.layer3[0].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
                                          ('combine',CBAMBlock_v2("FC", 5, channels = 54, ratio = 16))]))
com_resnet2.layer3[1].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
                                          ('combine',CBAMBlock_v2("FC", 5, channels = 54, ratio = 16))]))
com_resnet2.layer3[2].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
                                          ('combine',CBAMBlock_v2("FC", 5, channels = 54, ratio = 16))]))
com_resnet2.layer3[3].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
                                          ('combine',CBAMBlock_v2("FC", 5, channels = 54, ratio = 16))]))
com_resnet2.layer3[4].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
                                          ('combine',CBAMBlock_v2("FC", 5, channels = 54, ratio = 16))]))
com_resnet2.layer3[5].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
                                          ('combine',CBAMBlock_v2("FC", 5, channels = 54, ratio = 16))]))
com_resnet2.layer4[0].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
                                          ('combine',CBAMBlock_v2("FC", 5, channels = 54, ratio = 16))]))
com_resnet2.layer4[1].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
                                          ('combine',CBAMBlock_v2("FC", 5, channels = 54, ratio = 16))]))
com_resnet2.layer4[2].relu = nn.Sequential(OrderedDict([('relu', nn.ReLU(inplace=True)),
                                          ('combine',CBAMBlock_v2("FC", 5, channels = 54, ratio = 16))]))
com_resnet2.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,1))]))


# In[25]:


com_resnet1.layer2[0]


# In[18]:


from Model import se_resnet50
se_resnet = se_resnet50(pretrained=False)
print(se_resnet)


# In[26]:


## Convnext + one attention layer
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
convnet.classifier[2] = nn.Linear(convnet2.classifier[2].in_features, 1)
#convnet2 = convnet2.to(device)
#train_valid(convnet2, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[36]:


convnet.features[7][2][6] = nn.Sequential(OrderedDict([('permute', nn.Permute()),
                                          ('spatial', CBAMBlock_v2("FC", 5, channels = 54, ratio = 16))]))


# In[ ]:


## add multiple attention layers


# In[30]:


print(convnet)


# In[ ]:


class CBAMBlock_v3(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None):
        super(CBAMBlock_v3, self).__init__()
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = SELayer(channel = channels, reduction = ratio)#Channel_Attention_Module_FC(channels = channels, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and gamma != None and b != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = ECABlock(channels = channels, gamma = gamma, b = b)#Channel_Attention_Module_Conv(channels = channels, gamma = gamma, b = b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = Spatial_Attention_Module(k = spatial_attention_kernel_size)

    def forward(self, x):
        return self.channel_attention_block(x)*self.spatial_attention_block(x)*x


# In[ ]:


## adding dropout layers after concatenate 
class CBAMBlock_v4(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None, dropout_ratio: float = 0.2):
        super(CBAMBlock_v4, self).__init__()
        self.dropout = nn.Dropout(dropout_ratio)
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = SELayer(channel = channels, reduction = ratio)#Channel_Attention_Module_FC(channels = channels, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and gamma != None and b != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = ECABlock(channels = channels, gamma = gamma, b = b)#Channel_Attention_Module_Conv(channels = channels, gamma = gamma, b = b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = Spatial_Attention_Module(k = spatial_attention_kernel_size)

    def forward(self, x):
        out = self.channel_attention_block(x)*self.spatial_attention_block(x)*x
        out = self.dropout(out)
        return out


# ## LSTM

# ### LSTM_dropout

# In[17]:


print(train_set[0][0].shape)


# In[10]:


import Model


# In[23]:


print(train_set[0][0][3]) # 50 **50 


# In[16]:


t = torch.zeros((3,50,50))
t.shape


# In[17]:


t.permute(0,2,1)


# In[61]:


t.reshape(3,-1).shape


# In[59]:


def train_valid1(model, device, criterion, train_set, valid_set, train_batch_size, learning_rate, num_epochs,
                val_batch_size, schedule=False):
    import torchmetrics
    # dataloader
    train_loader = DataLoader(train_set, train_batch_size, shuffle=True)
    val_loader = DataLoader(valid_set, val_batch_size, shuffle=True)

    # model nitialization
    criterion_collect = {'MSE': nn.MSELoss(), 'MAE': nn.L1Loss()}
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    
    # metrics initialization
    if criterion.upper() == 'MSE':
        print('Current loss is MSE, Metrics for training/validation is MAE')
        train_metric = torchmetrics.MeanAbsoluteError().to(device)
        val_metric = torchmetrics.MeanAbsoluteError().to(device)
    else:
        print('Current loss is MAE, Metrics for training/validation is MSE')
        train_metric = torchmetrics.MeanSquaredError().to(device)
        val_metric = torchmetrics.MeanSquaredError().to(device)
    
    train_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    val_mape = torchmetrics.MeanAbsolutePercentageError().to(device)

    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_metrics = np.inf


    #### start train the model
    for epoch in range(num_epochs):
        torch.manual_seed(1+epoch)
        # train the model
        model.train()
        print(f'Epoch {epoch+1}/{num_epochs - 1}')
        print('-' * 10)

        running_loss = 0.0
        for i_batches, (data, target) in enumerate(train_loader,1):
            data = data.reshape(train_batch_size,3,-1).float().to(device) #reshape(-1, 3, 2500).
            targets = target.float().to(device)  
            #print(data.shape)
            predict = model(data)
            optimizer.zero_grad()
            loss = criterion_collect[criterion.upper()](predict, targets)
            loss.backward()
            optimizer.step()

            # metric calculation (loss, mse/mae, mape)
            running_loss += loss.item()  # total loss of the current batch(not averaged)
            train_metric(predict, targets)
            train_mape(predict, targets)
            
            if schedule == True:
                scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
                scheduler.step()

        # average training loss over an epoch
        train_loss = running_loss/i_batches # average sample loss over an epoch
        # average mae/mse and mape over all training batches
        avg_train_metric = train_metric.compute()
        avg_train_mape = train_mape.compute()

        print('Training set: Average Loss: {:.5f}, Average_metric:{:.4f}, Average_MAPE:{:.4f}'.format(train_loss, avg_train_metric, avg_train_mape))

        # validate the model
        model.eval()
        val_loss = 0
        
        # no gradient computation
        with torch.no_grad():
            for i_batches_val, (data, target) in enumerate(val_loader,1):
                data = data.reshape(val_batch_size, 3, -1).float().to(device)
                targets = target.float().to(device)
                y_hat = model(data)
                loss = criterion_collect[criterion.upper()](y_hat, targets)

                # metric calculation (loss, mse/mae, mape)
                val_loss += loss.item()
                val_metric(y_hat, targets)
                val_mape(y_hat, targets)
            
        # average validation loss over an epoch
        val_loss = val_loss/ i_batches_val
        # average mae/mse and mape over all training batches
        avg_val_metric = val_metric.compute()
        avg_val_mape = val_mape.compute()

        print('Validation set: Average loss: {:.4f}, Average_metric:{:.4f}, Average_MAPE:{:.4f}'.format(val_loss, avg_val_metric, avg_val_mape))
        

        # reset metrics for the next epoch
        train_metric.reset()
        val_metric.reset()
        train_mape.reset()
        val_mape.reset()



# In[55]:


class LSTM_dropout(nn.Module):
    """
    Param:
    input_size: feature size
    hidden_size: the number of hidden layers
    output_size: the number of output
    num_layers: the number of layers for LSTM
    dropout: the probability ratio of dropout
    """

    def __init__(self, input_size, output_size, hidden_size=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bn = nn.BatchNorm2d(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # initialize the cell state:
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        #x = self.bn(x)
        out, _ = self.lstm(x, (h0,c0))
        # size, batch, hidden = out.shape
        # out = out.view(batch, size * hidden)
        out = self.linear(out[:,-1,:])
        return out


# In[60]:


## 
input_size = 2500
sequence_len = 3
#hidden_size = 128
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
lstm = LSTM_dropout(input_size=input_size, output_size=1).to(device)
train_valid1(lstm, device, 'MAE', train_set, test_set, 128, 0.001,100,128)


# In[ ]:





# In[ ]:




