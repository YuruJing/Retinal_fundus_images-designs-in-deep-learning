#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install torchmetrics # add that into docker 


# In[5]:


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

# In[1]:


ukbb_images = '/data/alzeye/ukbb/crop_50/'
ukbb_csv    = '/data/alzeye/ukbb/MERGED_UKBB_FUNDUS_final.csv'


# In[3]:


get_ipython().system('nvidia-smi')


# In[6]:


data = pd.read_csv('all_crop50_eyes.csv')
data


# In[10]:


91986*0.1


# In[7]:


data.describe()


# In[11]:


full_eye = list(data['file_name'])
full_target = list(data['spherical_equivalent'])


# In[12]:


# without considering the gender
full_data = Custom_Dataset(ukbb_images, full_eye, full_target)


# In[42]:


len(data[data['gender'] == 1.0]), len(data[data['gender'] == 0.0])


# In[44]:


len(data[data['gender'] == 1.0])/len(data), len(data[data['gender'] == 0.0])/len(data)


# In[5]:


### Male and female data
males_eye= list(data[data['gender'] == 1.0]['file_name'])
males_target = list(data[data['gender']==1.0]['spherical_equivalent'])


# In[6]:


full_data = Custom_Dataset(ukbb_images, males_eye, males_target)


# In[22]:


female_eye = list(data[data['gender'] == 0.0]['file_name'])
female_target = list(data[data['gender']==0.0]['spherical_equivalent'])


# In[23]:


full_data = Custom_Dataset(ukbb_images, female_eye, female_target)


# ## Add Data Augmentation 

# In[13]:


# normalize the images and train_test_split the data  
transf = transforms.Compose([ transforms.Resize(224),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomRotation(45),
                              transforms.ToTensor()])

test_transf = transforms.Compose([ transforms.Resize(224),
                              transforms.ToTensor()])


# In[14]:


# without normalization
train_set, test_set = train_test_split(full_data, test_ratio = 0.1, train_trans=transf, test_trans=test_transf, workers=0) 


# In[15]:


# get the mean and std for batchsize 128
mean, std = get_mean_sd(train_set, 128) 
test_mean, test_std = get_mean_sd(test_set, 128)


# In[16]:


# with normalization 
train_set.transform = transforms.Compose([ 
                              transforms.ToTensor(),
                              transforms.Normalize(mean=mean, std=std)])
test_set.transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=test_mean, std=test_std)])


# In[12]:


mean, std


# In[11]:


test_mean, test_std


# In[21]:


total = 0
for i in range(len(train_set)):
    total += train_set[i][1]


# In[22]:


mean = total/len(train_set)
mean 


# In[23]:


std_sum = 0
for i in range(len(train_set)):
    std_sum += (train_set[i][1] - mean)**2


# In[24]:


std = np.sqrt(std_sum/len(train_set))
std


# In[25]:


total = 0
for i in range(len(test_set)):
    total += test_set[i][1]


# In[26]:


mean = total/len(test_set)
mean 


# In[27]:


std_sum = 0
for i in range(len(test_set)):
    std_sum += (test_set[i][1] - mean)**2


# In[28]:


std = np.sqrt(std_sum/len(test_set))
std


# ## Google Net

# ### batch_size= 128, test_ratio=0.1, with data augmentation, learning rate 0.001

# In[14]:


use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
model = inception.GoogLeNet(aux_logits=True, num_classes=1).to(device)
googlenet_train_valid(model, device, 'MAE', train_set, test_set, 128 , 0.001,100,128)


# ### batch_size= 128, test_ratio=0.1, with data augmentation, learning rate 0.00001

# In[16]:


use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
model2 = inception.GoogLeNet(aux_logits=True, num_classes=1).to(device)
googlenet_train_valid(model2, device, 'MAE', train_set, test_set, 128 , 0.00001,100,128)


# In[18]:


# save the model 
torch.save(model1.state_dict(), "/data/alzeye/yuru/GoogleNet_crop50_final128.pt")


# ## VGG-16, ResNet, Densenet, EfficientNet, Convnext, (Pre-trained False)

# In[11]:


from torchvision.models import resnet50, ResNet50_Weights, densenet161, DenseNet161_Weights, efficientnet_v2_l, EfficientNet_V2_L_Weights, convnext_large, ConvNeXt_Large_Weights,convnext_small, ConvNeXt_Small_Weights, vgg16_bn, VGG16_BN_Weights


# ### batch_size= 128, test_ratio=0.1, without data augmentation, learning rate 0.01

# #### resnet50

# In[18]:


use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
resnet.fc = nn.Linear(resnet.fc.in_features, 1)
resnet = resnet.to(device)
train_valid(resnet, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[21]:


# save the model 
torch.save(resnet.state_dict(), "/data/alzeye/yuru/Resnet_crop50_128.pt")


# In[13]:


import Model
from torch.distributions import uniform


# In[14]:


train_set[0][0].shape


# In[13]:


import random
IMAGE_WIDTH = 50
IMAGE_HEIGHT = 50
MAX_MASK_SIZE = 5 #IMAGE_WIDTH / 5


# In[14]:


def cutout(image):
    """
    Apply the cutout algorithm to the image, as detailed in the first coursework specification.
    :param image: (Tensor) Image to apply cutout to.
    :return:
    """
    new_image = image.clone()
    for img in new_image:
        mask_size = round(random.uniform(0, MAX_MASK_SIZE))
        mask_top_left = mask_size // 2
        mask_bottom_right = mask_size - mask_top_left
        centre_pixel = (round(random.uniform(mask_top_left, IMAGE_WIDTH - mask_bottom_right - 1)),
                        round(random.uniform(mask_top_left, IMAGE_HEIGHT - mask_bottom_right - 1)))
        centre_x = centre_pixel[0]
        centre_y = centre_pixel[1]
        left = centre_x - mask_top_left
        right = centre_x + mask_bottom_right
        upper = centre_y - mask_top_left
        lower = centre_y + mask_bottom_right
        for x in range(left, right):
            for y in range(upper, lower):
                for channel in range(len(img)):
                #for channel in range(len(new_image)):
                    #print(img.shape)
                    img[channel, y, x] = 0
                    #new_img[channel, y, x] = 0
    return new_image


# In[15]:


def train_valid2(model, device, criterion, train_set, valid_set, train_batch_size, learning_rate, num_epochs,
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
    
    #train_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    #val_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    train_r2 = torchmetrics.R2Score().to(device)
    val_r2 = torchmetrics.R2Score().to(device)

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
            data = cutout(data).float().to(device)
            targets = target.float().to(device)  
            predict = model(data)
            optimizer.zero_grad()
            loss = criterion_collect[criterion.upper()](predict, targets)
            loss.backward()
            optimizer.step()

            # metric calculation (loss, mse/mae, mape)
            running_loss += loss.item()  # total loss of the current batch(not averaged)
            train_metric(predict, targets)
            #train_mape(predict, targets)
            train_r2(predict, targets)
            
            if schedule == True:
                scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
                scheduler.step()

        # average training loss over an epoch
        train_loss = running_loss/i_batches # average sample loss over an epoch
        # average mae/mse and mape over all training batches
        avg_train_metric = train_metric.compute()
        #avg_train_mape = train_mape.compute()
        avg_train_r2 = train_r2.compute()

        print('Training set: Average Loss: {:.5f}, Average_metric:{:.4f}, Average_R2:{:.4f}'.format(train_loss, avg_train_metric, avg_train_r2))

        # validate the model
        model.eval()
        val_loss = 0
        
        # no gradient computation
        with torch.no_grad():
            for i_batches_val, (data, target) in enumerate(val_loader,1):
                data = data.float().to(device)
                targets = target.float().to(device)
                y_hat = model(data)
                loss = criterion_collect[criterion.upper()](y_hat, targets)

                # metric calculation (loss, mse/mae, mape)
                val_loss += loss.item()
                val_metric(y_hat, targets)
                #val_mape(y_hat, targets)
                val_r2(y_hat, targets)
            
        # average validation loss over an epoch
        val_loss = val_loss/ i_batches_val
        # average mae/mse and mape over all training batches
        avg_val_metric = val_metric.compute()
        #avg_val_mape = val_mape.compute()
        avg_val_r2 = val_r2.compute()

        print('Validation set: Average loss: {:.4f}, Average_metric:{:.4f}, Average_R2:{:.4f}'.format(val_loss, avg_val_metric, avg_val_r2))
        

        # reset metrics for the next epoch
        train_metric.reset()
        val_metric.reset()
        #train_mape.reset()
        #val_mape.reset()
        train_r2.reset()
        val_r2.reset()


# In[12]:


def train_valid3(model, device, criterion, train_set, valid_set, train_batch_size, learning_rate, num_epochs,
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
    
    #train_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    #val_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    train_r2 = torchmetrics.R2Score().to(device)
    val_r2 = torchmetrics.R2Score().to(device)

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
            data = data.float().to(device)
            targets = target.float().to(device)  
            predict = model(data)
            optimizer.zero_grad()
            loss = criterion_collect[criterion.upper()](predict, targets)
            loss.backward()
            optimizer.step()

            # metric calculation (loss, mse/mae, mape)
            running_loss += loss.item()  # total loss of the current batch(not averaged)
            train_metric(predict, targets)
            #train_mape(predict, targets)
            train_r2(predict, targets)
            
            if schedule == True:
                scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
                scheduler.step()

        # average training loss over an epoch
        train_loss = running_loss/i_batches # average sample loss over an epoch
        # average mae/mse and mape over all training batches
        avg_train_metric = train_metric.compute()
        #avg_train_mape = train_mape.compute()
        avg_train_r2 = train_r2.compute()

        print('Training set: Average Loss: {:.5f}, Average_metric:{:.4f}, Average_R2:{:.4f}'.format(train_loss, avg_train_metric, avg_train_r2))

        # validate the model
        model.eval()
        val_loss = 0
        
        # no gradient computation
        with torch.no_grad():
            for i_batches_val, (data, target) in enumerate(val_loader,1):
                data = data.float().to(device)
                targets = target.float().to(device)
                y_hat = model(data)
                loss = criterion_collect[criterion.upper()](y_hat, targets)

                # metric calculation (loss, mse/mae, mape)
                val_loss += loss.item()
                val_metric(y_hat, targets)
                #val_mape(y_hat, targets)
                val_r2(y_hat, targets)
            
        # average validation loss over an epoch
        val_loss = val_loss/ i_batches_val
        # average mae/mse and mape over all training batches
        avg_val_metric = val_metric.compute()
        #avg_val_mape = val_mape.compute()
        avg_val_r2 = val_r2.compute()

        print('Validation set: Average loss: {:.4f}, Average_metric:{:.4f}, Average_R2:{:.4f}'.format(val_loss, avg_val_metric, avg_val_r2))
        

        # reset metrics for the next epoch
        train_metric.reset()
        val_metric.reset()
        #train_mape.reset()
        #val_mape.reset()
        train_r2.reset()
        val_r2.reset()


# In[30]:


def train_valid4(model1, model2, device, criterion, train_set, valid_set, train_batch_size, learning_rate, num_epochs,
                val_batch_size, schedule=False):
    import torchmetrics
    
    # dataloader
    train_loader = DataLoader(train_set, train_batch_size, shuffle=True)
    val_loader = DataLoader(valid_set, val_batch_size, shuffle=True)

    # model nitialization
    criterion_collect = {'MSE': nn.MSELoss(), 'MAE': nn.L1Loss()}
    optimizer = torch.optim.Adam(params=model1.parameters(), lr=learning_rate)
    
    # metrics initialization
    if criterion.upper() == 'MSE':
        print('Current loss is MSE, Metrics for training/validation is MAE')
        train_metric = torchmetrics.MeanAbsoluteError().to(device)
        val_metric = torchmetrics.MeanAbsoluteError().to(device)
    else:
        print('Current loss is MAE, Metrics for training/validation is MSE')
        train_metric = torchmetrics.MeanSquaredError().to(device)
        val_metric = torchmetrics.MeanSquaredError().to(device)
    
    #train_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    #val_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    train_r2 = torchmetrics.R2Score().to(device)
    val_r2 = torchmetrics.R2Score().to(device)

    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_metrics = np.inf


    #### start train the model
    for epoch in range(num_epochs):
        torch.manual_seed(1+epoch)
        # train the model
        #model.train()
        print(f'Epoch {epoch+1}/{num_epochs - 1}')
        print('-' * 10)

        running_loss = 0.0
        for i_batches, (data, target) in enumerate(train_loader,1):
            data = cutout(data).float().to(device)
            targets = target.float().to(device)  
            pre_value1 = model1(data)
            pre_value2 = model2(data)
            predict = (pre_value1 + pre_value2)/2
            optimizer.zero_grad()
            loss = criterion_collect[criterion.upper()](predict, targets)
            loss.backward()
            optimizer.step()

            # metric calculation (loss, mse/mae, mape)
            running_loss += loss.item()  # total loss of the current batch(not averaged)
            train_metric(predict, targets)
            #train_mape(predict, targets)
            train_r2(predict, targets)
            
            if schedule == True:
                scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
                scheduler.step()

        # average training loss over an epoch
        train_loss = running_loss/i_batches # average sample loss over an epoch
        # average mae/mse and mape over all training batches
        avg_train_metric = train_metric.compute()
        #avg_train_mape = train_mape.compute()
        avg_train_r2 = train_r2.compute()

        print('Training set: Average Loss: {:.5f}, Average_metric:{:.4f}, Average_R2:{:.4f}'.format(train_loss, avg_train_metric, avg_train_r2))

        # validate the model
        #model.eval()
        val_loss = 0
        
        # no gradient computation
        with torch.no_grad():
            for i_batches_val, (data, target) in enumerate(val_loader,1):
                data = data.float().to(device)
                targets = target.float().to(device)
                y1 = model1(data)
                y2 = model2(data)
                y_hat = (y1 + y2)/2
                loss = criterion_collect[criterion.upper()](y_hat, targets)

                # metric calculation (loss, mse/mae, mape)
                val_loss += loss.item()
                val_metric(y_hat, targets)
                #val_mape(y_hat, targets)
                val_r2(y_hat, targets)
            
        # average validation loss over an epoch
        val_loss = val_loss/ i_batches_val
        # average mae/mse and mape over all training batches
        avg_val_metric = val_metric.compute()
        #avg_val_mape = val_mape.compute()
        avg_val_r2 = val_r2.compute()

        print('Validation set: Average loss: {:.4f}, Average_metric:{:.4f}, Average_R2:{:.4f}'.format(val_loss, avg_val_metric, avg_val_r2))
        

        # reset metrics for the next epoch
        train_metric.reset()
        val_metric.reset()
        #train_mape.reset()
        #val_mape.reset()
        train_r2.reset()
        val_r2.reset()


# In[13]:


def test(model, device, criterion, valid_set, learning_rate, val_batch_size):

    val_loader = DataLoader(valid_set, val_batch_size, shuffle=True)

    # model nitialization
    criterion_collect = {'MSE': nn.MSELoss(), 'MAE': nn.L1Loss()}
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    
    # metrics initialization
    if criterion.upper() == 'MSE':
        print('Current loss is MSE, Metrics for training/validation is MAE')
        val_metric = torchmetrics.MeanAbsoluteError().to(device)
    else:
        print('Current loss is MAE, Metrics for training/validation is MSE')
        val_metric = torchmetrics.MeanSquaredError().to(device)
    
    #train_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    #val_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    val_r2 = torchmetrics.R2Score().to(device)
    # test the model
    model.eval()
    val_loss = 0
        
    # no gradient computation
    with torch.no_grad():
        for i_batches_val, (data, target) in enumerate(val_loader,1):
            data = data.float().to(device)
            targets = target.float().to(device)
            y_hat = model(data)
            loss = criterion_collect[criterion.upper()](y_hat, targets)

            # metric calculation (loss, mse/mae, mape)
            val_loss += loss.item()
            val_metric(y_hat, targets)
            #val_mape(y_hat, targets)
            val_r2(y_hat, targets)
            
    # average validation loss over an epoch
    val_loss = val_loss/ i_batches_val
    # average mae/mse and mape over all training batches
    avg_val_metric = val_metric.compute()
    #avg_val_mape = val_mape.compute()
    avg_val_r2 = val_r2.compute()

    print('Test set: Average loss: {:.4f}, Average_metric:{:.4f}, Average_R2:{:.4f}'.format(val_loss, avg_val_metric, avg_val_r2))


# In[34]:


def test1(model1,model2, device, criterion, valid_set, learning_rate, val_batch_size):

    val_loader = DataLoader(valid_set, val_batch_size, shuffle=True)

    # model nitialization
    criterion_collect = {'MSE': nn.MSELoss(), 'MAE': nn.L1Loss()}
    optimizer = torch.optim.Adam(params=model1.parameters(), lr=learning_rate)
    
    # metrics initialization
    if criterion.upper() == 'MSE':
        print('Current loss is MSE, Metrics for training/validation is MAE')
        val_metric = torchmetrics.MeanAbsoluteError().to(device)
    else:
        print('Current loss is MAE, Metrics for training/validation is MSE')
        val_metric = torchmetrics.MeanSquaredError().to(device)
    
    #train_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    #val_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    val_r2 = torchmetrics.R2Score().to(device)
    # test the model
    #model.eval()
    val_loss = 0
        
    # no gradient computation
    with torch.no_grad():
        for i_batches_val, (data, target) in enumerate(val_loader,1):
            data = data.float().to(device)
            targets = target.float().to(device)
            y1 = model1(data)
            y2 = model2(data)
            y_hat = (y1+y2)/2
            loss = criterion_collect[criterion.upper()](y_hat, targets)

            # metric calculation (loss, mse/mae, mape)
            val_loss += loss.item()
            val_metric(y_hat, targets)
            #val_mape(y_hat, targets)
            val_r2(y_hat, targets)
            
    # average validation loss over an epoch
    val_loss = val_loss/ i_batches_val
    # average mae/mse and mape over all training batches
    avg_val_metric = val_metric.compute()
    #avg_val_mape = val_mape.compute()
    avg_val_r2 = val_r2.compute()

    print('Test set: Average loss: {:.4f}, Average_metric:{:.4f}, Average_R2:{:.4f}'.format(val_loss, avg_val_metric, avg_val_r2))


# In[ ]:


def test3(model, device, criterion, valid_set, learning_rate, val_batch_size):

    val_loader = DataLoader(valid_set, val_batch_size, shuffle=True)

    # model nitialization
    criterion_collect = {'MSE': nn.MSELoss(), 'MAE': nn.L1Loss()}
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    
    # metrics initialization
    if criterion.upper() == 'MSE':
        print('Current loss is MSE, Metrics for training/validation is MAE')
        val_metric = torchmetrics.MeanAbsoluteError().to(device)
    else:
        print('Current loss is MAE, Metrics for training/validation is MSE')
        val_metric = torchmetrics.MeanSquaredError().to(device)
    
    #train_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    #val_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    val_r2 = torchmetrics.R2Score().to(device)
    # test the model
    model.eval()
    val_loss = 0
    
    diff_value = []
        
    # no gradient computation
    with torch.no_grad():
        for i_batches_val, (data, target) in enumerate(val_loader,1):
            data = data.float().to(device)
            targets = target.float().to(device)
            y_hat = model(data)
            loss = criterion_collect[criterion.upper()](y_hat, targets)

            # metric calculation (loss, mse/mae, mape)
            val_loss += loss.item()
            val_metric(y_hat, targets)
            #val_mape(y_hat, targets)
            val_r2(y_hat, targets)
            
    # average validation loss over an epoch
    val_loss = val_loss/ i_batches_val
    # average mae/mse and mape over all training batches
    avg_val_metric = val_metric.compute()
    #avg_val_mape = val_mape.compute()
    avg_val_r2 = val_r2.compute()

    print('Test set: Average loss: {:.4f}, Average_metric:{:.4f}, Average_R2:{:.4f}'.format(val_loss, avg_val_metric, avg_val_r2))


# In[ ]:





# In[18]:


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


# In[14]:


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


# In[20]:


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


# In[21]:


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


# In[19]:


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


# In[28]:


def split_kfold(dataset, k_fold):
    '''
    Arguments:
    dataset: the full dataset
    k_fold: the number of cross-validation fold wants to split
    
    return:
    train_list: all possible training set after split
    test_list: all possible testing set after split
    '''
    # the total dataset 
    total_size = len(dataset)
    # the proportion of the testing data set
    prop = 1/k_fold
    ## containing size for the validation each time (need to be integer)
    vali_size = torch.round(torch.tensor(total_size * prop)) 
    vali_size = vali_size.to(torch.int)
  
    # starting split the test and train data sets
    train_list = []
    vali_list = []

    for i in range(k_fold):
        
    ## splitting vali and train
        ### get the splitting indices for training set 
        train_left = list(range(0,i*vali_size))
        train_right = list(range(i*vali_size + vali_size, total_size))
        train_indices = train_left + train_right
        ### get the splitting indices for testing set
        vali_indices = list(range(i*vali_size, i*vali_size + vali_size))
    ## split the test and train data sets
        train_set = torch.utils.data.dataset.Subset(dataset,train_indices)
        vali_set = torch.utils.data.dataset.Subset(dataset,vali_indices)
        print("The length of the training set is {}".format(len(train_set)))
        print("The length of the training set is {}".format(len(vali_set)))
        train_list.append(train_set)
        vali_list.append(vali_set)

    return train_list, vali_list


# In[29]:


train_list, valid_list = split_kfold(train_set, 5)


# ### Male Vesus Female

# In[ ]:





# In[ ]:





# ### Make Histogram and scatter plot

# In[16]:


import matplotlib.pyplot as plt
import numpy as np


# In[33]:


val_loader = DataLoader(test_set, 128, shuffle=True)


# In[34]:


val_loader


# In[35]:


for i_batches_val, (data, target) in enumerate(val_loader,1):
       print(target.float())


# In[36]:


def test3(model, device, vali_loader):
    #diff_value = []
    #val_loader = DataLoader(valid_set, val_batch_size, shuffle=True)
    for i_batches_val, (data, target) in enumerate(vali_loader,1):
        data = data.float().to(device)
        targets = target.float().to(device)
        y_hat = model(data)
        print(y_hat)
    #return diff_value


# In[17]:


import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:1' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet6 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet6.classifier[1] = nn.Sequential(OrderedDict([
                                          ('se', ECABlock(1536)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet6.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                                    ('fc1',nn.Linear(1536, 500)),
                                                 ('fc2', nn.Linear(500, 200)),
                                                 ('fc3', nn.Linear(200, 1))]))
convnet6 = convnet6.to(device)


# In[39]:


convnet6.load_state_dict(torch.load("/data/alzeye/yuru/cv_convnet5.pt"))
test3(convnet6, device, val_loader)


# In[18]:


# male model (1.0)
for i in range(5):
    train_valid3(convnet6, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(convnet6.state_dict(), "/data/alzeye/yuru/male_convnet.pt")
    print("finished fold {}".format(i+1))


# In[19]:


# male average metrics
avg_mae = (1.1124+0.2950+0.1839+0.1426 + 0.1144)/5
avg_mse = (2.4294+0.1840+0.0715+0.0377+0.0246)/5
avg_r = (0.5994+0.9690+0.9887+0.9940+0.9960)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[20]:


# female model (0.0)
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet7 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet7.classifier[1] = nn.Sequential(OrderedDict([
                                          ('se', ECABlock(1536)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet7.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                                    ('fc1',nn.Linear(1536, 500)),
                                                 ('fc2', nn.Linear(500, 200)),
                                                 ('fc3', nn.Linear(200, 1))]))
convnet7 = convnet7.to(device)


# In[30]:


for i in range(5):
    train_valid3(convnet7, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(convnet7.state_dict(), "/data/alzeye/yuru/female_convnet.pt")
    print("finished fold {}".format(i+1))


# In[31]:


# female average metrics
avg_mae = (1.1447+0.3272+0.2133+0.1563+0.1195)/5
avg_mse = (2.6289+0.2199+0.0914+0.0450+0.0268)/5
avg_r = (0.6453+0.9682+0.9871+0.9937+0.9961)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[ ]:





# ### Ensemble

# In[26]:


### Ensemble (resnet+selayer+cutout+ 3fc) and (convnext+combinev3+cutout+3fc)
import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet.classifier[1] = nn.Sequential(OrderedDict([
                                           ('combine',CBAMBlock_v3("FC", 7, 1536, 16))]))
convnet.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                                    ('fc1',nn.Linear(3072, 500)),
                                                 ('fc2', nn.Linear(500, 200)),
                                                 ('fc3', nn.Linear(200, 1))]))
convnet = convnet.to(device)
se_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
se_resnet.avgpool = nn.Sequential(OrderedDict([('se', SELayer(2048)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,500)),
                                         ('fc3', nn.Linear(500,1))]))

se_resnet = se_resnet.to(device)


# In[31]:


for i in range(5):
    train_valid4(convnet, se_resnet, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(convnet.state_dict(), "/data/alzeye/yuru/ensemble_convnet.pt")
    torch.save(se_resnet.state_dict(), "/data/alzeye/yuru/ensemble_resnet.pt")
    print("finished fold {}".format(i+1))


# In[33]:


# avgerage for validation 
avg_mae = (1.0718 + 0.3661 + 0.2418 + 0.1850 + 0.19625)/5
avg_mse = (2.2757 + 0.2944 + 0.1283 + 0.0827 + 0.0968)/5
avg_r = (0.6610 + 0.9553 + 0.9808 + 0.9878 + 0.9855)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[35]:


test1(convnet, se_resnet, device, 'MAE', test_set, 0.0001, 128)


# ### convnext + cutout + 3 fc layers

# In[20]:


from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
model1 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
model1.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                                    ('fc1',nn.Linear(1536, 500)),
                                                 ('fc2', nn.Linear(500, 200)),
                                                 ('fc3', nn.Linear(200, 1))]))
model1 = model1.to(device)


# In[21]:


for i in range(5):
    train_valid2(model1, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(model1.state_dict(), "/data/alzeye/yuru/origin_cv_convnet1.pt")
    print("finished fold {}".format(i+1))


# In[22]:


for i in range(5):
    train_valid2(model1, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(model1.state_dict(), "/data/alzeye/yuru/origin_cv_convnet1.pt")
    print("finished fold {}".format(i+1))


# In[ ]:


# avgerage for validation 
avg_mae = (1.0599 + 0.3129 + )/5
avg_mse = (2.2651 + 0.2089)/5
avg_r = (0.6649 + 0.9685)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[ ]:


test(convnet2, device, 'MAE', test_set, 0.0001, 128)


# ### convnext + general augmentation + 3 fc layers

# In[19]:


from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
model = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
model.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                                    ('fc1',nn.Linear(1536, 500)),
                                                 ('fc2', nn.Linear(500, 200)),
                                                 ('fc3', nn.Linear(200, 1))]))
model = model.to(device)


# In[20]:


for i in range(5):
    train_valid3(model, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(model.state_dict(), "/data/alzeye/yuru/origin_cv_convnet.pt")
    print("finished fold {}".format(i+1))


# In[ ]:


# avgerage for validation 
avg_mae = ()/5
avg_mse = ()/5
avg_r = ()/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[ ]:


test(convnet2, device, 'MAE', test_set, 0.0001, 128)


# ### ResNet50 + cutout + 3 fc layers

# In[19]:


from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet3 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)     
se_resnet3.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,500)),
                                         ('fc3', nn.Linear(500,1))]))

se_resnet3 = se_resnet3.to(device)


# In[20]:


for i in range(5):
    train_valid2(se_resnet3, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(se_resnet3.state_dict(), "/data/alzeye/yuru/origin_cv_resnet.pt")
    print("finished fold {}".format(i+1))


# In[21]:


# avgerage for validation 
avg_mae = (1.1762 + 0.4599 + 0.3195 + 0.2314 + 0.26423)/5
avg_mse = (2.8536 + 0.4164 + 0.1916 + 0.1008 + 0.1554)/5
avg_r = (0.5744 + 0.9391 + 0.9712 + 0.9852 + 0.9770)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[23]:


import torchmetrics
test(se_resnet3, device, 'MAE', test_set, 0.0001, 128)


# ### ResNet50 + general augmentation + 3 fc layers

# In[19]:


from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet4 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)     
se_resnet4.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,500)),
                                         ('fc3', nn.Linear(500,1))]))

se_resnet4 = se_resnet4.to(device)


# In[20]:


for i in range(5):
    train_valid3(se_resnet4, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(se_resnet4.state_dict(), "/data/alzeye/yuru/origin_cv_resnet1.pt")
    print("finished fold {}".format(i+1))


# In[21]:


# avgerage for validation 
avg_mae = (1.1918 + 0.4391 + 0.2772 +0.2151 + 0.22481)/5
avg_mse = (2.7519 + 0.3896 + 0.1414 + 0.0895 + 0.1112)/5
avg_r = (0.5873 + 0.9419 + 0.9789 + 0.9868 + 0.9834)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[22]:


import torchmetrics
test(se_resnet4, device, 'MAE', test_set, 0.0001, 128)


# ### covnext + selayer + cutout + 3 fc layers

# In[39]:


## convnet_final
import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet2 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet2.classifier[1] = nn.Sequential(OrderedDict([
                                          ('se', SELayer(1536)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet2.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                                    ('fc1',nn.Linear(1536, 500)),
                                                 ('fc2', nn.Linear(500, 200)),
                                                 ('fc3', nn.Linear(200, 1))]))
convnet2 = convnet2.to(device)


# In[21]:


for i in range(5):
    train_valid2(convnet2, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(convnet2.state_dict(), "/data/alzeye/yuru/cv_convnet.pt")
    print("finished fold {}".format(i+1))


# In[29]:


# avgerage for validation 
avg_mae = (1.0481 + 0.3178+0.2014 + 0.1516 + 0.18956)/5
avg_mse = (2.1620 + 0.1998 + 0.0777 + 0.0433 + 0.0816)/5
avg_r = (0.6663 + 0.9708 + 0.9888 + 0.9936 + 0.9879)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[31]:


# test metrics
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
model = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
model.classifier[1] = nn.Sequential(OrderedDict([
                                          ('se', SELayer(1536)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
model.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                                    ('fc1',nn.Linear(1536, 500)),
                                                 ('fc2', nn.Linear(500, 200)),
                                                 ('fc3', nn.Linear(200, 1))]))
model = model.to(device)
model.load_state_dict(torch.load("/data/alzeye/yuru/cv_convnet.pt"))
test(model, device, 'MAE', test_set, 0.0001, 128)


# In[40]:


convnet2.load_state_dict(torch.load("/data/alzeye/yuru/cv_convnet.pt"))
test(convnet2, device, 'MAE', test_set, 0.0001, 128)


# ### covnext + con_selayer + cutout + 3 fc layers

# In[18]:


from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:1' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet3 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet3.classifier[1] = nn.Sequential(OrderedDict([
                                          ('se', ECABlock(1536)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet3.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                                    ('fc1',nn.Linear(1536, 500)),
                                                 ('fc2', nn.Linear(500, 200)),
                                                 ('fc3', nn.Linear(200, 1))]))
convnet3 = convnet3.to(device)


# In[19]:


for i in range(5):
    train_valid2(convnet3, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(convnet3.state_dict(), "/data/alzeye/yuru/cv_convnet1.pt")
    print("finished fold {}".format(i+1))


# In[21]:


# avgerage for validation 
avg_mae = (1.0642 + 0.3353 + 0.1995 + 0.1507 + 0.19487)/5
avg_mse = (2.2419 + 0.2187 + 0.0803 + 0.0431 + 0.0929)/5
avg_r = (0.6689 + 0.9680 + 0.9883 + 0.9936 + 0.9858)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[27]:


import torchmetrics
use_cuda = True
device = torch.device('cuda:1' if torch.cuda.is_available() and use_cuda else 'cpu')
model1 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
model1.classifier[1] = nn.Sequential(OrderedDict([
                                          ('se', ECABlock(1536)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
model1.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                                    ('fc1',nn.Linear(1536, 500)),
                                                 ('fc2', nn.Linear(500, 200)),
                                                 ('fc3', nn.Linear(200, 1))]))
model1 = model1.to(device)
model1.load_state_dict(torch.load("/data/alzeye/yuru/cv_convnet1.pt"))
test(model1, device, 'MAE', test_set, 0.0001, 128)


# In[38]:


import torchmetrics
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
model1 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
model1.classifier[1] = nn.Sequential(OrderedDict([
                                          ('se', ECABlock(1536)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
model1.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                                    ('fc1',nn.Linear(1536, 500)),
                                                 ('fc2', nn.Linear(500, 200)),
                                                 ('fc3', nn.Linear(200, 1))]))
model1 = model1.to(device)
model1.load_state_dict(torch.load("/data/alzeye/yuru/cv_convnet1.pt"))
test(model1, device, 'MAE', test_set, 0.0001, 128)


# ### covnext + combine_v3 + cutout + 3 fc layers

# In[36]:


import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet.classifier[1] = nn.Sequential(OrderedDict([
                                           ('combine',CBAMBlock_v3("FC", 7, 1536, 16))]))
convnet.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                                    ('fc1',nn.Linear(3072, 500)),
                                                 ('fc2', nn.Linear(500, 200)),
                                                 ('fc3', nn.Linear(200, 1))]))
convnet = convnet.to(device)


# In[22]:


for i in range(5):
    train_valid2(convnet, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(convnet.state_dict(), "/data/alzeye/yuru/cv_convnet2.pt")
    print("finished fold {}".format(i+1))


# In[25]:


#train_valid2(convnet, device, 'MAE', train_list[4], valid_list[4], 128, 0.0001,20,128)
torch.save(convnet.state_dict(), "/data/alzeye/yuru/cv_convnet2.pt")


# In[26]:


# avgerage for validation 
avg_mae = (1.0471 + 0.3109 + 0.2017 + 0.1465 + 0.19007)/5
avg_mse = ( 2.1595 + 0.1966 + 0.0778 + 0.0401 + 0.0837)/5
avg_r = (0.6743 + 0.9709 + 0.9886 + 0.9937 + 0.9874)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[28]:


test(convnet, device, 'MAE', test_set, 0.0001, 128)


# In[37]:


convnet.load_state_dict(torch.load("/data/alzeye/yuru/cv_convnet2.pt"))
test(convnet, device, 'MAE', test_set, 0.0001, 128)


# ### covnext + combine_v4 + cutout + 3 fc layers

# In[33]:


import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet4 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet4.classifier[1] = nn.Sequential(OrderedDict([
                                          ('combine',CBAMBlock_v4("FC", 7, 1536, 16))]))
convnet4.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(1536, 500)),
                                          ('fc2', nn.Linear(500, 200)),
                                          ('fc3', nn.Linear(200, 1))]))
convnet4 = convnet4.to(device)


# In[24]:


for i in range(5):
    train_valid2(convnet4, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(convnet4.state_dict(), "/data/alzeye/yuru/cv_convnet3.pt")
    print("finished fold {}".format(i+1))


# In[25]:


# avgerage for validation 
avg_mae = (1.0678 + 0.3282 + 0.2036 + 0.1467 + 0.20046)/5
avg_mse = (2.3336 + 0.2199 + 0.0776 + 0.0417 + 0.1126)/5
avg_r = (0.6573 + 0.9684 + 0.9880 + 0.9936 + 0.9832)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[27]:


test(convnet4, device, 'MAE', test_set, 0.0001, 128)


# In[34]:


convnet4.load_state_dict(torch.load("/data/alzeye/yuru/cv_convnet3.pt"))
test(convnet4, device, 'MAE', test_set, 0.0001, 128)


# ### convnext + selayer + general_augmentation + 3 fc layers

# In[31]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet5 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet5.classifier[1] = nn.Sequential(OrderedDict([
                                          ('se', SELayer(1536)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet5.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                                    ('fc1',nn.Linear(1536, 500)),
                                                 ('fc2', nn.Linear(500, 200)),
                                                 ('fc3', nn.Linear(200, 1))]))
convnet5 = convnet5.to(device)


# In[30]:


for i in range(5):
    train_valid(convnet5, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(convnet5.state_dict(), "/data/alzeye/yuru/cv_convnet4.pt")
    print("finished fold {}".format(i+1))


# In[32]:


# avgerage for validation 
avg_mae = (1.0777 + 0.3147 +  0.1950 + 0.1465 + 0.18637)/5
avg_mse = (2.3844 + 0.2061 + 0.0739 + 0.0398 + 0.0913)/5
avg_r = (0.6537 + 0.9702 + 0.9893 + 0.9937 + 0.9899)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[33]:


test(convnet5, device, 'MAE', test_set, 0.0001, 128)


# In[32]:


convnet5.load_state_dict(torch.load("/data/alzeye/yuru/cv_convnet4.pt"))
test(convnet5, device, 'MAE', test_set, 0.0001, 128)


# ### convnext + conv_layer + general_augmentation + 3 fc layers

# In[29]:


import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet6 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet6.classifier[1] = nn.Sequential(OrderedDict([
                                          ('se', ECABlock(1536)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
convnet6.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                                    ('fc1',nn.Linear(1536, 500)),
                                                 ('fc2', nn.Linear(500, 200)),
                                                 ('fc3', nn.Linear(200, 1))]))
convnet6 = convnet6.to(device)


# In[26]:


for i in range(5):
    train_valid3(convnet6, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(convnet6.state_dict(), "/data/alzeye/yuru/cv_convnet5.pt")
    print("finished fold {}".format(i+1))


# In[27]:


# avgerage for validation 
avg_mae = (1.0651 + 0.2942 + 0.1890 + 0.1456 + 0.18437)/5
avg_mse = (2.2552 + 0.1830 + 0.0676 + 0.0390 + 0.0809)/5
avg_r = (0.6685 + 0.9720 + 0.9901 + 0.9941 + 0.9879)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[28]:


test(convnet6, device, 'MAE', test_set, 0.0001, 128)


# In[30]:


convnet6.load_state_dict(torch.load("/data/alzeye/yuru/cv_convnet5.pt"))
test(convnet6, device, 'MAE', test_set, 0.0001, 128)


# ### convnext + combine_v3 + general_augmentation + 3 fc layers

# In[27]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet7 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet7.classifier[1] = nn.Sequential(OrderedDict([
                                           ('combine',CBAMBlock_v3("FC", 7, 1536, 16))]))
convnet7.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                                    ('fc1',nn.Linear(3072, 500)),
                                                 ('fc2', nn.Linear(500, 200)),
                                                 ('fc3', nn.Linear(200, 1))]))
convnet7 = convnet7.to(device)


# In[30]:


for i in range(5):
    train_valid3(convnet7, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(convnet7.state_dict(), "/data/alzeye/yuru/cv_convnet6.pt")
    print("finished fold {}".format(i+1))


# In[31]:


# avgerage for validation 
avg_mae = (1.0668 + 0.3078 + 0.1995 + 0.1480 + 0.18718)/5
avg_mse = (2.3080 + 0.1921 + 0.0780 + 0.0397 + 0.0825)/5
avg_r = (0.6607 + 0.9706 + 0.9886 + 0.9940 + 0.9877)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[32]:


test(convnet7, device, 'MAE', test_set, 0.0001, 128)


# In[28]:


convnet7.load_state_dict(torch.load("/data/alzeye/yuru/cv_convnet6.pt"))
test(convnet7, device, 'MAE', test_set, 0.0001, 128)


# ### convnext + combine_v4 + general_augmentation + 3 fc layers

# In[33]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet8 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
convnet8.classifier[1] = nn.Sequential(OrderedDict([
                                          ('combine',CBAMBlock_v4("FC", 7, 1536, 16))]))
convnet8.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(1536, 500)),
                                          ('fc2', nn.Linear(500, 200)),
                                          ('fc3', nn.Linear(200, 1))]))
convnet8 = convnet8.to(device)


# In[34]:


for i in range(5):
    train_valid3(convnet8, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(convnet8.state_dict(), "/data/alzeye/yuru/cv_convnet7.pt")
    print("finished fold {}".format(i+1))


# In[24]:


# avgerage for validation 
avg_mae = (1.0540 + 0.2979 + 0.1994 + 0.1490 + 0.18437)/5
avg_mse = (2.2549 + 0.1861 + 0.0753 + 0.0424 + 0.0809)/5
avg_r = (0.6685 + 0.9715 + 0.9890 + 0.9936 + 0.9877)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[26]:


import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
model1 = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
model1.classifier[1] =  nn.Sequential(OrderedDict([
                                          ('combine',CBAMBlock_v4("FC", 7, 1536, 16))]))
model1.classifier[2] = nn.Sequential(OrderedDict([('flatten',nn.Flatten(start_dim=1, end_dim=-1)), 
                                          ('fc1',nn.Linear(1536, 500)),
                                          ('fc2', nn.Linear(500, 200)),
                                          ('fc3', nn.Linear(200, 1))]))
model1 = model1.to(device)
model1.load_state_dict(torch.load("/data/alzeye/yuru/cv_convnet7.pt"))
test(model1, device, 'MAE', test_set, 0.0001, 128)


# ### resnet50 + selayer + cutout + 3 fc layer

# In[17]:


from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
se_resnet.avgpool = nn.Sequential(OrderedDict([('se', SELayer(2048)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,500)),
                                         ('fc3', nn.Linear(500,1))]))

se_resnet = se_resnet.to(device)


# In[21]:


for i in range(5):
    train_valid2(se_resnet, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(se_resnet.state_dict(), "/data/alzeye/yuru/cv_resnet.pt")
    print("finished fold {}".format(i+1))


# In[22]:


# avgerage for validation 
avg_mae = (1.1598 + 0.4572 + 0.3124 + 0.2181 + 0.25392)/5
avg_mse = (2.6718 + 0.4280 + 0.1838 + 0.0920 + 0.1511)/5
avg_r = (0.5859 + 0.9378 + 0.9725 + 0.9860 + 0.9773)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[30]:


import torchmetrics
test(se_resnet, device, 'MAE', test_set, 0.0001, 128)


# ### resnet50 + conv_layer + cutout + 3 fc layer

# In[20]:


from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
se_resnet1.avgpool = nn.Sequential(OrderedDict([('se', ECABlock(2048)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet1.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,500)),
                                         ('fc3', nn.Linear(500,1))]))

se_resnet1 = se_resnet1.to(device)


# In[24]:


for i in range(5):
    train_valid2(se_resnet1, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(se_resnet1.state_dict(), "/data/alzeye/yuru/cv_resnet1.pt")
    print("finished fold {}".format(i+1))


# In[33]:


# avgerage for validation 
avg_mae = (1.1748 + 0.4745 + 0.2986 + 0.2165 + 0.2679)/5
avg_mse = (2.6877 + 0.4721 + 0.1782 + 0.1012 + 0.1972)/5
avg_r = (0.5834 + 0.9314 + 0.9733 + 0.9846 + 0.9700)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[24]:


import torchmetrics
se_resnet1.load_state_dict(torch.load("/data/alzeye/yuru/cv_resnet1.pt"))
test(se_resnet1, device, 'MAE', test_set, 0.0001, 128)


# ### resnet50 + combine_v3 + cutout + 3 fc layer

# In[35]:


from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
se_resnet1.avgpool = nn.Sequential(OrderedDict([('combine',CBAMBlock_v3("FC", 7, 2048, 16)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet1.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(4096,1000)),
                                          ('fc2',nn.Linear(1000,500)),
                                         ('fc3', nn.Linear(500,1))]))

se_resnet1 = se_resnet1.to(device)


# In[36]:


for i in range(5):
    train_valid2(se_resnet1, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(se_resnet1.state_dict(), "/data/alzeye/yuru/cv_resnet2.pt")
    print("finished fold {}".format(i+1))


# In[37]:


# avgerage for validation 
avg_mae = (1.1518 + 0.4841 + 0.2908 + 0.2259 + 0.26032)/5
avg_mse = (2.6151 + 0.4911 + 0.1597 + 0.0985 +0.1545)/5
avg_r = (0.5987 + 0.9291 + 0.9761 + 0.9854 + 0.9770)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[38]:


test(se_resnet1, device, 'MAE', test_set, 0.0001, 128)


# ### resnet50 + combine_v4 + cutout + 3 fc layer

# In[41]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
se_resnet2.avgpool = nn.Sequential(OrderedDict([('combine',CBAMBlock_v4("FC", 7, 2048, 16)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet2.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1000)),
                                          ('fc2',nn.Linear(1000,500)),
                                         ('fc3', nn.Linear(500,1))]))

se_resnet2 = se_resnet2.to(device)


# In[42]:


for i in range(5):
    train_valid2(se_resnet2, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(se_resnet1.state_dict(), "/data/alzeye/yuru/cv_resnet3.pt")
    print("finished fold {}".format(i+1))


# In[43]:


# avgerage for validation 
avg_mae = (1.1593 + 0.6667 + 0.3460 + 0.2469 + 0.30382)/5
avg_mse = (2.6306 + 3.5785 + 0.2471 + 0.1280 + 0.2309)/5
avg_r = (0.5964 + 0.4832 + 0.9630 + 0.9811 + 0.9656)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[44]:


test(se_resnet2, device, 'MAE', test_set, 0.0001, 128)


# ### resnet50 + selayer + general_augmentation + 3 fc layer

# In[25]:


from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet3 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
se_resnet3.avgpool = nn.Sequential(OrderedDict([('se', SELayer(2048)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet3.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,500)),
                                         ('fc3', nn.Linear(500,1))]))

se_resnet3 = se_resnet3.to(device)


# In[27]:


for i in range(5):
    train_valid3(se_resnet3, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(se_resnet3.state_dict(), "/data/alzeye/yuru/cv_resnet4.pt")
    print("finished fold {}".format(i+1))


# In[31]:


# avgerage for validation 
avg_mae = (1.1615 + 0.4841 + 0.2126 + 0.1764 + 0.18328)/5
avg_mse = (2.6821 + 0.4911 + 0.0981 + 0.0703 + 0.0875)/5
avg_r = (0.6106 + 0.9759 + 0.9855 + 0.9893 + 0.9870)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[29]:


import torchmetrics
test(se_resnet3, device, 'MAE', test_set, 0.0001, 128)


# ### resnet50 + conv_layer + general_augmentation + 3 fc layer

# In[17]:


from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet4 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
se_resnet4.avgpool = nn.Sequential(OrderedDict([('se', ECABlock(2048)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet4.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,500)),
                                         ('fc3', nn.Linear(500,1))]))

se_resnet4 = se_resnet4.to(device)


# In[18]:


for i in range(5):
    train_valid3(se_resnet4, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(se_resnet4.state_dict(), "/data/alzeye/yuru/cv_resnet5.pt")
    print("finished fold {}".format(i+1))


# In[21]:


# avgerage for validation 
avg_mae = (1.2070 + 0.4409 + 0.2773 + 0.1932 + 0.20963)/5
avg_mse = (2.8770 + 0.3779 + 0.1432 + 0.0689 + 0.0942)/5
avg_r = (0.5720 + 0.9445 + 0.9784 + 0.9897 + 0.9860)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[20]:


import torchmetrics
test(se_resnet4, device, 'MAE', test_set, 0.0001, 128)


# ### resnet50 + combine_v3 + general_augmentation + 3 fc layer

# In[21]:


from collections import OrderedDict
import torchmetrics
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet5 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
se_resnet5.avgpool = nn.Sequential(OrderedDict([('combine',CBAMBlock_v3("FC", 7, 2048, 16)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet5.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(4096,1000)),
                                          ('fc2',nn.Linear(1000,500)),
                                         ('fc3', nn.Linear(500,1))]))

se_resnet5 = se_resnet5.to(device)


# In[22]:


for i in range(5):
    train_valid3(se_resnet5, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(se_resnet5.state_dict(), "/data/alzeye/yuru/cv_resnet6.pt")
    print("finished fold {}".format(i+1))


# In[23]:


# avgerage for validation 
avg_mae = (1.1863 + 0.4290 +  0.2764 + 0.2055 + 0.22050)/5
avg_mse = (2.7838 + 0.3647 + 0.1494 + 0.0780 + 0.1092)/5
avg_r = (0.5896 + 0.9441 + 0.9785 + 0.9884 + 0.9838)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[24]:


test(se_resnet5, device, 'MAE', test_set, 0.0001, 128)


# ### resnet50 + combine_v4 + general_augmentation + 3 fc layer

# In[20]:


from collections import OrderedDict
import torchmetrics
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet6 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
se_resnet6.avgpool = nn.Sequential(OrderedDict([('combine',CBAMBlock_v4("FC", 7, 2048, 16)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet6.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1000)),
                                          ('fc2',nn.Linear(1000,500)),
                                         ('fc3', nn.Linear(500,1))]))

se_resnet6 = se_resnet6.to(device)


# In[22]:


for i in range(5):
    train_valid3(se_resnet6, device, 'MAE', train_list[i], valid_list[i], 128, 0.0001,20,128)
    torch.save(se_resnet6.state_dict(), "/data/alzeye/yuru/cv_resnet7.pt")
    print("finished fold {}".format(i+1))


# In[13]:


# avgerage for validation 
avg_mae = (1.2036 + 0.5526 + 0.3120 + 0.2396 + 0.26454)/5
avg_mse = (2.8986 + 0.6487 + 0.2100 + 0.1332 + 0.1783)/5
avg_r = (0.5694 + 0.9068 + 0.9683 + 0.9797 + 0.9735)/5
print("Average MAE: {:.4f}, Average MSE: {:.4f}, Average R2: {:.4f}".format(avg_mae, avg_mse, avg_r))


# In[21]:


se_resnet6.load_state_dict(torch.load("/data/alzeye/yuru/cv_resnet7.pt"))
test(se_resnet6, device, 'MAE', test_set, 0.0001, 128)


# In[ ]:


### Ensemble (resnet50 + convnet)


# In[ ]:





# In[19]:


## resnet50 + cutout (unifrom-1)
import torchmetrics
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
resnet.fc = nn.Linear(resnet.fc.in_features, 1)
resnet = resnet.to(device)
train_valid2(resnet, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[16]:


## resnet 50 + cutout (unifrom-5)
import torchmetrics
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
resnet.fc = nn.Linear(resnet.fc.in_features, 1)
resnet = resnet.to(device)
train_valid2(resnet, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[17]:


# save the model 
torch.save(desnet.state_dict(), "/data/alzeye/yuru/Densenet_crop50_128.pt")


# #### EffecientNet

# In[17]:


import torchmetrics
use_cuda = True
device = torch.device('cuda:1' if torch.cuda.is_available() and use_cuda else 'cpu')
effectnet = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
effectnet.classifier[1] = nn.Linear(effectnet.classifier[1].in_features, 1)
effectnet = effectnet.to(device)
train_valid2(effectnet, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[ ]:


# save the model 
torch.save(effectnet.state_dict(), "/data/alzeye/yuru/Effectnet_crop50_128.pt")


# #### ConvxNet_small

# In[17]:


## convnet_small + cutout
import torchmetrics
use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
convnet1 = convnext_small(weights=ConvNeXt_Small_Weights)
convnet1.classifier[2] = nn.Linear(convnet1.classifier[2].in_features, 1)
convnet1 = convnet1.to(device)
train_valid2(convnet1, device, 'MAE', train_set, test_set, 128, 0.0001,100,128)


# In[ ]:





# ## VGG-16, ResNet, Densenet, EfficientNet, Convnext, (Pre-trained True)

# #### resnet50 (pretrained)

# In[44]:


print(resnet1)


# ## Add Attention Layer 

# #### SE-resnet50 alone

# In[17]:


from Model import se_resnet50
import Model


# In[18]:


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


# In[19]:


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


# In[20]:


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


# In[26]:


## combine 2 attention 
class CBAMBlock(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None):
        super(CBAMBlock, self).__init__()
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
        x = self.channel_attention_block(x)
        x = self.spatial_attention_block(x)
        return x


# In[22]:


class CBAMBlock_v2(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None):
        super(CBAMBlock, self).__init__()
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
        return self.channel_attention_block(x)*self.spatial_attention_block(x)


# In[13]:


# save the model 
torch.save(se_resnet.state_dict(), "/data/alzeye/yuru/SEResnet_crop50_128.pt")


# In[24]:


## se_resnet50 (pretrained) + cuout + mae
import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
se_resnet.avgpool = nn.Sequential(OrderedDict([('se', SELayer(2048)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,1))]))

se_resnet = se_resnet.to(device)
train_valid2(se_resnet, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[26]:


## se_resnet50 (pretrained) + cuout + mse
import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
se_resnet1.avgpool = nn.Sequential(OrderedDict([('se', SELayer(2048)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet1.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,1))]))

se_resnet1 = se_resnet1.to(device)
train_valid2(se_resnet1, device, 'MSE', train_set, test_set, 128, 0.01,100,128)


# ### ECABlock_resnet50 (pretrained) + cutout + mae

# In[ ]:


use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
se_resnet2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
se_resnet2.avgpool = nn.Sequential(OrderedDict([('se', ECABlock(2048)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
se_resnet2.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,1))]))


#print(se_resnet1)
se_resnet2 = se_resnet2.to(device)
train_valid2(se_resnet2, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# ### Spatial_resnet50 (pretrained) + cutout + mae

# In[24]:


import torchmetrics
from collections import OrderedDict
use_cuda = True
device = torch.device('cuda:2' if torch.cuda.is_available() and use_cuda else 'cpu')
sp_resnet3 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
sp_resnet3.avgpool = nn.Sequential(OrderedDict([('spatial', Spatial_Attention_Module(7)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
sp_resnet3.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,1))]))


#print(sp_resnet)
sp_resnet3 = sp_resnet3.to(device)
train_valid2(sp_resnet3, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# ### CBAMBlock_resnet50 (pretrained) + cutout + mae

# In[27]:


use_cuda = True
device = torch.device('cuda:3' if torch.cuda.is_available() and use_cuda else 'cpu')
com_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
com_resnet.avgpool = nn.Sequential(OrderedDict([('combine', CBAMBlock("FC", 7, 2048, 16)),
                                          ('pool',nn.AdaptiveAvgPool2d(output_size=1))]))
        
com_resnet.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,1024)),
                                          ('fc2',nn.Linear(1024,1))]))


com_resnet = com_resnet.to(device)
train_valid2(com_resnet, device, 'MAE', train_set, test_set, 128, 0.01,100,128)


# In[28]:


torch.cuda.empty_cache()


# ### CBABlockv2_resnet50(pretrained) + cutout + mae

# In[ ]:





# In[ ]:




