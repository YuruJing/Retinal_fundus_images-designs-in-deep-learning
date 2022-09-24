import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
from PIL import Image
import numpy as np 
import math
import matplotlib.pyplot as plt
import pandas as pd
import torchmetrics 
import copy



def extract_images(data, cols, eye_label, images_path):
    """
    data: the csv data files contain all labels
    cols: picked the columns of the data you need (list of strings) [need to have the file_name column at least]
    eye_label: which the eye images you want (L or R):left or right (string)
    images_path: images' path 
    """
    # link the csv labels with images data
    extract_csv_images = data[cols][data["laterality"] == eye_label.upper()]
    extract_images_name = os.listdir(images_path)
    common_images = list()
    for i, values in extract_csv_iterrows():
        if values['file_name'] in extract_images_name:
            common_images.append(i)

    # extract and save the common images' file name
    image_folder = images_path.split('/')[-2]
    with open(''.join([eye_label, '_', image_folder, '.txt']), 'w') as f:
        for i in common_images:
            f.write("%s\n" % i)
        print('Done')
    return common_images
    
# calculate the proper normalized mean and std for all images 
def get_mean_sd(data, batch_size, num_workers=0):
    """
    data: All of the current images files (custom_dataset)
    batch_size: Setting the batch_size for the data loader 
    num_workers: Setting the number of workers 
    """
    # var(x) = E(x^2) - [E(x)]^2
    # initialization
    features_sum, features_sqsum, num_batchs = 0,0,0
    # get the data loader
    data_loader = DataLoader(data, batch_size = batch_size, num_workers = num_workers, shuffle=True)
    for data, _ in data_loader:
        features_sum += torch.mean(data, dim=[0,2,3])
        features_sqsum += torch.mean(data**2, dim=[0,2,3])
        num_batchs += 1

        mean = features_sum/num_batchs
        std = (features_sqsum/num_batchs - mean**2)**0.5

        return mean, std

# split data (train_test_split)
def train_test_split(full_data, test_ratio, train_trans, test_trans, workers):
    """
    full_data: The full image dataset
    test_ratio: The proportion of test set
    trans: Choose the proper transformation for the data (used calculated mean and std to normalised images)
    workers: Setting the number of workders for the loaders
    """
    test_len = int(len(full_data) * test_ratio)
    train_len = int(len(full_data) - test_len)
    
    train_set, test_set = random_split(full_data, [train_len, test_len])
    train_set.transform = train_trans
    test_set.transform = test_trans
    
    return train_set, test_set


# create the data set
class Custom_Dataset(Dataset):
    import numpy as np
    def __init__(self, directory_path, files, files_target): #, target_string
        super().__init__()
        """
        Custom Dataset designed for training later.
        directory_path: (string) Path of the directory containing images folder  
        files: The file name's collections
        files_target: The files name's matched target values list
        target_string: The chosen eye's spherical equivalent (string)
        """
        self.directory = directory_path #./data/alzeye/ukbb/
        self.files = files
        self.target_list = files_target
        self.transform = transforms.ToTensor()
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        images = Image.open(os.path.join(self.directory, self.files[index])) # open the image path
        trans_image = self.transform(images)  # transform image (to tensor and normalize)
        target = torch.from_numpy(np.array([self.target_list[index]]))
        
        return trans_image, target 




# train and validate the model 

### For GoogleNet trainning ###

def googlenet_train_valid(model, device, criterion, train_set, valid_set, train_batch_size, learning_rate, num_epochs,
                val_batch_size, schedule=False):
    
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
            data = data.float().to(device)
            targets = target.float().to(device)  
            predict = model(data)[2]
            optimizer.zero_grad()
            loss = criterion_collect[criterion.upper()](predict, targets)
            loss.backward()
            optimizer.step()

            # metric calculation (loss, mse/mae, mape)
            running_loss += loss.item()  # total loss of the current batch(not averaged)
            train_metric(predict, targets)
            train_mape(predict, targets)
            
            if schedule == True:
                scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
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
                data = data.float().to(device)
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


### For Other models' trainning ###
def train_valid(model, device, criterion, train_set, valid_set, train_batch_size, learning_rate, num_epochs,
                val_batch_size, schedule=False):
    
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
                data = data.float().to(device)
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



# load the model and for test set


