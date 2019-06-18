import numpy as np 
import mat4py 
import pandas as pd 
import torch 
from torch.utils.data import Dataset, DataLoader
import os 
from PIL import Image,ImageDraw
from skimage import io, transform
import sys 
from torchvision import transforms, utils, datasets, models, transforms
from torch.optim import lr_scheduler
import time 
import copy

from datapreprocess import * 

# note that the folder is organized as 
# ../data/
LABEL_PATH = "../data/devkit/cars_train_annos.mat"
SAVE_CSV_PATH = "../data/cars_annos.csv"
ROOT_FOLDER = "../data/cars_train"



### Now data preprocessing is completed 
# we can build the transfer learning model for training 



#construct baseline model 
def train_model(device, model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs=25):
    """ train a model with the given parameters 
    """
    print("start model training... ")
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        scheduler.step()
        running_loss = 0.0
        dataset_sizes = len(train_dataloader)
        
        # Iterate over data to train the model 
        for sample in train_dataloader:
            # Set up the inputs and label in GPU
            image = sample['image'] 
            bbox = sample['bbox'] 
            image = image.type('torch.FloatTensor')
            bbox = bbox.type('torch.FloatTensor')
            image = image.to(device)
            bbox = bbox.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(image)   
            loss = criterion(outputs, bbox)        
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item()  
            
        epoch_loss = running_loss / dataset_sizes

        print('Loss: {:.4f}'.format(epoch_loss))
        
    print("training is complete") 
    return model
    


def train_bbox_model(train= True):
    """ returns the trained bbox model, if train = True, train the model 
    """
    #define the device
    device = torch.device('cuda:0')   
    
    # create the dataset 
    create_labels_csv()
    label_df = read_labels_csv_to_pddf(SAVE_CSV_PATH)
    cardataset = CarDataset(ROOT_FOLDER,label_df, transforms= transforms.Compose([Rescale(256),ToTensor()])) 
    print("finshed loading data")
    #sample  = cardataset[10]
    #draw_bounding_box(sample)
    
    # create a datalaoder
    batch_size = 4 
    TRAIN_RATIO = 0.8
    len_train = int(len(cardataset) * TRAIN_RATIO)
    len_test = len(cardataset) - len_train 
    trainset, testset = torch.utils.data.random_split(cardataset, [len_train, len_test])
    
    # create customized dataloader 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True) 
    
    # load the resnet 18 model 
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    NUMBER_OF_OUTPUTS = 4
    model_ft.fc = torch.nn.Linear(num_ftrs, NUMBER_OF_OUTPUTS)
    model_ft = model_ft.to(device)

    # use the MSE loss for regression task 
    criterion = torch.nn.MSELoss()
    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        
    # gives model the features for running and the scheduler 
    model_ft = train_model(device, model_ft, trainloader, testloader, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=20)
    
    # save the model 
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")
    PATH = os.path.join("saved_models", "bbox_model")
    torch.save(model_ft, PATH)
    print("model saved at: ", PATH)
    

def evaluate_bbox_model():
    """ visulize a batch of data  
    
    """
    # create an evaluation folder if it does not exist already 
    EVAL_FOLDER = "model_eval"
    if not os.path.exists(EVAL_FOLDER):
        os.mkdir(EVAL_FOLDER)
        
    # create the dataset 
    create_labels_csv()
    batch_size = 4 
    label_df = read_labels_csv_to_pddf(SAVE_CSV_PATH)
    cardataset = CarDataset(ROOT_FOLDER,label_df, transforms= transforms.Compose([Rescale(256),ToTensor()])) 
    trainloader = torch.utils.data.DataLoader(cardataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print("finished loading data") 
    
    # Load the model 
    PATH = os.path.join("saved_models", "bbox_model")
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    NUMBER_OF_OUTPUTS = 4
    model_ft.fc = torch.nn.Linear(num_ftrs, NUMBER_OF_OUTPUTS)
    model_ft = torch.load(PATH)
    # eval refreshes the model 
    model_ft.eval()  
    
    
    # randomly choose a image 
    def pred(sample,model):
        image = sample['image']
        device = torch.device('cuda:0')   
        image = image.type('torch.FloatTensor') 
        image = image.to(device)
        model = model.to(device)
        outputs = model(image)
        return outputs 
    
    # draw the output 
    batch = next(iter(trainloader)) 
    outputs = pred(batch, model_ft)
    bboxes = batch['bbox']
    #print(outputs, bboxes)
    
    images = batch['image'] 
    for i in range(len(images)):
        image = images[i]
        # get the prediction from the outputs 
        pred_bbox = outputs[i]
        draw_bounding_box(image, pred_bbox, EVAL_FOLDER +"/" + str(i)+".png")
   

    
def main():
    train_bbox_model()
    #evaluate_bbox_model()
    
    
    
    
if __name__ == "__main__":
    main() 
    
    
    
    
    
    