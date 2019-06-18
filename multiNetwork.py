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
import torch.nn.functional as F

from datapreprocess import * 



""" ------------------ 
This file defines a basic multi task network to solve several tasks at the same time 
The basic network(hard shared parameters are going to be the Resnet18) --> Transfer learning is used here to accelerate training process 
The initial loss is simply the sum of the two objectives 
Smaller network are going to be defined based with a few FC layers 


One concern is the design of the multi task network --> how to implement an efficient structure 
----------------------"""






class MultiTaskNetwork(torch.nn.Module):
    def __init__(self, num_classes):
        super(MultiTaskNetwork, self).__init__()
        # the base structure of the network is the RES18 
        self.res18 = models.resnet18(pretrained=True)
        
        #reduce the number of output features -> currently 1000 since trained on imagenet 
        #num_outftrs = 500 
        #model_ft.fc = torch.nn.Linear(num_ftrs, num_outftrs)
        
        # define two subnetworks 
        # subnetwork1 --> Bbox network 
        self.bbox_fc1 = torch.nn.Linear(1000, 512)
        self.bbox_fc2 = torch.nn.Linear(512, 128)                
        self.bbox_fc3 = torch.nn.Linear(128, 4)
                
        # subnetwork2 --> 
        self.class_fc1 = torch.nn.Linear(1000, 512)
        self.class_fc2 = torch.nn.Linear(512, num_classes)                
        
    def return_parameter_lists(self):
        shared_params = [ para for para in self.res18.parameters()]
        bbox_params = [] 
        class_params = [] 
        
        for name, param in self.state_dict().items():
            if "bbox" in name:
                bbox_params.append(param)
            elif "class" in name:
                class_params.append(param)
                
        return shared_params, bbox_params, class_params
        
        
        
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        
        base_outputs = self.res18(x)
        # go through the bbox model 
        bx = F.relu(base_outputs)
        bx = self.bbox_fc1(bx) 
        bx = self.bbox_fc2(bx)
        bx = self.bbox_fc3(bx)

        cx = self.class_fc1(F.relu(base_outputs))
        cx = self.class_fc2(cx)

        return bx, cx
    
    
def train_multitask_network(dataloader):
    device = torch.device('cuda:0')   
    #define the model 
    multimodel = MultiTaskNetwork(10)
    multimodel = multimodel.to(device)
    
    # currently only train two tests
    bbox_criterion = torch.nn.MSELoss()
    classification_criterion = torch.nn.CrossEntropyLoss()
    
    # get parameters 
    shared_params, bbox_params, class_params = multimodel.return_parameter_lists()
    
    bbox_optimizer = torch.optim.SGD(shared_params+bbox_params , lr=0.01, momentum=0.9)
    class_optimizer = torch.optim.SGD(shared_params+class_params, lr=0.001, momentum=0.9)
    
    
    NUM_EPOCH = 10 
    for i in range(NUM_EPOCH):
        print("EPOCH " +str(i) +"/"+ str(NUM_EPOCH) ) 
        print("-"*15)
        bbox_running_loss = 0.0 
        class_running_loss = 0.0 
        
        for sample in dataloader:
            bbox_optimizer.zero_grad()
            class_optimizer.zero_grad() 
            
            image = sample['image'] 
            label = sample['label'] 
            label = label.long()
            bbox = sample['bbox'] 
            
            image = image.to(device)
            label = label.to(device)
            bbox = bbox.to(device)
            
            bbox_out, class_out = multimodel(image)
            
            bbox_loss = bbox_criterion(bbox_out,bbox )
            class_loss = classification_criterion(class_out, label)
            
            bbox_running_loss += bbox_loss 
            class_running_loss += class_loss
            
            # update parameters 
            bbox_loss.backward(retain_graph=True)
            bbox_optimizer.step()
            bbox_optimizer.zero_grad()
            
            # do i need zero_grad again ? 
            class_loss.backward()
            class_optimizer.step()
        
        # compute the average loss 
        bbox_running_loss = bbox_running_loss/len(small_dataloader)
        class_running_loss = class_running_loss/len(small_dataloader)
        
        print("the current epoch loss is bbox loss: {:.4f} and classification loss: {:.4f}".format(bbox_running_loss, class_running_loss))

    # save the model parameters 
    
    
    
def visualize_model_performance():
    # generate a small car dataset 
    num_classes = 10 
    batch_size = 4 
    small_dataset = generate_small_cardataset(10) 
    small_dataloader = torch.utils.data.DataLoader(small_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print("finished loading data.")
    
    train_multitask_network(small_dataloader)
    
    # save the model 
    # test the model on the original dataset 
    
      

            
       
    


            

    
    
if __name__ == "__main__":
    train_multitask_network()
    
    
    
    
    
    
    
    
    
    
    
    