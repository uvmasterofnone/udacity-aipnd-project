'''

This scripts 
 - accepts the arguments from the user 
 - loads image classification utility script ( imageclassutil.py)
 - trains the model
 - saves the modelcheckpoint at the desired file path
 
 '''


# import as required
import sys
import os
import json
import argparse

import numpy as np
from collections import OrderedDict

import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
import seaborn as sb
import imgclassutilv3

ap = argparse.ArgumentParser(description='Train.py')


# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/",help=" Directory for the image data")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth", help= " Path where model checkpoint is saved")
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json',help="File that has the flower lables")
# ap.add_argument('--catfile_path', dest="catfile_path", action="store", default="/home/workspace/aipnd-project/cat_to_name.json", help ="File that has flower labels")
ap.add_argument('--catfile_path', dest="catfile_path", action="store", default="./cat_to_name.json", help ="File that has flower labels")
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str,help= "Enter the desired architecture. Only vgg16 and densenet121 architectures are accepted")

# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001, help= "Learning Rate")
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=512, help= "Hidden Units")
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=20, help= "# of epochs")
ap.add_argument('--gpu', dest="gpu", action="store", default="cuda",help="Enter training model option. Accepted options include cpu and cuda")

pa = ap.parse_args()
datadir = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
catfilepath = pa.catfile_path
catfile = pa.category_names
                
# obtain user input for architecture
if pa.arch == "vgg16":
    arch = pa.arch
elif pa.arch == "densenet121":
    arch = pa.arch
else:
    print ("Sorry, you have entered an incorrect argument for architecture.Only vgg16 and densent121 architectures are supported")
    exit()


hiddenunits = pa.hidden_units
# power = pa.gpu
if pa.gpu == "cuda":
    # power = (torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    if torch.cuda.is_available():
        power = "cuda" 
    else:
        power = "cpu"
elif pa.gpu == "cpu":
    power = "cpu"
else:
    exit()

epochs = pa.epochs

# Load data
trainloader,validationloader,testingloader = imgclassutilv3.load_data(datadir)
training_data,validation_data,testing_data = imgclassutilv3.load_dataset(datadir)
print("Checkpoint -1 Completed data loaders and datasets" +'\n')

# Open category file 
cat_to_name,output_size = imgclassutilv3.opencatfile(catfilepath)

# Get model
print(" Model will trained using ", power +'\n')
model,optimizer,criterion,output_size = imgclassutilv3.nn_setup(arch,hiddenunits,lr,power,output_size)
print("Checkpoint -2 Model set up completed" +'\n')

# train 
imgclassutilv3.train_network(model, optimizer, criterion, epochs, trainloader,validationloader,power)
print("Checkpoint -3 Model training completed" +'\n')

# save checkpoint
model.class_to_idx = training_data.class_to_idx

model_state = {
    'epoch': epochs,
    'state_dict': model.state_dict(),
    'optimizer_dict': optimizer.state_dict(),
    'classifier': model.classifier,
    'class_to_idx': model.class_to_idx,
    'power': model.to(power),
    'arch': arch
}
print("Checkpoint -4 completed. Model paramaters are saved in model_state" +'\n')

imgclassutilv3.save_checkpoint(model_state,path)
print("Checkpoint -5 completed. checkpoint saved successfully" +'\n')

print(" ALL MODULES SUCCESSFULLY COMPLETED!")