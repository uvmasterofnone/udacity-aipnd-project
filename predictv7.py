'''
This script
 - Loads the pre-trained model from the checkpoint path passed through the argsparser
 - Predicts and prints the top 5 probabilities and the corresponding label values

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


# Define argparse
ap = argparse.ArgumentParser(description='Predict.py')

# Use GPU for prediction
ap.add_argument('--gpu', dest="gpu", action="store", default="cuda",help="Enter training model option. Accepted options include cpu and cuda")
# pass arguments for prediction
ap.add_argument('--input_img', default='/home/workspace/aipnd-project/flowers/test/50/image_06297.jpg', nargs='*', action="store", type = str,help="Image imput for prediction")
ap.add_argument('--checkpoint', default='/home/workspace/aipnd-project/checkpoint.pth', nargs='*', action="store",type = str,help="Checkpoint path")
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int, help= "top 5 probabilities")
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json',help="File that has the flower lables")
# ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/",help =" Data directory")

pa = ap.parse_args()
image_path = pa.input_img
outputs = pa.top_k
# ref : https://pytorch.org/docs/stable/cuda.html#torch.cuda.is_available
if pa.gpu == "cuda":
    # power = (torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    power = "cuda" if torch.cuda.is_available() else "cpu"
elif pa.gpu == "cpu":
    power = pa.gpu
else:
     print ("Sorry, you have entered an incorrect argument for training option.Accepted options include cpu and gpu")
     exit()
input_img = pa.input_img
path = pa.checkpoint
print(type(path))
# datadir = pa.data_dir
catfile = pa.category_names
print("Checkpoint-1: argsparse set up completed"+ '\n')

# trainloader,validationloader,testingloader = imgclassutilv3.load_data(datadir)
# training_data,validation_data,testing_data = imgclassutilv3.load_dataset(datadir)
# print("Checkpoint -2 Completed data loaders and datasets" +'\n')

#load checkpoint
model = imgclassutilv3.load_checkpoint_model(path)
# model = imgclassutilv3.load_checkpoint_model(path,map_location=lambda storage, loc: storage)
if power == "cuda":
    model.to('cuda')
else:
    model.to('cpu')
print("Checkpoint-2: Checkpoint model successfully loaded " + '\n')

# open the category file
with open(catfile, 'r') as json_file:
    cat_to_name = json.load(json_file)


print("Execution sattus : Starting prediction" '\n')
probs,classes = imgclassutilv3.predict(image_path,model,outputs)
top_probs,top_classes = probs.cpu().numpy().tolist()[0],classes.cpu().numpy().tolist()[0]
print("Checpoint-3: Prediction Completed!")

print('\n'"Top Probabilities:" , top_probs)
print("Class idx of top probabilities:", top_classes)

idx_to_class = {val: key for key, val in model.class_to_idx.items()}
top_classes = [idx_to_class[each] for each in top_classes]

labels = []
for class_idx in top_classes:
    labels.append(cat_to_name[class_idx])
print('\n'" Prediction Results - Final")
i=0
while i< pa.top_k:
    print('\n'" Probability: {}  Category: {}". format(top_probs[i],labels[i]))
    i += 1

print('\n' " It's all done now!!!!")
   
          
 