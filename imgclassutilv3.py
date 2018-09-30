
# coding: utf-8

'''
This script provides the following utilies for image classification
 - Data transformation
 - Data Loading
 - Model creation
 - Model training
 - Saving model checkpoint
 - Image processing
 - Image Prediction
 The utilities all functions and call relevant function invoke the requisite utility
 
 '''
 

# Imports here

import json
import numpy as np

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import torch.nn.functional as F
import torchvision

from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from collections import OrderedDict
from PIL import Image
import seaborn as sb
from matplotlib.ticker import FuncFormatter

# load, split ( training,validation,testing ) and tranform datasets
def load_data(datadir = "./flowers" ):

    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    max_image_size = 224
    batch_size = 32
    
    training_transform =  transforms.Compose([transforms.RandomHorizontalFlip(p=0.25),
                                        transforms.RandomRotation(25),
                                        transforms.RandomGrayscale(p=0.02),
                                        transforms.RandomResizedCrop(max_image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means, std)])

    validation_transform =  transforms.Compose([transforms.Resize(max_image_size + 1),
                                          transforms.CenterCrop(max_image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means,std)])

    testing_transform = transforms.Compose([transforms.Resize(max_image_size + 1),
                                       transforms.CenterCrop(max_image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize(means, std)])

    #  Load the datasets with ImageFolder

    training_data = datasets.ImageFolder(train_dir, transform=training_transform)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transform)
    testing_data = datasets.ImageFolder(test_dir, transform=testing_transform)

    #Using the image datasets and the tranforms, define the dataloaders

    trainloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)
    testingloader = torch.utils.data.DataLoader(testing_data, batch_size=64, shuffle=True)

    return trainloader , validationloader, testingloader

def load_dataset(datadir = "./flowers" ):


    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    max_image_size = 224
    batch_size = 32


    
    training_transform =  transforms.Compose([transforms.RandomHorizontalFlip(p=0.25),
                                        transforms.RandomRotation(25),
                                        transforms.RandomGrayscale(p=0.02),
                                        transforms.RandomResizedCrop(max_image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means, std)])

    validation_transform =  transforms.Compose([transforms.Resize(max_image_size + 1),
                                          transforms.CenterCrop(max_image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means,std)])

    testing_transform = transforms.Compose([transforms.Resize(max_image_size + 1),
                                       transforms.CenterCrop(max_image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize(means, std)])

    # TODO: Load the datasets with ImageFolder

    training_data = datasets.ImageFolder(train_dir, transform=training_transform)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transform)
    testing_data = datasets.ImageFolder(test_dir, transform=testing_transform)
    
    return training_data,validation_data,testing_data

def opencatfile(catfile): 
    
    with open(catfile, 'r') as f:
        cat_to_name = json.load(f)
    print("Execution Status:  category names in JSON files were read and loaded in cat_to_name variable"+'\n')
    # Get model Output Size = Number of Categories
    output_size = len(cat_to_name)
    return cat_to_name,output_size

# model, optimizer, criterion = imgclassutilv2.nn_setup(arch,hiddenunits,lr,power,output_size)
def nn_setup(arch,hiddenunits,lr,power,output_size):
    # Using pre-trained model
    # model = models.arch(pretrained=True)
    model = getattr(torchvision.models, arch)(pretrained=True)
    #print(nn_model)

    # check input feature size of the model and assign the same
    # input_size = model.classifier[0].in_features
    if arch == "vgg16":
       input_size = model.classifier[0].in_features
    elif arch == "densenet121":
       input_size = model.classifier.in_features
    hidden_size = [(input_size//8),(input_size//32)]
    output_size = output_size
    

    # Prevent backpropagation on parameters
    for param in model.parameters():
        param.requires_grad = False 

    # Create nn.Module with Sequential using an OrderedDict

    classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_size[0])),
            ('relu1', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.15)),
            ('fc2', nn.Linear(hidden_size[0], hidden_size[1])),
            ('relu2', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.15)),
            ('output', nn.Linear(hidden_size[1], output_size)),
            # LogSoftmax is needed by NLLLoss criterion
            ('softmax', nn.LogSoftmax(dim=1))
        ]))

    # Replace classifier
    model.classifier = classifier
    #print(nn_model)
    # The negative log likelihood loss as criterion.
    criterion = nn.NLLLoss()
    # Choosing optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    # model.cuda()
    model.to(power)
            
    return model, criterion, optimizer,output_size

# imgclassutilv3.train_network(model, optimizer, criterion, epochs, trainloader,power)
def train_network(model, optimizer,criterion, epochs,trainloader,validationloader, power):

    epochs = 3
    print_every = 40
    steps = 0
    loss_show=[]
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), 0.001)
    
    # change to requisite training power ( cpu or cuda )
    if power == "cuda":
        model.to('cuda')
        print(" model.to(cpu)"+ '\n')
    else:
        model.to('cpu')
        print(" model.to(cpu)"+ '\n')
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            if power == "cuda":
                if torch.cuda.is_available():
                    inputs,labels= inputs.to('cuda') , labels.to('cuda')
                    # print('\n'"Have set inputs and labels to cuda") 
                else: 
                    inputs,labels= inputs.to('cpu') , labels.to('cpu') 
                # model.to('cuda - remove this
            else: 
                inputs,labels= inputs.to('cpu') , labels.to('cpu') 
                # print('\n'"Have set inputs and labels to cpu") 
                # model.to('cpu') - remove this
                
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validationloss = 0
                accuracy=0


                for ii, (inputs_val,labels_val) in enumerate(validationloader):
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        inputs_val, labels_val= inputs_val.to('cuda') , labels_val.to('cuda')
                        model.to('cuda')
                    else:
                        inputs_val, labels_val= inputs_val.to('cpu') , labels_val.to('cpu')
                        model.to('cpu')
                    with torch.no_grad():    
                        outputs = model.forward(inputs_val)
                        validationloss = criterion(outputs,labels_val)
                        ps = torch.exp(outputs).data
                        equality = (labels_val.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                validationloss = validationloss / len(validationloader)
                accuracy = accuracy /len(validationloader)

                print("Epoch: {}/{} ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss {:.4f}".format(validationloss),
                       "Accuracy: {:.4f}".format(accuracy))

                running_loss = 0

# validate(testingloader)            
def validate(testingloader):    
    correct = 0
    total = 0
    if power == "cuda":
        nn_model.to('cuda:0')
    else:
        nn_model.to('cpu')
    # nn_model.to('cuda')
    
    with torch.no_grad():
        for data in testingloader:
            images, labels = data
            images= images.to('cuda')
            labels =labels.to('cuda')
            outputs = nn_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy: %d %%' % (100 * correct / total))
    
# imgclassutilv2.save_checkpoint(path,arch,hiddenunits,lr)
def save_checkpoint(model_state,path):
    torch.save(model_state, path)
    print("Execution satus: save checkpoint function executed successfuly and will return "+ '\n')   

# imgclassutilv2.load_checkpoint_model(path)
def load_checkpoint_model(path):
    # Ref : https://discuss.pytorch.org/t/problem-loading-model-trained-on-gpu/17745 
    # was getting an error when trying to predict using cpu. Amended the torch.load() function as suggesed in the above link and seem to have solved the issue    
    # model_state = torch.load(path)
    model_state = torch.load(path, map_location=lambda storage, loc: storage) 
    model = getattr(torchvision.models, model_state['arch'])(pretrained=True)
    model.classifier = model_state['classifier']
    model.class_to_idx = model_state['class_to_idx']
    model.load_state_dict(model_state['state_dict'])
        
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # section below is completed
    img_pil = Image.open(image_path)
   
    transform_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform_image(img_pil)
    
    return img_tensor

# probabilities = imgclassutilv2.predict(image_path, model, outputs, power)
def predict(image_path, model, topk=5):  
    
    model.to('cuda:0')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)


