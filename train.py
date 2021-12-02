import numpy as np
import pandas as pd
import glob
import math
import os
import onnx
import argparse
import torch
from tqdm import tqdm
import random
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def count_files(dir = '.'):
    folder = os.listdir(dir)
    file_count = 0
    classes = os.listdir(os.path.join(dir,folder[0]))
    for cat in folder:
        files = glob.glob(os.path.join(dir,cat,'*/*'))
        print(cat,'Images:',len(files), end = ', ')
    print('\n')
    for direc in classes:
        files = glob.glob(os.path.join(dir,'*',direc,'*'))
        file_count += len(files)
        print(direc, len(files))
    print("Total Images:",file_count)

def get_dataloaders(data_dir = '.', batch_size = 32, num_workers = 2):
    folder = os.listdir(data_dir)
    for i in folder:
        if 'train' in i.lower():
            train = i
        elif 'valid' in i.lower():
            valid = i
        elif 'test' in i.lower():
            test = i
    random.seed(41)

    #Data Tranforms (Augmentation and Normalization)
    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(0.6),
                                        transforms.Resize(size=(224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])
    val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(0.6),
                                        transforms.Resize(size=(224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(size=(224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])

    #Getting all the data with PyTorch Datasets
    train_data = datasets.ImageFolder(data_dir + '/'+train, transform= train_transforms)
    val_data = datasets.ImageFolder(data_dir + '/'+valid, transform= val_transforms)
    test_data = datasets.ImageFolder(data_dir + '/'+test, transform= test_transforms)

    #Loading the data into PyTorch DataLoader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size= batch_size, shuffle = True,num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size= batch_size, shuffle = True,num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size= batch_size, shuffle = True,num_workers=num_workers)

    #Creating a dictionary of all classes
    classes = dict(zip(list(range(len(train_data.classes))),train_data.classes))
    print("# DataLoaders are ready")
    return train_loader, valid_loader, test_loader, classes


def load_model(classes=[0,0,0,0], model_dir = None, cuda_no = 0):

    print("# Loading Model")
    model = models.resnet50(pretrained=True)
    # Freezing all the layers
    for param in model.parameters():
        param.requires_grad = False

    # Changing the Classifier
    model.fc = nn.Sequential(nn.Linear(2048,1024),
                            nn.ReLU(),
                            nn.Dropout(p=0.4),
                            nn.Linear(1024,512),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(512,128),
                            nn.ReLU(),
                            nn.Dropout(p=0.4),
                            nn.Linear(128,len(classes)))

    # Making the Classifier layer Trainable                           
    for param in model.fc.parameters():
        param.requires_grad = True

    if model_dir is not None:
        print("# Loading model weights")
        model.load_state_dict(torch.load(model_dir))
    print("# Model Loaded")
    # Moving the model to device
    return model.cuda(cuda_no)

def train(model, train_loader, valid_loader, lr = 0.01, step_size = 4, gamma = 0.35, epochs = 40, check_every = 5, print_every = 40 , cuda_no = 0): 
    criterion = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(model.fc.parameters(),lr = lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma = gamma)

    steps = 0
    train_losses, valid_losses,valid_acc = [], [], [] # List keeping track of losses and accuracy to plot later
    valid_loss_min = np.Inf # It will be used to save model whenever Vallidation loss decreases
    valid_acc_min = 0.0 
    print("Started Training")
    for e in tqdm(range(epochs)):
    
        train_loss = 0 
        model.train()
    #train the model
        for images, labels in train_loader:
            steps+=1
        # Move tensor to device('cuda' in case of GPU or 'cpu' in case of CPU)
            images, labels = images.cuda(cuda_no), labels.cuda(cuda_no)
        # Clearing all the previous gradients
            optimizer.zero_grad()
        # Forward Pass
            logits = model(images)
        # Loss calculation
            loss = criterion(logits,labels)
        # Backward Pass
            loss.backward()
        # Update the parameters
            optimizer.step()
        # Updating the losses list
            train_loss += loss.item()

        # Evaluating after specific amount of steps
            if steps % check_every == 0:
                valid_loss = 0
                accuracy = 0
        # Setting Model to Evaluation Mode
                model.eval()
                with torch.no_grad():
            # Getting Validation loss
                    for images, labels in valid_loader:
                        images, labels = images.cuda(cuda_no), labels.cuda(cuda_no)
                        logits = model(images)
                        batch_loss = criterion(logits,labels)
                        valid_loss += batch_loss.item()
            
            # Calculating Accuracy
                        output = F.softmax(logits,dim=1)
                        top_p,top_class = output.topk(1,dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        # Printing stats    
                    if steps%print_every==0:
                        print(f"Epoch {e+1}/{epochs}.. "
                        f"Train loss: {train_loss/check_every:.3f}.. "
                        f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                        f"Validation accuracy: {accuracy/len(valid_loader):.3f}.. "
                        f"LR : {scheduler.get_lr():}"
                        )
                    valid_loss = valid_loss/len(valid_loader)
                    train_losses.append(train_loss/check_every)
                    valid_losses.append(valid_loss)
                    valid_acc.append(accuracy/len(valid_loader))
        
        # Checking if Validation loss decreased
                    if valid_loss <= valid_loss_min:
            
            # if decreased, it will save the model
                        print('valid loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        valid_loss_min,
                        valid_loss))
                        torch.save(model.state_dict(), 'model.pt')
                        valid_loss_min = valid_loss
        
        
    # Scheduler performing a step to change learning rate of Optimizer    
        scheduler.step()


    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.savefig('Training and Validatioin Loss Plot.png')

    plt.plot(valid_acc, label='Validation Accuracy')
    plt.legend(frameon=False)
    plt.savefig('Validatioin Accuracy Plot.png')


if __name__ == "__main__":
    searchterms = []
    parser = argparse.ArgumentParser(description='Train Resnet50')
    parser.add_argument('-d', '--dataset', default='.', type=str, help='Dataset Directoy')
    parser.add_argument('-m', '--model', default=None, type=str, help='Model Directory')
    parser.add_argument('-g', '--gpu', default=0, type=int, help='GPU Device number')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch Size of DataLoaders')
    parser.add_argument('-nw', '--num_workers', default=2, type=int, help='Number Of Workers in DataLoaders')
    parser.add_argument('-e', '--epochs', default=40, type=int, help='Number of epochs to train for')
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='Initial Learning Rate')
    parser.add_argument('-gm', '--gamma', default=0.35, type=float, help='Gamma value for LR Scheduler')
    parser.add_argument('-ss', '--step_size', default=4, type=int, help='Step Size value for LR Scheduler')
    parser.add_argument('-ce', '--check_every', default=5, type=int, help='Check the performance of model after every specified number of steps using validation set')
    parser.add_argument('-pe', '--print_every', default=0, type=int, help='Print the performance of model after every specified number of steps')
    args = parser.parse_args()

    count_files(dir = args.dataset)
    train_loader, valid_loader, test_loader, classes = get_dataloaders(data_dir = args.dataset,batch_size = args.batch_size, num_workers = args.num_workers)
    model = load_model(model_dir = args.model, cuda_no = args.gpu, classes = classes)
    train(model, train_loader, valid_loader, lr = args.learning_rate, step_size = args.step_size, gamma = args.gamma, epochs = args.epochs,check_every = args.check_every, print_every = args.print_every , cuda_no = args.gpu)    
    print('\n\nTraining Completed!')