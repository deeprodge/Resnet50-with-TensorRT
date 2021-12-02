import argparse
import torch
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import torch.nn.functional as F



def preprocess_image(img_path, device = 'cpu'):
    # transformations for the input data
    transformss =transforms.Compose([transforms.Resize(size=(224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    
    # read input image
    input_img = Image.open(img_path)
    # do transformations
    input_data = transformss(input_img)
    batch_data = torch.unsqueeze(input_data, 0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_data = batch_data.to(device)
    return batch_data

def postprocess_output(output):
    classes = {0: 'Glioma', 1: 'Meningioma', 2: 'No Tumor', 3: 'Pituitary'}
    ps = F.softmax(output, dim =1 )
    return classes[int(torch.argmax(ps).item())]