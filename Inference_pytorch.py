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
from utils import preprocess_image, postprocess_output
from train import load_model


if __name__ == "__main__":
    searchterms = []
    parser = argparse.ArgumentParser(description='Create Inference from PyTorch Model')
    parser.add_argument('-i', '--image_dir', default='test_image.jpg', type=str, help='Image Directory')
    parser.add_argument('-m', '--model_dir', default='model.pt', type=str, help='Model Directory')
    args = parser.parse_args()

    image_dir = args.image_dir
    model_dir = args.model_dir
    image = preprocess_image(image_dir)
    model = load_model(model_dir = model_dir)
    output = postprocess_output(model(image))
    print("# The Image Contains:",output)