
from train import load_model
from Inference_trt import load_engine
import numpy as np
import pandas as pd
import glob
import time
import math
import os
import onnx
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import torch
import tensorrt as trt
from tqdm import tqdm
import random
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def benchmark( model_or_engine, context = None,typee='pytorch', input_shape=(1, 3, 224, 224), dtype='fp32', nwarmup=50, nruns=100):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()

    print("  Warm up ...")
    if typee.lower()=='pytorch':
        with torch.no_grad():
            for _ in range(nwarmup):
                features = model_or_engine(input_data)
        torch.cuda.synchronize()
    elif typee.lower()=='trt':
        engine = model_or_engine
        for binding in engine:
            if engine.binding_is_input(binding):
                input_shape = engine.get_binding_shape(binding)
                input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize
                device_input = cuda.mem_alloc(input_size)
            else:  # and one output
                output_shape = engine.get_binding_shape(binding)
                # create page-locked memory buffers (i.e. won't be swapped to disk)
                host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
                device_output = cuda.mem_alloc(host_output.nbytes)
        stream = cuda.Stream()
        for _ in range(nwarmup):
            host_input = np.array(input_data.cpu().numpy(), dtype=np.float32, order='C')
            cuda.memcpy_htod_async(device_input, host_input, stream)

            # run inference
            context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(host_output, device_output, stream)
        stream.synchronize()

            # postprocess results
           # output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[1])
           # postprocess_output(output_data)
            
    
    print("  Start timing ...")
    timings = []
    if typee.lower()=='pytorch':
        with torch.no_grad():
            for i in range(1, nruns+1):
                start_time = time.time()
                features = model(input_data)
                torch.cuda.synchronize()
                end_time = time.time()
                timings.append(end_time - start_time)
                if i%10==0:
                    print('  Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))
    if typee.lower()=='trt':
        for i in range(1, nruns+1):
            start_time = time.time()
            host_input = np.array(input_data.cpu().numpy(), dtype=np.float32, order='C')
            cuda.memcpy_htod_async(device_input, host_input, stream)
            context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(host_output, device_output, stream)
            features = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[1])
            stream.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('  Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))



    print("  Input shape:", input_data.size())
    print("  Output features size:", features.size())
    print('  Average batch time: %.2f ms'%(np.mean(timings)*1000))
    return np.mean(timings)*1000

if __name__ == "__main__":
    searchterms = []
    parser = argparse.ArgumentParser(description='Test Speeds of PyTorch vs TensorRT Engine')
    parser.add_argument('-m', '--model_dir', default='model.pt', type=str, help='Pytorch Model Directory')
    parser.add_argument('-n', '--nruns', default=100, type=int, help='Number Of Runs')
    parser.add_argument('-e', '--engine_dir', default='model.engine', type=str, help='TensorRT Engine Directory')
    args = parser.parse_args()

    model_dir = args.model_dir
    engine_dir = args.engine_dir
    nruns = args.nruns
    print("Testing PyTorch Model Normally...")
    model = load_model(model_dir = model_dir)
    py = benchmark( model,typee='pytorch', nwarmup=50, nruns=nruns)
    print("Testing TensorRT Engine...")
    engine, context = load_engine(engine_dir = engine_dir)
    tr = benchmark( engine, context = context,typee='trt', nwarmup=50, nruns=nruns)
    print('\n\n# Pytorch took around '+"%.2f"%py+"ms whereas TensorRT took "+'%.2f'%tr+"ms for a batch")
    print('# Therefore TensorRT is about %.2f times faster.'%(py/tr))


    fig = plt.figure(figsize = (10, 10))
    y = [round(py,2),round(tr,2)]
    plt.rcParams.update({'font.size': 15})
    # creating the bar plot
    barr = plt.bar(['PyTorch','TensorRT'], y)
    barr[0].set_color('orange')
    barr[1].set_color('green')
    plt.ylabel("Time taken for inference (in ms)",fontdict=dict(fontsize=15))
    plt.title("Comparision of Inference Time (Lower is better)",fontdict=dict(fontsize=15))
    for index,data in enumerate(y):
        plt.text(x=index , y =data +0.03, s=f"{data} ms" ,ha='center', fontdict=dict(fontsize=18))
    i = 0
    while True:
        if not os.path.isfile('Result{}.png'.format(i)):
            plt.savefig('Result{}.png'.format(i))
            break
        else:
            i+=1

    print('# A graph comparing Inference Time of both have been saved as Result{}.png'.format(i))
