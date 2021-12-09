import torch
import random
import numpy as np
import tensorrt as trt
from torchvision import datasets, transforms, models
from torch import nn, optim
import onnx
import torch.nn.functional as F
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import argparse
from utils import preprocess_image, postprocess_output
import torch.nn.functional as F


def load_engine(engine_dir='model.engine'):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_dir, "rb") as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()
    return engine, context

def inference(image,engine_dir='model.engine'):
    engine, context = load_engine(engine_dir=engine_dir)
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
    host_input = np.array(image.cpu().numpy(), dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)

    # run inference
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    # postprocess results
    output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[1])

    return postprocess_output(output_data)


if __name__ == "__main__":
    searchterms = []
    parser = argparse.ArgumentParser(description='Create Inference from TensorRT Engine')
    parser.add_argument('-i', '--image_dir', default='test_image.jpg', type=str, help='Image Directory')
    parser.add_argument('-e', '--engine_dir', default='model.engine', type=str, help='TensorRT Engine Directory')
    args = parser.parse_args()

    image_dir = args.image_dir
    engine_dir = args.engine_dir
    image = preprocess_image(image_dir)
    output = inference(image,engine_dir=engine_dir)
    print("# The Image Contains:",output)
