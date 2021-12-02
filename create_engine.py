import torch
import random
import tensorrt as trt
from torchvision import datasets, transforms, models
from torch import nn, optim
import onnx
import torch.nn.functional as F
import pycuda.driver as cuda
import pycuda.autoinit
import argparse



def load_model(model_dir = "."):
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
                            nn.Linear(128,4))

    # Making the Classifier layer Trainable                           
    for param in model.fc.parameters():
        param.requires_grad = True

    if model_dir is not None:
        print("# Loading model weights")
        model.load_state_dict(torch.load(model_dir))
    print("# Model Loaded")
    # Moving the model to device
    return model

def generate_onnx(model_dir='model.pt'):
    model = load_model(model_dir)
    input = torch.randn([1,3,224,224])
    ONNX_FILE_PATH = "model.onnx"
    torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True)
    onnx_model = onnx.load(ONNX_FILE_PATH)
    # check that the model converted fine
    onnx.checker.check_model(onnx_model)

    print("# Model was successfully converted to ONNX format.")
    print("# It was saved to", ONNX_FILE_PATH)

def create_engine(onnx_dir='model.onnx'):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(onnx_dir)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        print("# Not Success")

    config = builder.create_builder_config()
    #config.max_workspace_size = 1<<20
    #builder.max_batch_size = 1

    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
        print("# Building in FP16 Mode")
        
    print('# Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("# Saving Engine")

    with open("model.engine", "wb") as f:
        f.write(engine.serialize())
    print('# Engine created and saved!')



if __name__ == "__main__":
    searchterms = []
    parser = argparse.ArgumentParser(description='Create TensorRT Engine of ResNet50')
    parser.add_argument('-md', '--model_dir', default='model.pt', type=str, help='pt model directory')
    parser.add_argument('-od', '--onnx_dir', default=None, type=str, help='Onnx Model Directory')
    args = parser.parse_args()

    model_dir = args.model_dir
    onnx_dir = args.onnx_dir
    if onnx_dir==None:
        generate_onnx(model_dir)
        onnx_dir = 'model.onnx'
    create_engine(onnx_dir)