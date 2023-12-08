import os
import sys
import cv2
import time
import json
import timm
import queue
import torch
import argparse
import threading

import torch
import numpy as np
import torch.nn as nn

MODEL_NAME="mammals_20231119_0.9190_tiny"

def pressure_predict(net, tensor_img):
    t0 = time.time()
    with torch.no_grad():
        for _ in range(20):
            result = net(tensor_img)
            #result = softmax(result)
            values, indices = torch.topk(result, 10)
    t1 = time.time()
    print("time:", t1 - t0)
    print(values, indices)

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

if __name__ == "__main__":
    net = timm.create_model("convnextv2_tiny", num_classes=11120)
    state_dict = torch.load(f"{MODEL_NAME}.pth", map_location="cpu")
    del state_dict["_config"]
    net.load_state_dict(state_dict)
    net.eval()
    net = net.float()
    # quant
    backend = "x86"
    net.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_static_quantized = torch.quantization.prepare(net, inplace=False)
    model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
    print_model_size(model_static_quantized)
    model_dynamic_quantized = torch.quantization.quantize_dynamic(net, qconfig_spec={torch.nn.Linear, torch.nn.Conv2d, torch.nn.GroupNorm}, dtype=torch.qint8)
    print_model_size(model_dynamic_quantized)

    softmax = nn.Softmax(dim=1).eval()

    # read image
    img = cv2.imread("pigs.jpg")
    img = cv2.resize(img, (300, 300))
    tensor_img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float()/255.0
    print("normal:")
    pressure_predict(net, tensor_img)

    # quantization
    model_int8 = torch.quantization.quantize_dynamic(
            net,
            {torch.nn.Linear, torch.nn.Conv2d, torch.nn.GroupNorm},
            dtype=torch.qint8)
    torch.save(model_int8, "int8.pth")
    print("dynamic quantization:")
    pressure_predict(model_int8, tensor_img)

    print("static quantization:")
    pressure_predict(model_static_quantized, tensor_img)


    dummy_input = torch.randn(1, 3, 300, 300)
    with torch.jit.optimized_execution(True):
        traced_script_module = torch.jit.trace(net, dummy_input)

    net = torch.jit.optimize_for_inference(traced_script_module)
    print("torch jit opt:")
    pressure_predict(net, tensor_img)

    import intel_extension_for_pytorch as ipex
    net = net.to(memory_format=torch.channels_last)
    net = ipex.optimize(net)
    tensor_img = tensor_img.to(memory_format=torch.channels_last)

    print("intel opt:")
    with torch.no_grad():
        pressure_predict(net, tensor_img)
