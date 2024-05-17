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
import openvino as ov

#MODEL_NAME="mammals_20231119_0.9190_tiny"
MODEL_NAME="ckpt/mix_cls_1499877"

def pressure_predict(net, tensor_img):
    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
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
    net = timm.create_model("convnextv2_tiny", num_classes=29200)
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
    img = cv2.imread("/home/sanbai/dabailu.jpg")
    img = cv2.resize(img, (300, 300))
    tensor_img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float()/255.0
    print("normal:")
    pressure_predict(net, tensor_img)

    # openvino
    ov_model = ov.convert_model(net, example_input=(tensor_img,))
    core = ov.Core()
    cmodel = core.compile_model(ov_model, "CPU")
    print("openvino:")
    begin = time.time()
    for _ in range(100):
        result = cmodel({0: tensor_img.numpy()})
        #print("OVDict:", result[0], type(result[0]))
        values, indices = torch.topk(torch.tensor(result[0]), 10)
    print("time:", time.time() - begin)
    print(values, indices)

    # quantization
    model_int8 = torch.quantization.quantize_dynamic(
            net,
            {torch.nn.Linear, torch.nn.Conv2d, torch.nn.GroupNorm},
            dtype=torch.qint8)
    torch.save(model_int8, "int8.pth")
    print("dynamic quantization:")
    pressure_predict(model_int8, tensor_img)

    #print("static quantization:")
    #pressure_predict(model_static_quantized, tensor_img)


    dummy_input = torch.randn(1, 3, 300, 300)
    with torch.jit.optimized_execution(True):
        traced_script_module = torch.jit.trace(net, dummy_input)

    net = torch.jit.optimize_for_inference(traced_script_module)
    print("torch jit opt:")
    pressure_predict(net, tensor_img)

    import intel_extension_for_pytorch as ipex
    net = net.to(memory_format=torch.channels_last)
    net = ipex.optimize(net, weights_prepack=False)
    net = torch.compile(net, backend="ipex")
    tensor_img = tensor_img.to(memory_format=torch.channels_last)

    print("intel opt:")
    with torch.no_grad():
        pressure_predict(net, tensor_img)
