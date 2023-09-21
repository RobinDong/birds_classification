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

import pycls.core.builders as model_builder
from pycls.core.config import cfg

MODEL_NAME="bird_img_convnextv2_nano_20230720"

def pressure_predict(net, tensor_img):
    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            result = net(tensor_img)
            result = softmax(result)
            values, indices = torch.topk(result, 10)
    t1 = time.time()
    print("time:", t1 - t0)
    print(values, indices)

if __name__ == "__main__":
    cfg.MODEL.TYPE = "regnet"
    # RegNetY-8.0GF
    cfg.REGNET.DEPTH = 17
    cfg.REGNET.SE_ON = False
    cfg.REGNET.W0 = 192
    cfg.REGNET.WA = 76.82
    cfg.REGNET.WM = 2.19
    cfg.REGNET.GROUP_W = 56
    cfg.BN.NUM_GROUPS = 4
    cfg.MODEL.NUM_CLASSES = 11120
    net = model_builder.build_model()
    net = timm.create_model("convnextv2_nano", num_classes=11120)
    state_dict = torch.load(f"{MODEL_NAME}.pth", map_location="cpu")
    del state_dict["_config"]
    net.load_state_dict(state_dict)
    net.eval()
    #net = net.float()

    torch.onnx.export(
        net,
        torch.randn(1, 3, 300, 300),
        f"{MODEL_NAME}.onnx",
        input_names = ['input'],
        output_names = ['output'])

    import onnx
    onnx_model = onnx.load(f"{MODEL_NAME}.onnx")
    onnx.checker.check_model(onnx_model)

    # read image
    img = cv2.imread("/home/sanbai/like2.jpg")
    img = cv2.resize(img, (300, 300))
    #img = np.expand_dims(img, axis=0)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    feed = [img]
    print(img.shape, img.dtype)

    import onnxruntime as ort

    ort_sess = ort.InferenceSession(f"{MODEL_NAME}.onnx")
    outputs = ort_sess.run(None, {'input': feed})
    ind = outputs[0][0].argsort()[-10:][::-1]
    print("outputs:", outputs[0][0], ind)
