import sys
import cv2
import time
import json
import queue
import torch
import argparse
import threading

import torch
import numpy as np
import torch.nn as nn

import pycls.core.builders as model_builder
from pycls.core.config import cfg

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
    net.load_state_dict(torch.load("bird_cls_ca2_20220711_8778_2793532.pth", map_location="cpu"))
    #net.eval()
    #net = net.float()

    torch.onnx.export(
        net,
        torch.randn(1, 3, 300, 300),
        "bird_cls_ca2_20220711_8778_2793532.onnx",
        input_names = ['input'],
        output_names = ['output'])

    import onnx
    onnx_model = onnx.load("bird_cls_ca2_20220711_8778_2793532.onnx")
    onnx.checker.check_model(onnx_model)

    # read image
    img = cv2.imread("blujay.jpg")
    img = cv2.resize(img, (300, 300))
    img = np.expand_dims(img, axis=0)
    img = img.transpose(0, 3, 1, 2).astype(np.float32)
    print(img.shape, img.dtype)

    import onnxruntime as ort

    ort_sess = ort.InferenceSession("bird_cls_ca2_20220711_8778_2793532.onnx")
    outputs = ort_sess.run(None, {'input': img})
    ind = outputs[0][0].argsort()[-10:][::-1]
    print("outputs:", outputs, ind)
