import os
import cv2
import csv
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from math import sqrt
from itertools import product as product
from torch.autograd import Variable, Function
from efficientnet_pytorch import EfficientNet

IMAGE_SHAPE = (300, 300)


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError("nms_threshold must be non negative.")
        self.conf_thresh = conf_thresh
        self.variance = voc["variance"]

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat(
                    (scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1
                )
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg["min_dim"]
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg["aspect_ratios"])
        self.variance = cfg["variance"] or [0.1]
        self.feature_maps = cfg["feature_maps"]
        self.min_sizes = cfg["min_sizes"]
        self.max_sizes = cfg["max_sizes"]
        self.steps = cfg["steps"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.clip = cfg["clip"]
        self.version = cfg["name"]
        for v in self.variance:
            if v <= 0:
                raise ValueError("Variances must be greater than 0")

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


voc = {
    "num_classes": 21,
    "lr_steps": (80000, 100000, 120000),
    "max_iter": 120000,
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "min_dim": 300,
    "steps": [8, 16, 32, 64, 100, 300],
    "min_sizes": [30, 60, 111, 162, 213, 264],
    "max_sizes": [60, 111, 162, 213, 264, 315],
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    "variance": [0.1, 0.2],
    "clip": True,
    "name": "VOC",
}


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = voc
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == "test":
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(
                    conf.view(conf.size(0), -1, self.num_classes)
                ),  # conf preds
                self.priors.type(type(x.data)),  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors,
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == ".pkl" or ".pth":
            print("Loading weights into state dict...")
            self.load_state_dict(
                torch.load(base_file, map_location=lambda storage, loc: storage)
            )
            print("Finished!")
        else:
            print("Sorry only .pth and .pkl files supported.")


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "C":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != "S":
            if v == "S":
                layers += [
                    nn.Conv2d(
                        in_channels,
                        cfg[k + 1],
                        kernel_size=(1, 3)[flag],
                        stride=2,
                        padding=1,
                    )
                ]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [
            nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)
        ]
        conf_layers += [
            nn.Conv2d(
                vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1
            )
        ]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [
            nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)
        ]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    "300": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "C",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ],
    "512": [],
}
extras = {
    "300": [256, "S", 512, 128, "S", 256, 128, 256, 128, 256],
    "512": [],
}
mbox = {
    "300": [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    "512": [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print(
            "ERROR: You specified size "
            + repr(size)
            + ". However, "
            + "currently only SSD300 (size=300) is supported!"
        )
        return
    base_, extras_, head_ = multibox(
        vgg(base[str(size)], 3),
        add_extras(extras[str(size)], 1024),
        mbox[str(size)],
        num_classes,
    )
    return SSD(phase, size, base_, extras_, head_, num_classes)


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels


def detect(img, net, transform):
    height, width = img.shape[:2]
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x)  # forward pass
    detections = y.data[0]
    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])
    for index, loc in enumerate(detections[3]):
        score = loc.numpy()[0]
        if score >= 0.5:
            loc = loc[1:]
            pt = loc * scale
            print(score, pt)
            img = img[int(pt[1]):int(pt[3]), int(pt[0]):int(pt[2])]
            return img


def load_label_map(filename):
    label_map = {}

    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            id = int(row[0])
            name = row[1]
            label_map[id] = name

    return label_map


def predict(args):
    detect_net = build_ssd("test", 300, 21)  # initialize SSD
    detect_net.load_state_dict(torch.load(args.detect_model, map_location="cpu"))
    transform = BaseTransform(detect_net.size, (104 / 256.0, 117 / 256.0, 123 / 256.0))

    net = EfficientNet.from_name('efficientnet-b2', override_params={'num_classes': 11000})
    net.load_state_dict(torch.load(args.classify_model, map_location='cpu'))
    net.eval()

    softmax = nn.Softmax(dim=1)

    img = cv2.imread(args.image_file)

    t0 = time.time()
    # Just find the most possible bird (if there are many)
    img = detect(img, detect_net, transform)
    img = cv2.resize(img, IMAGE_SHAPE)

    tensor_img = torch.from_numpy(img)
    result = net(tensor_img.unsqueeze(0).permute(0, 3, 1, 2).float())
    result = softmax(result)
    values, indices = torch.topk(result, 10)
    t1 = time.time()

    print(indices)
    labelmap = load_label_map("labelmap.csv")
    for id, accu in zip(indices[0].tolist(), values[0].tolist()):
        print("{:1.4f}, {}".format(accu, labelmap.get(id, "Unknown")))
    print('time:', t1 - t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', default=None, type=str, help='Image file to be predicted')
    parser.add_argument('--classify_model', default='ckpt/bird_cls_0.pth',
                        type=str, help='Trained ckpt file path to open')
    parser.add_argument('--detect_model', default='ckpt/bird_cls_0.pth',
                        type=str, help='Trained ckpt file path to open')
    args = parser.parse_args()

    predict(args)
