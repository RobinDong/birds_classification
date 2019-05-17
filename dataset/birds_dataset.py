import os
import cv2

import torch
import torch.utils.data as data

class BirdsDataset(data.Dataset):
    """ All images and classes for birds through the world """
    def __init__(self, root_path):
        self.image_list = []
        self.labelmap = {}
        for directory in os.walk(root_path):
            for dir_name in directory[1]: # All subdirectories
                type_id, type_name = dir_name.split('.')
                type_id = int(type_id)
                self.labelmap[type_id] = type_name
                for image_file in os.listdir(os.path.join(root_path, dir_name)):
                    full_path = os.path.join(root_path, dir_name, image_file)
                    self.image_list.append((full_path, type_id))

    def __getitem__(self, index):
        image_path, type_id = self.image_list[index]
        image = cv2.imread(image_path)
        return image, type_id

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    bds = BirdsDataset('/disk3/donghao/data/bird/')
    image, type_id = bds[3]
    print(image, type_id)
