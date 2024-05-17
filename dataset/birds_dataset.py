import os
import cv2

import numpy as np
import torch.utils.data as data

from collections import Counter

IMAGE_SHAPE = (300, 300)
# SEED = 20190519
SEED = 20240119
EVAL_RATIO = 0.05
INCORRECT_DATA_FILE = "incorrect.txt"


class ListLoader(object):
    def __init__(self, root_path, num_classes, finetune, category_prefix):
        np.random.seed(SEED)

        self.category_count = Counter()  # number of images for each category
        self.image_list = []
        self.labelmap = {}
        if category_prefix:
            self.load_labelmap()
        type_id = -1
        for directory in os.walk(root_path):
            for dir_name in directory[1]:  # All subdirectories
                # For tetrapod, we need to map name to id
                if category_prefix:
                    if dir_name[0] != category_prefix:
                        continue
                    type_id = self.labelmap[dir_name]
                else:
                    type_id += 1
                    self.labelmap[dir_name] = type_id
                if type_id < 0 or type_id > num_classes:
                    print("Wrong directory: {}!".format(dir_name))
                    continue
                for image_file in os.listdir(os.path.join(root_path, dir_name)):
                    self.category_count[type_id] += 1

                if not finetune and self.category_count[type_id] < 200:
                    continue

                for image_file in os.listdir(os.path.join(root_path, dir_name)):
                    full_path = os.path.join(root_path, dir_name, image_file)
                    self.image_list.append((full_path, type_id))

        avg_count = sum(self.category_count.values()) / len(self.category_count)
        print("Avg count per category:", avg_count)
        minimum = min(self.category_count, key=self.category_count.get)
        print("Min count category:", self.category_count[minimum])
        maximum = max(self.category_count, key=self.category_count.get)
        print("Max count category:", self.category_count[maximum])

        self.category_multiple = {}
        small_cat = 0
        for type_id in self.category_count:
            multiple = int(3 * avg_count / self.category_count[type_id])
            if multiple > 1:
                small_cat += 1
            self.category_multiple[type_id] = multiple
        print("Small categories:", small_cat)

    def image_indices(self):
        """Return train/eval image files' list"""
        length = len(self.image_list)
        indices = np.random.permutation(length)
        point = int(length * EVAL_RATIO)
        eval_indices = indices[0:point]
        train_indices = indices[point:]

        # For categories which have small number of images, oversample it
        extra_train_indices = []
        for index in train_indices:
            _, type_id = self.image_list[index]
            multiple = self.category_multiple[type_id]
            if multiple > 1:
                for i in range(multiple):
                    extra_train_indices.append(index)

        extra_train_indices = np.asarray(extra_train_indices, dtype=train_indices.dtype)
        train_indices = np.concatenate((train_indices, extra_train_indices))
        return self.image_list, train_indices, eval_indices

    def multiples(self):
        return self.category_multiple

    def export_labelmap(self, path="labelmap.csv"):
        with open(path, "w") as fp:
            for type_name, type_id in self.labelmap.items():
                count = self.category_count[type_id]
                fp.write(str(type_id) + "," + type_name + "," + str(count) + "\n")

    def load_labelmap(self, path="labelmap.csv"):
        with open(path, "r") as fp:
            for line in fp:
                arr = line.split(",")
                type_id, type_name = int(arr[0]), arr[1]
                self.labelmap[type_name] = type_id


class BirdsDataset(data.Dataset):
    """All images and classes for birds through the world"""

    def __init__(
        self,
        image_list,
        image_indices,
        category_multiple,
        is_training,
        load_incorrect=False,
    ):
        self.image_list = image_list
        self.image_indices = image_indices
        self.category_multiple = category_multiple
        self.is_training = is_training
        self._base_len = len(self.image_indices)
        if load_incorrect:
            # Load incorrect samples as additional list (weight + 2)
            addition_list = []
            with open(INCORRECT_DATA_FILE, "r") as fp:
                for line in fp:
                    pos = line.index(",", line.index(",") + 1)
                    line = line[:pos] + ")"
                    path, type_id = eval(line)
                    for _ in range(2):
                        addition_list.append((path, type_id))
            np.random.shuffle(addition_list)
            self._addition_list = addition_list

        if load_incorrect:
            self._addition_len = len(addition_list)
        else:
            self._addition_len = 0

    def __getitem__(self, index):
        if index >= self._base_len:
            pair = self._addition_list[index - self._base_len]
            image_path, type_id = pair
        else:
            image_path, type_id = self.image_list[self.image_indices[index]]

        image = cv2.imread(image_path)
        if image is None:
            print("[Error] {} can't be read".format(image_path))
            return None
        if image.shape != (IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3):
            image = cv2.resize(image, IMAGE_SHAPE)
            print("[Warn] {} has shape {}".format(image_path, image.shape))
            return None
        if isinstance(type_id, str):
            print("What:", type_id)
        return image, int(type_id)

    def __len__(self):
        return self._base_len + self._addition_len

    @staticmethod
    def my_collate(batch):
        batch = filter(lambda img: img is not None, batch)
        return data.dataloader.default_collate(list(batch))


if __name__ == "__main__":
    list_loader = ListLoader("/media/data2/i18n/V5", 11120, True)
    img_list, train_lst, eval_lst = list_loader.image_indices()
    with open("train.txt", "w") as fp:
        for ind in train_lst:
            fp.write(str(img_list[ind]) + "\n")
    with open("eval.txt", "w") as fp:
        for ind in eval_lst:
            fp.write(str(img_list[ind]) + "\n")

    bd = BirdsDataset(img_list, eval_lst, list_loader.multiples(), False)
    image, type_id = bd[1023]
    print("image", image)
    print("type_id", type_id, type(type_id))
