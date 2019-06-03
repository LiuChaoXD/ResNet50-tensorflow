import numpy as np
import cv2
import os


def get_dict(path):
    class_name = set()
    filepath = os.listdir(path)
    filepath.sort()
    for item in filepath:
        name = item[:-6]
        class_name.add(name)
    class_list = list(class_name)
    class_list.sort()
    class_number = [i for i in range(21)]
    key = dict(zip(class_list, class_number))
    return key


def load_data(path, size=224):
    dictor = get_dict(path)
    filename = os.listdir(path)
    filename.sort()
    data = []
    label = []
    for item in filename:
        filepath = os.path.join(path, item)
        img = cv2.imread(filepath)
        img = cv2.resize(img, dsize=(size, size))
        class_num = dictor[item[:-6]]
        data.append(img)
        label.append(class_num)
    return np.array(data), np.array(label)


def convert_to_hot(label):
    class_num = np.max(label + 1)
    return np.eye(class_num)[label]
