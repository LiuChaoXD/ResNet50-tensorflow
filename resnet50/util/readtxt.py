import cv2
import numpy as np

def read_data(path):
    data = []
    label = []
    with open(path, 'r') as f:
        x = f.readlines()
        for name in x:
            filename = name.strip().split()[0]
            filelabel = int(name.strip().split()[1])
            label = np.eye(label_num)[filelabel]
            temp = cv2.imread(filename)
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            temp = cv2.resize(temp, dsize=(224, 224))
            data.append(temp / 255.)
            label.append(label)
    data = np.array(data)
    label = np.array(label)
    return data, label