# ResNet50-tensorflow
This model build with python 3.6, and tensorflow 1.13
Build ResNet50 with tensorflow, load the pretrained parameters' weights 

# Some path settings in train.py
path_train = "path/to/training set/" 
path_test = "path/to/testing set/"
pretrained_weights = "path/to/pretrained-resent50's weights/"
saved_weights = "path/to/trained-resent50's weights"
saved_model = "path/to/saved_model/"

# How to run
The run file is train.py, so if you want to train the pretrained-resnet50 net for your classification task, you can run train.py. Notice that:
label_num = 45 means the classification task supports for 45 classes. Maybe you should turn this number for your task.

# conclusion
If you have some questions, ask me.
