import tensorflow as tf
import numpy as np
from resnet50.ResNet import ResNet
from resnet50.readtxt import read_data
import cv2
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batchsize = 128
Epoch = 2000
label_num = 45

# the path of training set
path_train = ""
# the path of testing set
path_test = ""
# the path of pretrained resnet50's weights
pretrained_weights = "D:/PycharmProjects/learning/resnet50/pretrained_weights/"
# the path of trained resnet50's weights
saved_weights = "D:/PycharmProjects/learning/resnet50/saved_weights/"
# the path of saved model
saved_model = "D:/PycharmProjects/learning/resnet50/saved_model/"


train_data, train_label = read_data(path_train)
test_data, test_label = read_data(path_test)

print("train data shape: {}, label shape: {}".format(train_data.shape, train_label.shape))
print("test data shape: {}, label shape: {}".format(test_data.shape, test_label.shape))

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, label_num])
kp = tf.placeholder(tf.float32)
resnet = ResNet(resnet_npy_path=pretrained_weights + weights_resnet.npy)
resnet.build(x, label_num=label_num, kp=kp, last_layer_type="no")
res_logits = resnet.prob
predict = tf.nn.softmax(res_logits)

with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=res_logits))
    var_net = tf.trainable_variables()
    l2loss = 0
    for var in var_net:
        l2loss += tf.nn.l2_loss(var)
    loss = cross_entropy + 2e-4 * l2loss

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 200, 0.96, staircase=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_op = tf.group([optimizer, update_ops])

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict, axis=1), tf.argmax(y, axis=1)), tf.float32))
var = tf.trainable_variables()
for item in var:
    print(item)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    resnet.load_weights(sess)
    saver.restore(sess, saved_model + resnet.ckpt)

    epoch = 0
    while epoch < Epoch:
        idx = np.arange(train_data.shape[0])
        np.random.shuffle(idx)
        current_train_data = train_data[idx]
        current_train_label = train_label[idx]

        for batch in range(current_train_data.shape[0] // batchsize):
            start = batch * batchsize
            end = min(start + batchsize, current_train_data.shape[0])
            batchdata = current_train_data[start:end]
            batchlabel = current_train_label[start:end]
            _, loss_value = sess.run([train_op, loss], feed_dict={x: batchdata, y: batchlabel, kp: 0.65})

            if batch % 5 == 0:
                train_idx = np.arange(train_data.shape[0])
                np.random.shuffle(train_idx)
                train_acc_data, train_acc_label = train_data[train_idx[:batchsize]], train_label[train_idx[:batchsize]]
                train_acc = sess.run(accuracy, feed_dict={x: train_acc_data, y: train_acc_label, kp: 1.0})

                test_idx = np.arange(test_data.shape[0])
                np.random.shuffle(test_idx)
                test_acc_data, test_acc_label = test_data[test_idx[:batchsize]], test_label[test_idx[:batchsize]]
                test_acc = sess.run(accuracy, feed_dict={x: test_acc_data, y: test_acc_label, kp: 1.0})
                epoch = epoch + 1
                print("Epoch [{}:{}], Loss: {:.6f}, Train Acc: {}, Test Acc: {}".format(epoch, Epoch, loss_value,
                                                                                        train_acc, test_acc))
                #saver.save(sess, "/home/admin1/PycharmProjects/resnet50/weights/resnet.ckpt")
                resnet.save_weights(path=saved_weights + resnet.npy)
