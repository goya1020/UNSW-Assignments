"""
You are encouraged to edit this file during development, however your final
model must be trained using the original version of this file. This file
trains the model defined in implementation.py, performs tensorboard logging,
and saves the model to disk every 10000 iterations. It also prints loss
values to stdout every 50 iterations.
"""


import numpy as np
import tensorflow as tf
from random import randint
import datetime
import os
import matplotlib.pyplot as plt

import implementation as imp

batch_size = imp.batch_size
iterations = 10000
# iterations = 20001
seq_length = 40  # Maximum length of sentence

checkpoints_dir = "./checkpoints"

def getTrainBatch():
    labels = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        if (i % 2 == 0):
            # num = randint(0, 12499)
            num = randint(0, 9999)
            labels.append([1, 0])
        else:
            # num = randint(12500, 24999)
            num = randint(12500, 22499)
            labels.append([0, 1])
        arr[i] = training_data[num]
    return arr, labels


# Call implementation
glove_array, glove_dict = imp.load_glove_embeddings()
training_data = imp.load_data(glove_dict)
input_data, labels, dropout_keep_prob, optimizer, accuracy, loss = imp.define_graph(glove_array)

#test
test_data = np.zeros([5000, 40])

for i in range(10000, 12500):
    test_data[i-10000] = training_data[i] 
for i in range(22500, 25000):
    test_data[i-20000] = training_data[i]

test_labels = [[1, 0]] * 2500 + [[0, 1]] * 2500


test_data = np.reshape(test_data, (100, 50, 40))
test_labels = np.reshape(test_labels, (100, 50, 2))

# tensorboard
train_accuracy_op = tf.summary.scalar("training_accuracy", accuracy)
tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge_all()

# saver
all_saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

logdir = "tensorboard/" + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)


# training
test_acc_list = []
for i in range(iterations):
    batch_data, batch_labels = getTrainBatch()
    sess.run(optimizer, {input_data: batch_data, labels: batch_labels, dropout_keep_prob:0.38})
    if (i % 50 == 0):
        loss_value, accuracy_value, summary = sess.run(
            [loss, accuracy, summary_op],
            {input_data: batch_data,
             labels: batch_labels, dropout_keep_prob:1})
        writer.add_summary(summary, i)
        print("Iteration: ", i)
        print("loss", loss_value)
        print("acc", accuracy_value)
    if (i % 10000 == 0 and i != 0):
    # if (i % 999 == 0 and i != 0):
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        save_path = all_saver.save(sess, checkpoints_dir +
                                   "/trained_model.ckpt",
                                   global_step=i)
        print("Saved model to %s" % save_path)


        all_saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
        accumulate_acc = []
        for n in range(100):
            accuracy_value, summary = sess.run(
                [accuracy, summary_op],
                {input_data: test_data[n],
                labels: test_labels[n], dropout_keep_prob:1})
            # print("--------- testing loss", loss_value)
            print("----------testing acc", accuracy_value)
            accumulate_acc.append(accuracy_value)

        acc_mean = np.mean(accumulate_acc)
        print(acc_mean)
        test_acc_list.append((i, acc_mean))


test_acc_list = np.array(test_acc_list)
plt.plot(test_acc_list[:, 0], test_acc_list[:, 1], 'bo')
plt.title('Testing accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')

plt.show()


# # testing
# test_data = np.zeros([5000, 40])

# for i in range(2500):
#     test_data[i] = training_data[i] 
# for i in range(12500, 15000):
#     test_data[i-10000] = training_data[i]

# test_labels = [[1, 0]] * 2500 + [[0, 1]] * 2500


# test_data = np.reshape(test_data, (100, 50, 40))
# test_labels = np.reshape(test_labels, (100, 50, 2))

# all_saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))

# accumulate_acc = []
# for i in range(100):
#     accuracy_value, summary = sess.run(
#         [accuracy, summary_op],
#         {input_data: test_data[i],
#         labels: test_labels[i]})
#     # print("--------- testing loss", loss_value)
#     print("----------testing acc", accuracy_value)
#     accumulate_acc.append(accuracy_value)

# print(np.mean(accumulate_acc))



sess.close()
