# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import glob  # this will be useful when reading reviews from file
import os
import tarfile
from collections import deque
import string
from tensorflow.contrib import rnn


def load_glove_embeddings():

    data = open("glove.6B.50d.txt", 'r', encoding="utf-8")
    embeddings = deque()
    word_index_dict = dict()
    for i, line in enumerate(data):
        word_index_dict[line.split()[0].encode('utf-8')] = i+1
        embeddings.append(np.array(line.split()[1:], dtype=np.float))

    # add 0 vector for 'UNK'
    embeddings.appendleft(np.array([0] * embeddings[0].shape[0]))
    word_index_dict[b'UNK'] = 0

    embeddings = np.array(embeddings)
    del data
    return embeddings, word_index_dict


embeddings, glove_dict = load_glove_embeddings()
# print(glove_dict[b'UNK']) # first elem
# print(glove_dict[b'the']) # last elem
# print(embeddings[42])
# print(embeddings.shape)


filename = 'reviews.tar.gz'
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data2/')):
    with tarfile.open(filename, "r", encoding="utf8") as tarball:
        dir = os.path.dirname(__file__)
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tarball, os.path.join(dir,"data2/"))

print("READING DATA")
# basically they are pronouns, prepositions and articles.
stop_words_set = {b'a', b'an', b'the', b'about', b'after', b'along', b'amid',
                  b'among', b'as', b'at', b'for', b'from', b'in', b'inside', b'into', b'of',
                  b'on', b'onto', b'to', b'towards', b'toward', b'upon', b'via', b'with', b'within',
                  b'he', b'she', b'i', b'am', b'is', b'are', b'it', b'they', b'them', b'him', b'her',
                  b'me', b'my', b'his', b'its', b'their', b'our', b'us', b'we', b'you', b'yours', b'there',
                  b'mr', b'mrs', b'ms', b'miss', b'and', b'or', b'was', b'were', b'those', b'these', b'that',
                  b'be', b'by', b'this', b'others', b'other', b'have', b'has', b'being', b'hers', b'mine',
                  b'who', b'whom', b'whose', b'which', b'one', b'two', b'three'}

dir = os.path.dirname(__file__)
file_list = glob.glob(os.path.join(dir,
                                   'data2/pos/*'))
file_list.extend(glob.glob(os.path.join(dir,
                                        'data2/neg/*')))
print("Parsing %s files" % len(file_list))
# shape = [25000, 40], elements are indice.
data = np.zeros(shape=[len(file_list), 40], dtype=np.int32)

for ind, f in enumerate(file_list):

    with open(f, "r", encoding='utf8') as openf:
        s = openf.read()
        no_punct_lower = ''.join(c for c in s if c not in string.punctuation).lower(
        ).encode('utf-8')  # strip punctuation and lowercase
        tokens = no_punct_lower.split()  # a list of words of a review

        # capture all words which are in the glove dict (under the asumption that only these words are useful)
        # as well as get rid of stop words
        tokens = list(filter(lambda x: (x in glove_dict)
                             and (x not in stop_words_set), tokens))

        # cap the list to len == 40, be aware of the situation if len < 40
        while(len(tokens) < 40):
            tokens.append(b'UNK')
        tokens = tokens[:40]

        # since we have maken sure all words in the dictionary, it is ok ot do
        # this.
        data[ind] += list(map(lambda x: glove_dict[x], tokens))

# print(data1[12503])
# print(glove_dict[b'77'])

# print(data.shape) # (25000, 40)
# print(data[1])

batch_size = 50
n_hidden = 20
# input_data = data[:50, :]
input_data = tf.placeholder(tf.int32, [batch_size, 40])
train_data_batch = tf.nn.embedding_lookup(embeddings, input_data)
train_data_batch = tf.cast(train_data_batch, tf.float32)
train_data_batch = tf.unstack(train_data_batch, axis=1) # axis is 1 beacuse we need to take the last words' outputs of each states

# print(train_data_batch.shape)
# print(type(train_data_batch))
rnn_cell = rnn.BasicLSTMCell(n_hidden)
outputs, states = rnn.static_rnn(rnn_cell, train_data_batch, dtype=tf.float32)
print(outputs[-1])
print(len(outputs))
# print(len(outputs[-1]))

weights = tf.Variable(tf.truncated_normal([n_hidden, 2]))
biases = tf.Variable(tf.constant(0.1, shape=[2]))

logits = tf.matmul(outputs[-1], weights) + biases
predictions = tf.nn.softmax(logits)

labels = tf.placeholder(tf.int32, [batch_size, 2])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(1).minimize(loss)

accuracy, count = tf.metrics.accuracy(labels=tf.argmax(labels, 0), predictions=tf.argmax(predictions, 0))

init = tf.global_variables_initializer()
init2 = tf.initialize_local_variables()

with tf.Session() as sess:
    sess.run(init)
    sess.run(init2)
    test_batch = data[:50, :]
    test_labels = np.reshape(np.array([[1, 0] * batch_size]), (50, 2))
    _, loss, accuracy, predictions = sess.run([optimizer, loss, accuracy, predictions], feed_dict={input_data: test_batch, labels:test_labels})

    print('losssssssssss', loss)
# print(outputs, states)
# # print(train_data.shape)
    # print(len(accuracy))
    print('accuracy is', accuracy)
    # print('predictions shape', predictions.shape)
    print(predictions)