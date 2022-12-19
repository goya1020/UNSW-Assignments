import tensorflow as tf
import numpy as np
import glob  # this will be useful when reading reviews from file
import os
import tarfile
from collections import deque
from tensorflow.contrib import rnn
import string

batch_size = 50


def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
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

    # print("READING DATA")
    # basically they are pronouns, prepositions and articles.
    stop_words_set = {b'a', b'an', b'the', b'about', b'after', b'along', b'amid',
                      b'among', b'as', b'at', b'for', b'from', b'in', b'inside', b'into', b'of',
                      b'on', b'onto', b'to', b'towards', b'toward', b'upon', b'via', b'with', b'within',
                      b'he', b'she', b'i', b'am', b'is', b'are', b'it', b'they', b'them', b'him', b'her',
                      b'me', b'my', b'his', b'its', b'their', b'our', b'us', b'we', b'you', b'yours', b'there',
                      b'mr', b'mrs', b'ms', b'miss', b'and', b'or', b'was', b'were', b'those', b'these', b'that',
                      b'be', b'by', b'this', b'others', b'other', b'have', b'has', b'being', b'hers', b'mine',
                      b'who', b'whom', b'whose', b'which', b'one', b'two', b'three', b'myself', b'yourself',
                      b'yourselves', b'theirselves', b'itself'}

    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(dir,
                                       'data2/pos/*'))
    file_list.extend(glob.glob(os.path.join(dir,
                                            'data2/neg/*')))
    # print("Parsing %s files" % len(file_list))
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
            tokens = list(filter(lambda x: (x in glove_dict) and (x not in stop_words_set), tokens))

            # cap the list to len == 40, be aware of the situation if len < 40
            while(len(tokens) < 40):
                tokens.append(b'UNK')
            tokens = tokens[:40]

            # since we have maken sure all words in the dictionary, it is ok ot
            # do this.
            data[ind] += list(map(lambda x: glove_dict[x], tokens))
    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    # if you are running on the CSE machines, you can load the glove data from here
    # data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    data = open("glove.6B.50d.txt", 'r', encoding="utf-8")
    embeddings = deque()
    word_index_dict = dict()
    for i, line in enumerate(data):
        word_index_dict[line.split()[0].encode('utf-8')] = i + 1
        embeddings.append(np.array(line.split()[1:], dtype=np.float))

    # add 0 vector for 'UNK'
    embeddings.appendleft(np.array([0] * embeddings[0].shape[0]))
    word_index_dict[b'UNK'] = 0

    embeddings = np.array(embeddings)
    del data
    return embeddings, word_index_dict


def define_graph(glove_embeddings_arr):  # glove_embeddings_arr is embeddings
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, optimizer, accuracy and loss
    tensors"""

    # initialization
    global batch_size
    n_hidden = 6
    input_data = tf.placeholder(tf.int32, [batch_size, 40])
    train_data_batch = tf.nn.embedding_lookup(glove_embeddings_arr, input_data)
    train_data_batch = tf.cast(train_data_batch, tf.float32)
    # axis is 1 beacuse we need to take the last words' outputs of each states
    train_data_batch = tf.unstack(train_data_batch, axis=1)
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

    # rnn
    def lstm_cell():
        lstm = tf.contrib.rnn.GRUCell(n_hidden)
        lstm_with_dropout = tf.contrib.rnn.DropoutWrapper(
            lstm, output_keep_prob=dropout_keep_prob)
        return lstm_with_dropout
        # return lstm

    # cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for i in range(2)])
    cell = lstm_cell()
    outputs, _ = rnn.static_rnn(cell, train_data_batch, dtype=tf.float32)

    # evaluation
    weights = tf.Variable(tf.random_normal(stddev=1.0, shape=[n_hidden, 2]))
    biases = tf.Variable(tf.constant(value=0.1, shape=[2]))

    logits = tf.matmul(outputs[-1], weights) + biases
    predictions = tf.nn.softmax(logits)

    labels = tf.placeholder(tf.int32, [batch_size, 2])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))
    optimizer = tf.train.AdagradOptimizer(0.5).minimize(loss=loss)

    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1)), tf.float32))
    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss


# log (in developing, may reduce iterations to save time)
# 20k for training, 5k for testing, 2.5k each for pos and neg
# in terms of training: 0.8 and testing: 0.2
# iterations   cell_type    rnn_layer    n_hidden    accuracy    other
# --------------------------------------------------------
# 50k           BasicLSTM    1           30          0.7206      no dropout, SGD learning_rate = 1
# 100k          BasicLSTM    1           50          0.6912      no dropout, SGD learning_rate = 1, overfitting checked
# 100k          LSTMCell     3           30          0.718       with dropout keep_prb=0.5, SGD learning_rate = 1
# 10k           BasicLSTM    3           40          0.7456      with dropout keep_prb=0.65, SGD learning_rate = 1, modified ini weights and biases
# 10k           BasicLSTM    1           256         0.7384      with dropout keep_prb=0.65, SGD learning_rate = 1,
# 100k          BasicLSTM    1           256         0.7226      with dropout keep_prb=0.60, SGD learning_rate = 1,
# 100k          BasicLSTM    3           40          0.7076      with dropout keep_prb=0.65, SGD learning_rate = 1,
# 20k           BasicLSTM    8           10          0.7296      with dropout keep_prb=0.8, SGD learning_rate = 1,
# 20k           GRUCell      1           10          0.7372      with dropout keep_prb=0.8, SGD learning_rate = 1,  faster
# 20k           GRUCell      1           10          0.754       with dropout keep_prb=0.8, SGD learning_rate = 0.1,
# 20k           GRUCell      1           20          0.7446      with dropout keep_prb=0.8, SGD learning_rate = 0.1,
# 20k           GRUCell      1           15          0.7588      with dropout keep_prb=0.8, SGD learning_rate = 0.1,
# 20k           GRUCell      1           12          0.7534      with dropout keep_prb=0.8, SGD learning_rate = 0.1,
# 20k           GRUCell      1           17          0.7496      with dropout keep_prb=0.8, SGD learning_rate = 0.1,
# 20k           GRUCell      1           15          0.7446      with dropout keep_prb=0.8, SGD learning_rate = 0.01,
# fixed weights and biases initialization
# 20k           GRUCell      1           15          0.7592      with dropout keep_prb=0.8, SGD learning_rate = 0.05
# how about high n_hidden num and low dropout keeping rate
# 20k           GRUCell      1           60          0.7484      with dropout keep_prb=0.1, SGD learning_rate = 0.05
# 100k          GRUCell      1           15          0.7298      with dropout keep_prb=0.8, SGD learning_rate = 0.05
# 100k          GRUCell      1           15          0.7386      with dropout keep_prb=0.4, SGD learning_rate = 0.05
# 90k           GRUCell      1           15          0.7238      with
# dropout keep_prb=0.2, SGD learning_rate = 0.05

# summary:
# 1. Learning rate and state_size(n_hidden) are the most important hyperparameters.
# 2. GRUCell is generally faster than the other two.
# 3. Multi-layer network seems to be stupid to use in this assignment,
#    beacuse it may dramatically slow down the training speed with little increase on performance.
# 4. a small number for n_hidden could be fairly enough.

# updating the testing scritp, calculate the testing accuracy at each 10000 iterations (10000, 20000 ... 90000)
# and eventually a accuracy plot would come (using matplotlib.pyplot).
# fix: GRUCell, 1 layer
# optimizer                     n_hidden     accuracy              dropout keep_prb      comment
# AdagradOptimizer              30           from 0.745 to 0.69    0.6                   a perfect textbook-example-like overfitting...
# AdagradOptimizer              15           from 0.748 to 0.7     0.6                   still overfitting
# AdagradOptimizer              10           from 0.748 to 0.7     0.6                   still overfitting...
# AdagradOptimizer               8           from 0.748 to 0.7     0.6
# a little bit overfitting ...?
