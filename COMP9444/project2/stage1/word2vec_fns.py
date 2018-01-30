import tensorflow as tf
import numpy as np
import collections
from six.moves import xrange
import time

data_index = 0


def generate_batch(data, batch_size, skip_window):
    """
    Generates a mini-batch of training data for the training CBOW
    embedding model.
    :param data (numpy.ndarray(dtype=int, shape=(corpus_size,)): holds the
        training corpus, with words encoded as an integer
    :param batch_size (int): size of the batch to generate
    :param skip_window (int): number of words to both left and right that form
        the context window for the target word.
    Batch is a vector of shape (batch_size, 2*skip_window), with each entry for the batch containing all the context words, with the corresponding label being the word in the middle of the context
    """
    global data_index
    assert batch_size % (skip_window*2) == 0
    batch = np.ndarray(shape=(batch_size, skip_window*2), dtype=np.int32) 
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in xrange(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in xrange(batch_size):
        mask = [1] * span
        mask[skip_window] = 0 
        batch[i, :] = [i for i, j in zip(buffer, mask) if j]
        labels[i, 0] = buffer[skip_window] 
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    del buffer
    return batch, labels



def get_mean_context_embeds(embeddings, train_inputs):
    """
    :param embeddings (tf.Variable(shape=(vocabulary_size, embedding_size))
    :param train_inputs (tf.placeholder(shape=(batch_size, 2*skip_window))
    returns:
        `mean_context_embeds`: the mean of the embeddings for all context words
        for each entry in the batch, should have shape (batch_size,
        embedding_size)
    """
    # cpu is recommended to avoid out of memory errors, if you don't
    # have a high capacity GPU
    batch_size = int(train_inputs.shape[0])
    embedding_size = int(embeddings.shape[1])
    double_skip_window = int(train_inputs.shape[1])

    mean_context_embeds = tf.zeros([batch_size, embedding_size])
    for i in xrange(double_skip_window):
        mean_context_embeds += tf.nn.embedding_lookup(embeddings, train_inputs[:, i]) # sum up
        
    mean_context_embeds /= double_skip_window # mean(average)


    return mean_context_embeds


