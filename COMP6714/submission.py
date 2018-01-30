# Submission.py for COMP6714-Project2
##########################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import zipfile
import numpy as np
import tensorflow as tf
import collections
import spacy
import pickle
import gensim


# By Boshen Hu for Programming Project 2, Information Retrieval
# and Web Search (COMP6714)

# A simple illustration of the code.
# The general order to process a training operations is
# Process data: {Raw Data --> Refinded Data (preprocessing) --> Build data set}
# --> Traning --> evaluation.

#
# Specifications
#
filename = './BBC_Data.zip'
data_index = 0
# just a default setting, modified by function later, if force_v_size is
# set to 1.
vocabulary_size = 13686
force_v_size = 0  # 0 to maximize the vocabulary size, 1 to set a number manually
data_processed_file = "./processed_file.pickle"

# Specification of Training data:
# MAKE SURE batch_size % num_samples == 0, skip_window % 2 == 0
batch_size = 32      # Size of mini-batch for skip-gram model.
embedding_size = 200  # Dimension of the embedding vector.
# How many words to consider left and right of the target word.
num_samples = 8       # How many times to reuse an input to generate a label.
skip_window = num_samples // 2
num_sampled = 10      # Sample size for negative examples.
learning_rate = 0.004
logs_path = './log/'
output_filename = 'adjective_embeddings.txt'

# # Specification of test Sample:
Sample_adj = ['new', 'more', 'last', 'best', 'next', 'many', 'good']


# Function to read the data into a list of words.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    data = str()
    with zipfile.ZipFile(filename) as f:
        for i in range(len(f.namelist())):
            data += tf.compat.as_str(f.read(f.namelist()[i]))
    return data


def make_refined_doc(data):
    global vocabulary_size

    nlp = spacy.load('en')
    doc = nlp(data)
    refined_doc = []
    adj_set = set()
    adj_indice = list()

    for ind, word in enumerate(doc):
        if word.pos_ not in ['ADJ', 'VERB', 'ADV', 'NOUN']:
            continue
        if word.pos_ != 'ADJ':  # not adjective but in verb adv noun
            # get rid of useless words, further
            if len(word) == 1 or (word.tag_ in ['WP', 'WRB', 'PRP', 'BES', 'HVS']):
                continue
            refined_doc.append(word.lemma_)
        else:
            # if word.tag_ in ['WP$', 'WDT', 'PRP$', 'AFX']:
            #     continue
            refined_doc.append(word.lower_)
            adj_indice.append(len(refined_doc) - 1)
            adj_set.add(word.lower_)
    if not force_v_size:
        vocabulary_size = len(set(refined_doc))
    print('How many unique words in the doc: ', vocabulary_size)
    print('How many unique adjective in the doc: ', len(adj_set))
    return refined_doc, adj_set, adj_indice


def build_dataset(words, n_words):
    """Process raw inputs into a dataset.
       words: a list of words, i.e., the input data
       n_words: Vocab_size to limit the size of the vocabulary. Other words will be mapped to 'UNK'
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)  # what a terrible smart line it is
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # i.e., one of the 'UNK' words
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def generate_batch(batch_size, num_samples, skip_window, data, reverse_dictionary, adj_set, adj_indice):
    global data_index

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    refill_count = 0

    if data_index >= len(data):
        data_index = 0
    # chosen_adjs = random.choices(adj_indice, k=batch_size // num_samples) #
    # thanks to the docker file, this function (random.choices) cannot run in 3.5.2, it's 3.6 only.
    chosen_adjs = []
    for x in range(batch_size // num_samples):
        chosen_adjs.append(random.choice(adj_indice))
    for adj in chosen_adjs:
        for i in range(-skip_window, skip_window + 1):
            if i != 0:
                if (adj + i - 1 + 5 < 0) or (adj + i > len(data) - 5):  # make sure not out of index
                    context_word = 0
                else:
                    context_word = data[adj + i]
                if data[adj] >= vocabulary_size:
                    batch[refill_count] = 0
                else:
                    batch[refill_count] = data[adj]
                labels[refill_count, 0] = context_word
                refill_count += 1
    return batch, labels


def construct_graph(sample_examples):
    # Constructing the graph...
    graph = tf.Graph()

    with graph.as_default():
        with tf.device('/cpu:0'):
            # Placeholders to read input data.
            with tf.name_scope('Inputs'):
                train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

            # Look up embeddings for inputs.
            with tf.name_scope('Embeddings'):
                sample_dataset = tf.constant(sample_examples, dtype=tf.int32)
                embeddings = tf.Variable(tf.random_uniform(
                    [vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                              stddev=1.0 / math.sqrt(embedding_size)))
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            with tf.name_scope('Loss'):
                loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_biases,
                                                     labels=train_labels, inputs=embed,
                                                     num_sampled=num_sampled, num_classes=vocabulary_size))

            # Construct the Gradient Descent optimizer using a learning rate of
            # 0.01.
            with tf.name_scope('Gradient_Descent'):
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate).minimize(loss)

            # Normalize the embeddings to avoid overfitting.
            with tf.name_scope('Normalization'):
                norm = tf.sqrt(tf.reduce_sum(
                    tf.square(embeddings), 1, keep_dims=True))
                normalized_embeddings = embeddings / norm

            sample_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, sample_dataset)
            similarity = tf.matmul(
                sample_embeddings, normalized_embeddings, transpose_b=True)

            # Add variable initializer.
            init = tf.global_variables_initializer()

            # Create a summary to monitor cost tensor
            tf.summary.scalar("cost", loss)
            # Merge all summary variables.
            merged_summary_op = tf.summary.merge_all()
    return (graph, init, train_inputs, train_labels, optimizer, loss, merged_summary_op, similarity, normalized_embeddings)


def save_model(output_filename, embeddings, word_list, vocabulary_size, embedding_size):
    title = str(vocabulary_size) + ' ' + str(embedding_size) + '\n'
    with open(output_filename, "w", encoding='utf-8') as text_file:
        text_file.write(title)
        for i in range(embeddings.shape[0]):
            text_file.write(str(word_list[i]) + ' ')
            text_file.write(" ".join(str(np.round(x, 6))
                                     for x in embeddings[i, :]) + "\n")


def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):

    with open(data_file, "rb") as f:
        data, count, dictionary, reverse_dictionary, refined_doc, vocabulary_size, adj_set, adj_indice = pickle.load(
            f)
    sample_examples = list(map(lambda x: dictionary[x], Sample_adj))
    graph, init, train_inputs, train_labels, optimizer, loss, merged_summary_op, similarity, normalized_embeddings = construct_graph(
        sample_examples)

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        session.run(init)
        summary_writer = tf.summary.FileWriter(
            logs_path, graph=tf.get_default_graph())

        print('Initializing the model')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(
                batch_size, num_samples, skip_window, data, reverse_dictionary, adj_set, adj_indice)
            feed_dict = {train_inputs: batch_inputs,
                         train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op using
            # session.run()
            _, loss_val, summary = session.run(
                [optimizer, loss, merged_summary_op], feed_dict=feed_dict)

            summary_writer.add_summary(summary, step)
            average_loss += loss_val

            if step % 1000 == 0:
                print('Through step:', step)

            if step % 5000 == 0:
                if step > 0:
                    average_loss /= 5000

                    # The average loss is an estimate of the loss over the last
                    # 5000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

        final_embeddings = normalized_embeddings.eval().astype(np.float64)
        word_list = [reverse_dictionary[i] for i in range(vocabulary_size)]
        save_model(output_filename, final_embeddings,
                   word_list, vocabulary_size, embedding_size)


def process_data(input_data):
    print('Start to process data...')
    data = read_data(input_data)
    print('hold on...')
    refined_doc, adj_set, adj_indice = make_refined_doc(data)
    print('...')

    data, count, dictionary, reverse_dictionary = build_dataset(
        refined_doc, vocabulary_size)

    with open(data_processed_file, "wb") as f:
        pickle.dump((data, count, dictionary, reverse_dictionary,
                     refined_doc, vocabulary_size, adj_set, adj_indice), f)
    print("Processed data have dumpped in file.")
    return data_processed_file


def Compute_topk(model_file, input_adjective, top_k):
    model = gensim.models.KeyedVectors.load_word2vec_format(
        model_file, binary=False)
    similarity_list = model.most_similar([input_adjective], [], top_k + 1)[1:]
    result = [i[0] for i in similarity_list]
    return result
