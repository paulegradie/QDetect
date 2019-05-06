import os

import tensorflow as tf
from backend.utils import convert_int2word
from tensorflow import logging as logging
from tensorflow.python.client import device_lib

logging.set_verbosity(tf.logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_embeddings(embedding_matrix, vocabulary, eager=False):
    with tf.variable_scope('embeddings'):
        if not eager:
            index = tf.Variable(initial_value=vocabulary, trainable=False, dtype=tf.string)
            emb_vectors = tf.Variable(initial_value=embedding_matrix, trainable=False, dtype=tf.float32)
        else:
            index = tf.contrib.eager.Variable(initial_value=vocabulary, trainable=False, dtype=tf.string)
            emb_vectors = tf.contrib.eager.Variable(initial_value=embedding_matrix, trainable=False)

    lookup_table = tf.contrib.lookup.index_table_from_tensor(index, num_oov_buckets=1000)
    return lookup_table, emb_vectors


def predict_words(logits):
    probs = tf.nn.softmax(logits, axis=2)
    indices = tf.argmax(probs, axis=2)
    return indices


def embedding_helper(index, emb_vectors):
    embedding = tf.nn.embedding_lookup(emb_vectors, index)
    return embedding


def show_preds(prediction_gen, index2word):
    for sequence in prediction_gen:
        p = convert_int2word(index2word, sequence)
        print("Prediction: \n")
        print(p)


def get_forget_bias(params, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
        forget_bias = params['forget_bias']
    else:
        forget_bias = 1.0
    return forget_bias


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']