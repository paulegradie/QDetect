import os

import tensorflow as tf
from tensorflow import logging as logging

logging.set_verbosity(tf.logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_input_fn_params(batch_size, repeat=-1):
    return {
        'max_num_questions': 1,
        'max_num_elements': 3,
        'randomize_num_questions': False,
        "shuffle": 2000,
        'repeat': repeat,
        'batch_size': batch_size
    }


def load_graph_params(batch_size,
                      max_elements_meta,
                      vocab_size_meta,
                      embedding_dims_meta,
                      max_seq_length_meta,
                      vectors,
                      word2index,
                      index2word,
                      vocab,
                      **kwargs):
    return {
        'num_classes': max_elements_meta,
        'vocab_size': vocab_size_meta,
        'embed_dim': embedding_dims_meta,
        'input_max_length': max_seq_length_meta,
        'output_max_length': max_seq_length_meta,
        'batch_size': batch_size,
        'embedding_vectors': vectors,
        'word2index': word2index,
        'index2word': index2word,
        'vocab': vocab,

        # hyperparams
        "forget_bias": 0.5,
        "learning_rate": 0.001
    }


def load_estimator_config(save_every=1000, log_every=1000, keep_max=5):
    run_config = {
        'tf_random_seed': 42,
        'save_summary_steps': save_every,
        'save_checkpoints_steps': 500,
        'keep_checkpoint_max': keep_max,
        'keep_checkpoint_every_n_hours': 10000,  # default
        'log_step_count_steps': log_every,
        'train_distribute': None,
        'session_config': tf.ConfigProto(allow_soft_placement=True,
                                         log_device_placement=False,
                                         gpu_options=tf.GPUOptions(allow_growth=True))
    }

    estimator_config = tf.estimator.RunConfig(**run_config)
    return estimator_config
