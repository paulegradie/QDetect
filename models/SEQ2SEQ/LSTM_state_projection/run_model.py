import os
from functools import partial

import tensorflow as tf
from backend.model_helpers import (embedding_helper, get_forget_bias,
                                   load_embeddings, predict_words)
from training.training_seq2seq import train_model
from training.utils import gather_args
from backend.utils import get_available_gpus
from tensorflow import logging as logging
from tensorflow.python.client import device_lib
from tensorflow.python.layers.core import Dense

logging.set_verbosity(tf.logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def model_fn(features, labels, mode, params):
    # particular to this project
    word2index = params['word2index']
    # index2word = params['index2word']

    GPUs = get_available_gpus()
    GPU = {
        'titan': GPUs[1],
        'sidekick': GPUs[0]}

    lookup_table, emb_vectors = load_embeddings(params['embedding_vectors'], params['vocab'])
    embedded_enc_input = tf.nn.embedding_lookup(emb_vectors, features['encoder_inputs'])
    forget_bias = get_forget_bias(params, mode)

    num_units = [2048]
    init = tf.initializers.truncated_normal(0.0, 0.01)

    with tf.device(GPU['titan']):
        encoder_cells = [tf.nn.rnn_cell.LSTMCell(num_units=num, forget_bias=forget_bias, initializer=init) for num in num_units]
        encoder_stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_cells)

        enc_outputs, enc_final_state = tf.nn.dynamic_rnn(encoder_stacked_rnn_cell,
                                                         embedded_enc_input,
                                                         sequence_length=features['encoder_input_lengths'],
                                                         dtype=tf.float32)

    # state projection
    projected_state = Dense(1024)(enc_final_state[-1])

    # Decoder model
    with tf.device(GPU['sidekick']):
        partial_embedding_helper = partial(embedding_helper, emb_vectors=emb_vectors)
        if mode == tf.estimator.ModeKeys.TRAIN:
            embed_dec_inputs = tf.nn.embedding_lookup(emb_vectors, features['decoder_inputs'])
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=embed_dec_inputs,
                sequence_length=features['decoder_input_lengths'],
            )
        else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=partial_embedding_helper,
                start_tokens=tf.tile([word2index['<GO>']],
                                     [tf.shape(features['encoder_inputs'])[0]]),
                end_token=word2index['<EOS>'])

        dec_cell = tf.nn.rnn_cell.LSTMCell(num_units=1024,  # needs to match size of last layer of encoder
                                           forget_bias=forget_bias,
                                           initializer=init)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=dec_cell,
            helper=helper,
            initial_state=projected_state,
            output_layer=Dense(params['vocab_size'], use_bias=False))
        dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=params['output_max_length'])
    logits = tf.identity(dec_outputs.rnn_output, 'logits')

    if mode == tf.estimator.ModeKeys.PREDICT:
        indices = predict_words(logits)
        predictions = {'sentence_tokens': indices}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    training_labels = labels['target_sequences']
    weights = tf.cast(tf.cast(tf.not_equal(training_labels, tf.constant(word2index['<PAD>'])), tf.bool), tf.float32)
    sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=training_labels, weights=weights)

    tf.summary.scalar('sequence_loss', sequence_loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {'loss': sequence_loss}
        return tf.estimator.EstimatorSpec(mode, loss=sequence_loss, eval_metric_ops=metrics)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(sequence_loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=sequence_loss, train_op=train_op)


if __name__ == "__main__":
    """
    If we start from scratch, we'll make a new directory
     - if we name the directory, it takes the name, else, it is called 'test_directory'
    If we change hyperparameters, change the name of the output_dir.
    """
    args = gather_args()
    access_model(args, model_fn)
