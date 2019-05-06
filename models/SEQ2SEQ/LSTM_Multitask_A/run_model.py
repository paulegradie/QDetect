import os
from functools import partial

import tensorflow as tf
from backend.model_helpers import (embedding_helper, load_embeddings,
                                   predict_words)
from training.train_seq2seq import train_model
from training.utils import gather_args
from tensorflow import logging as logging
from tensorflow.python.layers.core import Dense

logging.set_verbosity(tf.logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_forget_bias(params, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
        forget_bias = params['forget_bias']
    else:
        forget_bias = 1.0
    return forget_bias


def model_fn(features, labels, mode, params):
    word2index = params['word2index']
    # index2word = params['index2word']
    lookup_table, emb_vectors = load_embeddings(params['embedding_vectors'], params['vocab'])
    embedded_enc_input = tf.nn.embedding_lookup(emb_vectors, features['encoder_inputs'])
    forget_bias = get_forget_bias(params, mode)

    init = tf.initializers.truncated_normal(0.0, 0.01)
    num_units = [params['num_rnn_units_1'], params['num_rnn_units_2']]
    cells = [tf.nn.rnn_cell.LSTMCell(num_units=n, forget_bias=forget_bias, initializer=init) for n in num_units]
    stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    enc_outputs, enc_final_state = tf.nn.dynamic_rnn(stacked_rnn_cell,
                                                     embedded_enc_input,
                                                     sequence_length=features['encoder_input_lengths'],
                                                     dtype=tf.float32)

    # Classifier
    init = tf.initializers.truncated_normal(0.0, .001)
    average_outputs = tf.reduce_mean(enc_outputs, axis=1)
    fcl = tf.layers.dense(average_outputs,
                          params['dense_1'],
                          activation=tf.nn.relu,
                          kernel_initializer=init)
    fc2 = tf.layers.dense(fcl,
                          params['dense_2'],
                          activation=tf.nn.relu,
                          kernel_initializer=init)
    class_logits = tf.layers.dense(fc2,
                                   params['num_classes'],
                                   kernel_initializer=init)
    probabilities = tf.nn.softmax(class_logits)
    pred_num_q = tf.argmax(probabilities, axis=1)

    one_hot_labels = tf.one_hot(labels['num_questions_labels'], params['num_classes'])
    crossentropy_loss = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels,
                                                   logits=class_logits))
    accuracy = tf.metrics.accuracy(labels=labels['num_questions_labels'], predictions=pred_num_q, name='accuracy_op')

    # Decoder model
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

    dec_cell = tf.nn.rnn_cell.LSTMCell(num_units=params['num_units'],
                                       forget_bias=forget_bias,
                                       initializer=init
                                       )
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=dec_cell,
        helper=helper,
        initial_state=enc_final_state[1],
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
    tf.summary.scalar('crossentropy_loss', crossentropy_loss)
    tf.summary.scalar('accuracy', accuracy)
    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
            'accuracy': accuracy,
            'sequence_loss': sequence_loss,
            'cross_entropy_loss': crossentropy_loss
        }
        return tf.estimator.EstimatorSpec(mode, loss=sequence_loss, eval_metric_ops=metrics)

    total_loss = sequence_loss + crossentropy_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=sequence_loss, train_op=train_op)


if __name__ == "__main__":
    """
    If we start from scratch, we'll make a new directory
     - if we name the directory, it takes the name, else, it is called 'test_directory'
    If we change hyperparameters, change the name of the output_dir.
    """
    args = gather_args()
    train_model(args, model_fn)
