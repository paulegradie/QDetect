import os
from functools import partial

import tensorflow as tf
from backend.model_helpers import load_embeddings, get_forget_bias
from training.train_pointer import train_model
from training.utils import gather_args
from tensorflow import logging as logging
from tensorflow.python.layers.core import Dense
from tensorflow.python.client import device_lib

logging.set_verbosity(tf.logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def transform_to_range(input_val, min_value, max_value):
    scale = (max_value - min_value)
    rescaled = input_val * scale + min_value
    return rescaled


def model_fn(features, labels, mode, params):
    # particular to this project
    word2index = params['word2index']


    GPUs = get_available_gpus()
    CPUs = get_available_cpus()

    GPU = {
        'titan': GPUs[0],
        'sidekick': GPUs[1]}
    CPU = {
        'main_cpu': CPUs[0]
    }

    lookup_table, emb_vectors = load_embeddings(params['embedding_vectors'], params['vocab'])
    embedded_input = tf.nn.embedding_lookup(emb_vectors, features['encoder_inputs'])
    forget_bias = get_forget_bias(params, mode)

    positional_embeddings = tf.get_variable('position_embedding', shape=(params['input_max_length'], 50))
    positions = tf.range(features['encoder_input_lengths'])
    position_embeddings = tf.cast(tf.nn.embedding_lookup(positional_embeddings, positions), tf.float32)

    num_units = [256]
    init = tf.initializers.truncated_normal(0.0, 0.01)

    embedded_enc_input = tf.add(embedded_input, position_embeddings)



    with tf.device(GPU['titan']):
        init = tf.initializers.truncated_normal(0.0, 0.01)

        three_channel = tf.expand_dims(embedded_enc_input, axis=3)
        conv = tf.layers.conv2d(tf.cast(three_channel, tf.float32), 126, (5, 5), activation=tf.nn.relu, use_bias=True, kernel_initializer=init, name='conv1')
        conv = tf.layers.conv2d(conv, 32,  (3, 3), activation=tf.nn.relu, use_bias=True, kernel_initializer=init, name='conv2')
        conv = tf.layers.conv2d(conv, 16,  (3, 3), activation=tf.nn.relu, use_bias=True, kernel_initializer=init, name='conv3')
        conv = tf.layers.conv2d(conv, 8,  (3, 3), activation=tf.nn.relu, use_bias=True, kernel_initializer=init, name='conv4')
        conv = tf.layers.conv2d(conv, 16,  (3, 3), activation=tf.nn.relu, use_bias=True, kernel_initializer=init, name='conv5')
        conv = tf.layers.conv2d(conv, 32,  (3, 3), activation=tf.nn.relu, use_bias=True, kernel_initializer=init, name='conv6')


        # start_flat = tf.reduce_mean(start_transformed, axis=1, name='encoded_text')
        start_flat = tf.layers.flatten(conv6, name='start_flatten')
        # end_flat = tf.layers.flatten(end_transformed, name='end_flatten')

        # start pred
        start_hidden = tf.layers.dense(start_flat, units=params['input_max_length'])
        start_predictions = tf.layers.dense(start_hidden, units=1, activation=tf.nn.sigmoid, kernel_initializer=init())
        start_predictions_transformed = transform_to_range(start_predictions, min_value=0, max_value=params['input_max_length'])

        # end pred
        # end_input = tf.concat((start_predictions, end_flat), 1, name='end_input')
        # end_hidden = tf.layers.dense(end_input, units=params['input_max_length'])
        # end_predictions = tf.layers.dense(end_hidden, activation=tf.nn.sigmoid, use_bias=True, units=1)
            # end_predictions_transformed = transform_to_range(end_predictions, min_value=0, max_value=params['input_max_length'])


    if mode == tf.estimator.ModeKeys.PREDICT:

        starts = tf.to_int32(start_predictions_transformed)
        ends = tf.to_int32(end_predictions_transformed)

        predictions = {'question_starts': starts}#, 'question_ends': ends}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    starts = labels['starts']

    # compute losses
    print(labels['starts'])
    question_start_labels = tf.reshape(tf.to_float(labels['starts']), ( -1, 1))
    # question_end_labels = tf.reshape(tf.to_float(labels['stops']), (-1, 1))

    start_loss = tf.losses.mean_squared_error(labels=question_start_labels, predictions=start_predictions_transformed)
    # end_loss = tf.losses.mean_squared_error(labels=question_end_labels, predictions=end_predictions_transformed)

    # order_penalty = tf.cast(
    #     tf.divide(
    #         tf.cast(
    #             tf.nn.relu(start_predictions_transformed - end_predictions_transformed),
    #             tf.float32),
    #         tf.constant(10.0, tf.float32)
    #     ), tf.float32
    # )
    # zero_spread_penalty = tf.cast(tf.reduce_sum(tf.abs(start_predictions_transformed - end_predictions_transformed)), tf.float32)

    combined_loss = start_loss# + end_loss + tf.reduce_mean(order_penalty)# + zero_spread_penalty

    tf.summary.scalar('start_loss', start_loss)
    # tf.summary.scalar('end_loss', end_loss)
    # tf.summary.scalar('penalty_loss', tf.reduce_mean(order_penalty))
    # tf.summary.scalar('zero_spread_penalty', zero_spread_penalty)

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
            'start_loss': combined_loss
            }
        return tf.estimator.EstimatorSpec(mode, loss=combined_loss, eval_metric_ops=metrics)


    global_step = tf.train.get_global_step()

    # starter_learning_rate = 0.1
    # learning_rate = tf.train.exponential_decay(
    #     learning_rate=starter_learning_rate,
    #     global_step=global_step,
    #     decay_steps=100000,
    #     decay_rate=0.96,
    #     staircase=False,
    #     name='lr_decay_rate')
    learning_rate = 0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98, epsiolon=1e-09)
    gvs = optimizer.compute_gradients(combined_loss)
    capped_gvs = [(tf.clip_by_value(grad, -.5, .5), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=combined_loss, train_op=train_op)










    # with tf.device(GPU['titan']):


    #     init = tf.initializers.truncated_normal(0.0, 0.01)

    #     three_channel = tf.expand_dims(embedded_enc_input, axis=3)
    #     conv1 = tf.layers.conv2d(tf.cast(three_channel, tf.float32), 126, (5, 5), activation=tf.nn.relu, use_bias=True, kernel_initializer=init, name='conv1')
    #     conv2 = tf.layers.conv2d(conv1, 32,  (3, 3), activation=tf.nn.relu, use_bias=True, kernel_initializer=init, name='conv2')
    #     conv3= tf.layers.conv2d(conv2, 16,  (3, 3), activation=tf.nn.relu, use_bias=True, kernel_initializer=init, name='conv3')
    #     conv4 = tf.layers.conv2d(conv3, 8,  (3, 3), activation=tf.nn.relu, use_bias=True, kernel_initializer=init, name='conv4')
    #     conv5 = tf.layers.conv2d(conv4, 16,  (3, 3), activation=tf.nn.relu, use_bias=True, kernel_initializer=init, name='conv5')
    #     conv6 = tf.layers.conv2d(conv5, 32,  (3, 3), activation=tf.nn.relu, use_bias=True, kernel_initializer=init, name='conv6')


    #     # conv1_flat = tf.layers.flatten(conv1, name='flatten')
    #     # conv2_flat = tf.layers.flatten(conv2, name='flatten')
    #     # conv3_flat = tf.layers.flatten(conv3, name='flatten')
    #     # conv4_flat = tf.layers.flatten(conv4, name='flatten')


    #     flat = tf.layers.flatten(conv4, name='flatten')

    #     fcl = tf.layers.dense(flat, units=64)

    #     start_predictions = tf.layers.dense(fcl, units=1, activation=tf.nn.sigmoid)
    #     start_predictions_transformed = transform_to_range(start_predictions, min_value=0, max_value=params['input_max_length'])

    #     # concat start for influencing end prediction
    #     end_input = tf.concat((start_predictions, fcl), 1, name='end_input')
    #     end_predictions = tf.layers.dense(end_input, activation=tf.nn.sigmoid, use_bias=True, units=1)
    #     end_predictions_transformed = transform_to_range(end_predictions, min_value=0, max_value=params['input_max_length'])


    # if mode == tf.estimator.ModeKeys.PREDICT:

    #     starts = tf.to_int32(start_predictions_transformed)
    #     ends = tf.to_int32(end_predictions_transformed)

    #     predictions = {'question_starts': starts, 'question_ends': ends}
    #     return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # starts = labels['starts']
    # starts = tf.Print(starts, [starts], message="******: ")


    # # compute losses
    # print(labels['starts'])
    # question_start_labels = tf.reshape(tf.to_float(labels['starts']), ( -1, 1))
    # question_end_labels = tf.reshape(tf.to_float(labels['stops']), (-1, 1))

    # start_loss = tf.losses.mean_squared_error(labels=question_start_labels, predictions=start_predictions_transformed)
    # end_loss = tf.losses.mean_squared_error(labels=question_end_labels, predictions=end_predictions_transformed)

    # combined_loss = start_loss + end_loss

    # tf.summary.scalar('start_loss', start_loss)
    # tf.summary.scalar('end_loss', end_loss)

    # if mode == tf.estimator.ModeKeys.EVAL:
    #     metrics = {
    #         'start_loss': combined_loss
    #         }
    #     return tf.estimator.EstimatorSpec(mode, loss=combined_loss, eval_metric_ops=metrics)

    # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    # train_op = optimizer.minimize(combined_loss, global_step=tf.train.get_global_step())

    # return tf.estimator.EstimatorSpec(mode, loss=combined_loss, train_op=train_op)


if __name__ == "__main__":
    """
    If we start from scratch, we'll make a new directory
     - if we name the directory, it takes the name, else, it is called 'test_directory'
    If we change hyperparameters, change the name of the output_dir.
    """
    args = gather_args()
    train_model(args, model_fn)
