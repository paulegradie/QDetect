MAX_SEQ_LENGTH = None
import tensorflow as tf


def serving_input_receiver_fn():
    feature_placeholders = {
        'encoder_inputs': tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH]),
        'encoder_lengths': tf.placeholder(tf.int32, [None])}
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()}

    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)
