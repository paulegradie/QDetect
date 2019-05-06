import tensorflow as tf
import numpy as np

def prediction_input_fn(test_features, batch_size):
    " data params is a dict "
    try:
        encoder_inputs, encoder_input_lengths, _, _ = test_features
    except ValueError:
        encoder_inputs, encoder_input_lengths = test_features

    data_dict = {
        'encoder_inputs': encoder_inputs,
        'encoder_input_lengths': np.array(encoder_input_lengths, dtype=np.int32)
    }
    dataset = tf.data.Dataset.from_tensor_slices((data_dict)).batch(batch_size)

    return dataset
