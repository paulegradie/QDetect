import os

import tensorflow as tf
from backend.container_fillers.generator_queue_fillers import GeneratorInputV3
from tensorflow import logging as logging

logging.set_verbosity(tf.logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def train_generator_input_fn_v3(questions, non_questions, input_fn_params, graph_params):
    " Main input function -- use this one "
    kwargs = {
        'questions': questions,
        'non_questions': non_questions,
        'Q_size': 700,
        'num_proc': 6,
        'word2index': graph_params['word2index'],
        'max_seq_length': graph_params['input_max_length'],
        'max_num_questions': input_fn_params['max_num_questions'],
        'max_num_elements': input_fn_params['max_num_elements'],
        'randomize_num_questions': input_fn_params['randomize_num_questions']
    }
    gen_obj = GeneratorInputV3(**kwargs)  # This should take max a few seconds to prefill the queue
    generator = gen_obj.from_queue_generator()

    msl = graph_params['input_max_length']
    bsize = input_fn_params['batch_size']

    dataset = tf.data.Dataset.from_generator(lambda: generator,
                                             output_types=((tf.int32, tf.int32, tf.int32, tf.int32),
                                                           (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)),
                                             output_shapes=(
                                                 (tf.TensorShape([msl]), tf.TensorShape([]), tf.TensorShape([msl]), tf.TensorShape([])),
                                                 (tf.TensorShape([msl]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]))
                                                 )
    )
    dataset = dataset.batch(bsize).prefetch(int(bsize * 3))
    feature, label = dataset.make_one_shot_iterator().get_next()

    features = {
        'encoder_inputs': feature[0],  # encoder_inputs
        'encoder_input_lengths': feature[1],  # encoder_input_lengths
        'decoder_inputs': feature[2],  # decoder_inputs
        'decoder_input_lengths': feature[3]  # decoder_input_lengths
    }
    labels = {
        'target_sequences': label[0],  # target_sequences
        'target_seq_lengths': label[1],  # target_seq_lengths
        'num_questions': label[2],  # num_questions
        'starts': label[3], # question start position
        'stops': label[4]  # question stop position
    }
    return features, labels



def train_static_input_fn(features, labels, input_fn_params):
    " Static features and labels loaded from data.features, data.labels "

    (encoder_inputs, encoder_input_lengths,
     decoder_inputs, decoder_input_lengths) = features
    (target_sequences, target_seq_lengths, num_questions_labels, starts, stops) = labels

    data_dict = {
        'encoder_inputs': encoder_inputs,
        'encoder_input_lengths': encoder_input_lengths,
        'decoder_inputs': decoder_inputs,
        'decoder_input_lengths': decoder_input_lengths,
    }
    labels_dict = {
        'num_questions_labels': num_questions_labels,
        'target_sequences': target_sequences,
        'target_seq_lengths': target_seq_lengths,
        'starts': starts,
        'stops': stops
    }
    dataset = tf.data.Dataset.from_tensor_slices((data_dict, labels_dict))
    dataset = (dataset.shuffle(input_fn_params['shuffle']).repeat(input_fn_params['repeat']).batch(input_fn_params['batch_size']))

    return dataset
