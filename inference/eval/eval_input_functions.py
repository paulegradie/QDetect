import tensorflow as tf

def eval_input_fn(test_features, test_labels, batch_size):
    " This will pass in the entire dataset -- the model decides what will be used "

    features = {
        'encoder_inputs': test_features[0],  # encoder_inputs
        'encoder_input_lengths': test_features[1],  # encoder_input_lengths
        'decoder_inputs': test_features[2],  # decoder_inputs
        'decoder_input_lengths': test_features[3]  # decoder_input_lengths
    }
    labels = {
        'target_sequences': test_labels[0],  # target_sequences
        'target_seq_lengths': test_labels[1],  # target_seq_lengths
        'num_questions': test_labels[2],  # num_questions
        'starts': test_labels[3], # question start position
        'stops': test_labels[4]  # question stop position
    }

    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)

    return dataset