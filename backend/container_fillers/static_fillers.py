import numpy as np
from backend.utils import _prepare_encoder_input_, _prepare_decoder_input_, _prepare_target_sequences_
from tqdm import tqdm
from backend.container_fillers.sequence_builder import _generate_sequence_
from functools import partial


def create_static_features(questions, non_questions, word2index,
                           size, max_seq_length, max_num_questions, max_num_elements, randomize_num_questions=False):

    encoder_inputs = list()
    encoder_input_lengths = list()
    decoder_inputs = list()
    decoder_input_lengths = list()
    target_sequences = list()
    target_seq_lengths = list()
    num_questions_labels = list()
    start_labels, stop_labels = list(), list()

    # to ensure no two questions togther reach the max_seq_length
    filtered_questions = list(filter(lambda x: len(x.split()) <= (max_seq_length // max_num_elements), questions))

    assert max_num_questions <= max_num_elements - 1, ' need to have at least 1 fewer questions than num element '  # to ensure that there will always be non-questions
    pbar = tqdm(total=size)
    while len(encoder_inputs) < size:

        if randomize_num_questions:  # if max elements is 3, then we can add 1 or 2 questions.
                max_num_questions = np.random.randint(1, max_num_elements - 1)

        input_sequence, target_sequence, num_questions, start, stop = _generate_sequence_(filtered_questions,
                                                                             non_questions,
                                                                             max_seq_length=max_seq_length,
                                                                             num_questions=max_num_questions,
                                                                             max_num_elements=max_num_elements)
        enc_input, enc_input_len = _prepare_encoder_input_(input_sequence, max_seq_length, word2index)
        encoder_inputs.append(enc_input)
        encoder_input_lengths.append(enc_input_len)

        dec_input, dec_input_len = _prepare_decoder_input_(target_sequence, max_seq_length, word2index)
        decoder_inputs.append(dec_input)
        decoder_input_lengths.append(dec_input_len)  # This seems weird, but... its the only thing that works.

        target_seq, target_seq_len = _prepare_target_sequences_(target_sequence, max_seq_length, word2index)
        target_sequences.append(target_seq)
        target_seq_lengths.append(target_seq_len)

        num_questions_labels.append([num_questions])
        start_labels.append(start)
        stop_labels.append(stop)
        pbar.update(1)

    pbar.close()

    if not encoder_inputs:
        print('No data, increase size (for now)')
        raise Exception

    ar = partial(np.array, dtype=np.int32)
    features = ar(encoder_inputs), ar(encoder_input_lengths), ar(decoder_inputs), ar(decoder_input_lengths)
    labels = ar(target_sequences), ar(target_seq_lengths), ar(num_questions_labels), ar(start_labels), ar(stop_labels)

    return features, labels