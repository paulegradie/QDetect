import numpy as np


class Appender_(object):

    def __init__(self):
        self.test_encoder_inputs = list()
        self.test_encoder_input_lengths = list()
        self.test_decoder_inputs = list()
        self.test_decoder_input_lengths = list()
        self.test_target_sequences = list()
        self.test_target_seq_lengths = list()
        self.test_num_questions_labels = list()
        self.test_starts = list()
        self.test_stops = list()

    def append(self, features, labels):

        encoder_inputs, encoder_input_lengths, decoder_inputs, decoder_input_lengths = features
        target_sequences, target_seq_lengths, num_questions_labels, starts, stops = labels

        self.test_encoder_inputs.append(encoder_inputs)
        self.test_encoder_input_lengths.append(encoder_input_lengths)
        self.test_decoder_inputs.append(decoder_inputs)
        self.test_decoder_input_lengths.append(decoder_input_lengths)
        self.test_target_sequences.append(target_sequences)
        self.test_target_seq_lengths.append(target_seq_lengths)
        self.test_num_questions_labels.append(num_questions_labels)
        self.test_starts.append(starts)
        self.test_stops.append(stops)


    def export_data(self):

        features = np.vstack(self.test_encoder_inputs), np.vstack(self.test_encoder_input_lengths).reshape(-1), np.vstack(self.test_decoder_inputs), np.vstack(self.test_decoder_input_lengths).reshape(-1)
        labels = np.vstack(self.test_target_sequences), np.vstack(self.test_target_seq_lengths).reshape(-1), np.vstack(self.test_num_questions_labels), np.vstack(self.test_starts), np.vstack(self.test_stops)

        return features, labels
