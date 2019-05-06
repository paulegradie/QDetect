import numpy as np
import tensorflow as tf
from tensorflow import logging as logging
from tqdm import tqdm
from random import shuffle
from backend.container.base_container import BaseContainer
from backend.container.appender import Appender_
from backend.utils import _make_breaks, _prepare_encoder_input_, _prepare_decoder_input_, _prepare_target_sequences_


from functools import partial
tf.logging.set_verbosity(tf.logging.INFO)


class GeneratorDataContainer(BaseContainer):
    def __init__(
        self,
        questions,
        non_questions,
        training_vocab,
        vectors,
        word2index,
        index2word,
        max_seq_length,
        max_elements,  # This is the total number of questions and non questions in a list
        max_num_questions,
        description,
        test_size=1000,
        randomize_num_questions=False
    ):
        super()
        # normal dataset
        self.questions = questions
        self.non_questions = non_questions
        self.vocab = training_vocab
        self.vocab_size_meta = len(training_vocab)
        self.vectors = vectors
        self.word2index = word2index
        self.index2word = index2word
        self.embedding_dims_meta = len(vectors[0])
        self.max_seq_length_meta = max_seq_length
        self.max_num_questions_meta = max_num_questions
        self.max_elements_meta = max_elements
        self.description_meta = description
        self.max_index_meta = max(word2index.values())
        self.randomize_num_questions = randomize_num_questions

        self._create_test_set_(test_size)

    def return_config(self):
        config = {
            'vocab_size_meta': self.vocab_size_meta,
            'word2index': self.word2index,
            'index2word': self.index2word,
            'embedding_dims_meta': self.embedding_dims_meta,
            'max_seq_length_meta': self.max_seq_length_meta,
            'max_num_questions_meta': self.max_num_questions_meta,
            'max_elements_meta': self.max_elements_meta,
            'description_meta': self.description_meta,
            'max_index_meta': self.max_index_meta,
            'randomize_num_questions': self.randomize_num_questions,
            'questions': self.questions,
            'non_questions': self.non_questions,
            'vectors': self.vectors,
            'vocab': self.vocab
            }
        return config

    def _instantiate_sample_generator(self):
        simple_gen = SimpleGenerator()
        sampler = simple_gen.generate(
            self.questions,
            self.non_questions,
            self.word2index,
            self.max_seq_length_meta,
            self.max_num_questions_meta,
            self.max_elements_meta,
            self.randomize_num_questions)
        return sampler

    def _create_test_set_(self, test_size):
        test_size = int(test_size)

        datapoints = Appender_()
        sampler = self._instantiate_sample_generator()

        for i in tqdm(range(test_size)):
            features, labels = next(sampler)
            datapoints.append(features, labels)

        test_features, test_labels = datapoints.export_data()
        self.test_features = test_features
        self.test_labels = test_labels

    def show_random_sample_of_data(self, sample_size):
        sampler = self._instantiate_sample_generator()
        for _ in range(sample_size):
            yield next(sampler)  #tuple (feature, label)

    def display_metadata(self):
        print('\n--DATASET METADATA --')
        for key in sorted(list(self.__dict__.keys())):
            if key.endswith('_meta'):
                print('{}: {}'.format(key[:-5], self.__dict__[key]))




class SimpleGenerator(object):

    """
    Non-multiprocessed/queue filling generator class for printing samples from the generator data container
    """
    def __init__(self):
        pass

    def generate(self, questions, non_questions, word2index,
                 max_seq_length, max_num_questions, max_num_elements, randomize_num_questions=False):
        " this shits gotta be callable... and graph params are needed for feature generator arguments "

        # to ensure no two questions togther reach the max_seq_length
        filtered_questions = list(filter(lambda x: len(x.split()) <= max_seq_length // max_num_elements, questions))
        shuffle(filtered_questions)

        assert max_num_questions <= max_num_elements - 1, ' need to have at least 1 fewer questions than num element '  # to ensure that there will always be non-questions
        while True:
            if randomize_num_questions:  # if max elements is 3, then we can add 1 or 2 questions.
                    max_num_questions = np.random.randint(1, max_num_elements - 1)

            input_sequence, target_sequence, num_questions, starts_list, stops_list = self._make_sequence(filtered_questions,
                                                                                 non_questions,
                                                                                 max_seq_length=max_seq_length,
                                                                                 num_questions=max_num_questions,
                                                                                 max_num_elements=max_num_elements)

            # if _is_acceptable_(input_sequence, max_seq_length):
            encoder_inputs, encoder_input_lengths = _prepare_encoder_input_(input_sequence, max_seq_length, word2index)
            decoder_inputs, decoder_input_lengths = _prepare_decoder_input_(target_sequence, max_seq_length, word2index)
            target_sequences, target_seq_lengths = _prepare_target_sequences_(target_sequence, max_seq_length, word2index)

            array = partial(np.array, dtype=np.int32)
            features = array(encoder_inputs), array(encoder_input_lengths), array(decoder_inputs), array(decoder_input_lengths)
            labels = array(target_sequences), array(target_seq_lengths), array(num_questions), array(starts_list), array(stops_list)

            yield features, labels


    def _make_sequence(self, questions, non_questions, max_seq_length, num_questions, max_num_elements):
        """
        - Aim to generate seq shorter than max_seq_length by filtering as data is processed
        """
        assert max_num_elements >= num_questions, 'must have more elements than questions'

        num_non_questions = max_num_elements - num_questions

        # keep track of questions for label making
        pre_choices = np.random.choice(questions, num_questions, replace=False).tolist()
        question_choices = [''.join([x, '<QQQ>']) for x in pre_choices]
        length_of_questions = sum([len(x.split()) for x in question_choices])

        filtered_non_questions = list(filter(lambda x: len(x.split()) <= int((max_seq_length - length_of_questions) / num_non_questions), non_questions))
        shuffle(filtered_non_questions)

        non_question_choices = np.random.choice(filtered_non_questions, num_non_questions, replace=False).tolist()

        # New code
        input_list = [(len(x.split()), x) for x in question_choices] + [(len(x.split()), x) for x in non_question_choices]

        shuffle(input_list)

        # New code
        starts, stops = _make_breaks(input_list)
        input_list = [x[1] for x in input_list]

        input_sequence = ' '.join(input_list).replace('<QQQ>', '')
        target_sequence = ' '.join(
            [x.replace('<QQQ>', '') for x in input_list if '<QQQ>' in x])
        return input_sequence, target_sequence, num_questions, starts, stops

