
import tensorflow as tf
from backend.container.base_container import BaseContainer
from backend.special_tokens import SPECIAL_TOKEN_IDS
from tensorflow import logging as logging

logging.set_verbosity(tf.logging.INFO)


FEATURE_NAMES = ['encoder_inputs', 'encoder_input_lengths', 'decoder_inputs', 'decoder_input_lengths']
LABEL_NAMES = ['target_sequences', 'target_seq_lengths', 'num_questions_labels']


class StaticDataContainer(BaseContainer):
    " class to make it easy to pickle all this data together "
    def __init__(self,
                 questions,
                 non_questions,
                 training_vocab,
                 vectors,
                 word2index,
                 index2word,
                 features,
                 labels,
                 num_samples,
                 mini_num_samples,
                 test_fraction,
                 max_seq_length,
                 max_input_elements,
                 description):
        super()
        # normal dataset
        self.questions = questions
        self.non_questions = non_questions
        self.vocab = training_vocab
        self.vectors = vectors
        self.word2index = word2index
        self.index2word = index2word
        self.embedding_dims_meta = len(vectors[0])
        self.mini_num_samples_meta = mini_num_samples
        self.num_samples_meta = num_samples
        self.max_seq_length_meta = max_seq_length
        self.max_input_elements_meta = max_input_elements
        self._is_split_ = False
        self.test_fraction = test_fraction
        self.description_meta = description
        self.max_index_meta = max(word2index.values())

        # Split data
        (self.train_features,
         self.train_labels,
         self.test_features,
         self.test_labels) = self._test_train_split_(features, labels)

        # create mini dataset (uses splitted data internally)
        self.mini_features, self.mini_labels = self._set_mini_dataset_(mini_num_samples)

        self._meta_set_ = False
        self._set_meta_data_()
        assert max(word2index.values()) == self.vocab_size_meta - 1

    def _sample_for_mini_(self, sample_indices, *args):
        sampled = list()
        for arg in args:
            sampled.append(arg[sample_indices])
        return tuple(sampled)

    def _set_mini_dataset_(self, num_samples):
        assert self._is_split_, 'Split your data b4 making mini dataset'

        sample_indices = np.random.choice(list(range(self.mini_num_samples_meta)), size=num_samples, replace=False)
        mini_features = self._sample_for_mini_(sample_indices, *self.train_features)
        mini_labels = self._sample_for_mini_(sample_indices, *self.train_labels)

        self.mini_dataset_set = True

        return mini_features, mini_labels

    def _set_meta_data_(self):
        self.embedding_dimensions = len(self.vectors[0])
        if self.mini_dataset_set:
            self.vocab_size_meta = len(self.vocab)
            self._meta_set_ = True
        else:
            logging.warning('Metadata was not set. Make mini dataset')

    def _split_(self, num_train, *args):
        train_arg = list()
        test_arg = list()
        for arg in args:
            train_arg.append(arg[:num_train])
            test_arg.append(arg[num_train:])
        return train_arg, test_arg

    def _test_train_split_(self, features, labels):
        num_train = int(len(features[0]) * (1. - self.test_fraction))
        train_features, test_features = self._split_(num_train, *features)
        train_labels, test_labels = self._split_(num_train, *labels)
        self._is_split_ = True
        return train_features, train_labels, test_features, test_labels

    def show_random_sample_of_data(self, n_samples, mini=False, words=True):
        assert isinstance(n_samples, int), 'must pass int'
        max_size = self.num_samples_meta - int(self.num_samples_meta * self.test_fraction) if not mini else self.mini_num_samples_meta
        indices = np.random.choice(list(range(max_size)),
                                   size=min(n_samples, max_size), replace=False)

        features = self.train_features if not mini else self.mini_features
        labels = self.train_labels if not mini else self.mini_labels

        def show_data(data, names, words, indices, index2word):
            for datam, name in zip(data, names):
                if len(datam.shape) == 1:
                    pass
                elif words:
                    datam = np.array([[' '.join([index2word[x] for x in y if x not in SPECIAL_TOKEN_IDS[:-1]])] for y in datam])
                # TODO still prints out empty string with the num_question_labels.
                print("{}".format(name))
                messg = ' (mini)' if mini else ''
                print("-- dataset{} indices: {}\n{}".format(messg, indices, datam[indices]))
                print()

        print("FEATURES\n")
        show_data(features, FEATURE_NAMES, words, indices, self.index2word)
        print("\nLABELS\n")
        show_data(labels, LABEL_NAMES, words, indices, self.index2word)

    def display_metadata(self):
        if not self._meta_set_:
            print("Metadata was not set")
            return
        print('\n--DATASET METADATA --')
        for key in sorted(list(self.__dict__.keys())):
            if key.endswith('_meta'):
                print('{}: {}'.format(key[:-5], self.__dict__[key]))
