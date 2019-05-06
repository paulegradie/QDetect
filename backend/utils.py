import os
import pickle
import shutil
import sys
from itertools import chain
from collections import OrderedDict
import numpy as np

from backend.special_tokens import SPECIAL_TOKENS, SPECIAL_TOKEN_IDS
from tensorflow import logging as logging
import io


def append_special_tokens(embedding_dim, vocabulary, vectors):
    special_vectors = [[0] * embedding_dim] * len(SPECIAL_TOKENS[:-1])  # check input file
    special_vectors += [vectors[vocabulary.index('<UNK>')]]

    try:
        filt_vocab = vocabulary.remove('<UNK>')
    except Exception as e:
        filt_vocab = vocabulary

    vocab = SPECIAL_TOKENS + list(filt_vocab)
    vecs = special_vectors + list(vectors)
    return vocab, np.array(vecs, dtype=np.float32)


def process_glove(glove_file='./glove.6B/glove.6B.50d.txt', filter_list=None, encoding='latin-1'):

    def _process_glove_line_(line):
        line = line.strip().split()
        word = line[0]
        try:
            vector = [float(x) for x in line[1:]]
        except ValueError:
            return None, None
        return word, vector

    with io.open(glove_file, 'r', encoding=encoding) as g:
        glove = g.readlines()

    vocabulary = list()
    vectors = list()
    for line in glove:
        word, vector = _process_glove_line_(line)
        if not word:
            continue
        if filter_list:
            if word not in filter_list:
                continue
        vocabulary.append(word)
        vectors.append(vector)

    word2index = OrderedDict({word: idx for idx, word in enumerate(vocabulary, 0)})

    return vocabulary, vectors, word2index


def get_raw_vocab(question_path, non_question_path, encoding='latin-1'):
    with io.open(question_path, 'r', encoding=encoding) as qs, io.open(non_question_path, 'r', encoding=encoding) as nqs:
        corpus = ' '.join([x.strip() for x in qs.readlines()] + [x.strip() for x in nqs.readlines()]).split()
    return list(set(corpus))


def read_files(question_path, non_question_path, encoding='latin-1'):
    with io.open(question_path, 'r', encoding=encoding) as qs, io.open(non_question_path, 'r', encoding=encoding) as nqs:
        questions = [x.strip() for x in qs.readlines()]
        non_questions = [x.strip() for x in nqs.readlines()]
    return questions, non_questions


def _truncat_long_sequence_(sequences, max_element_length):
    return [" ".join(x.split()[:max_element_length]) for x in sequences]


def _pad_sequence_(sequence, max_length, word2index):
    num_to_pad = max_length - len(sequence)
    if num_to_pad > 0:
        pads = [word2index['<PAD>']] * num_to_pad
        padded_seq = sequence + pads
    else:
        padded_seq = sequence
    return padded_seq


def _convert_to_ints_(input_sequence, word2index):
    return [word2index.get(x, word2index.get('<UNK>')) for x in input_sequence.split()]


def _prepare_encoder_input_(input_sequence, max_length, word2index):
    tokenized = _convert_to_ints_(input_sequence, word2index)
    padded = _pad_sequence_(tokenized, max_length, word2index)
    return np.array(padded, dtype=np.int32), len(tokenized)


def _prepare_decoder_input_(target_sequence, max_length, word2index):
    tokenized = _convert_to_ints_(target_sequence, word2index)[:max_length - 1]
    go_tokenized = [word2index['<GO>']] + tokenized
    padded = _pad_sequence_(go_tokenized, max_length, word2index)
    return np.array(padded, dtype=np.int32), max_length  # len(go_tokenized), only thing that works - gotta have them all the same I guess.


def _prepare_target_sequences_(target_sequence, max_length, word2index):
    tokenized = _convert_to_ints_(target_sequence, word2index)[:max_length - 1]
    eos_tokenized = tokenized + [word2index['<EOS>']]
    padded = _pad_sequence_(eos_tokenized, max_length, word2index)
    return np.array(padded, dtype=np.int32), len(eos_tokenized)


def _make_breaks(input_list):
    starts = list()
    stops = list()

    start = 0
    for length, sentence in input_list:
        stop = start + length -1

        if sentence.endswith('<QQQ>'):
            starts.append(start)
            stops.append(stop)

        start += length

    return starts[0], stops[0]


def convert_int2word(index2word, sequence, ints_to_exclude=SPECIAL_TOKEN_IDS[:-1]):
    return [index2word[x] for x in sequence if x not in ints_to_exclude]


def check_response(model_dir):
    if len(os.listdir(model_dir)) > 2:
        response = input('Continue training? y/n> ')
        if not response == 'y':
            print("Canceling training run. Chose not to continue training existing model.")
            sys.exit()


def make_model_dir(name='model_test', starting_dir='.', reuse=True, overwrite=False, digit=0):
    output_dir = os.path.join(starting_dir, name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        logging.info('Making new Model Directory: {}'.format(output_dir))
        return os.path.abspath(output_dir)
    # elif reuse:
    #     return output_dir
    else:
        if overwrite:
            shutil.rmtree(output_dir, ignore_errors=True)
            os.mkdir(output_dir)
            logging.info('Model Directory overwritten: {}'.format(output_dir))
            return os.path.abspath(output_dir)

        else:
            output_dir = os.path.join(starting_dir, name + '_{}'.format(str(digit)))
            if os.path.exists(output_dir):
                digit += 1
                return make_model_dir(name, starting_dir, overwrite, digit)
            else:
                os.mkdir(output_dir)
                return os.path.abspath(output_dir)


def load_a_pickle(picklefile):
    with open(picklefile, 'rb') as fin:
        data = pickle.load(fin)
    return data


def set_data_to_new_vocab(questions, non_questions, glove_vectors, glove_word2index):
    training_vocab = list(set(" ".join(chain([*questions, *non_questions])).split()))

    aa = np.zeros(shape=(len(SPECIAL_TOKENS[:-1]), len(glove_vectors[0])), dtype=np.float32)
    bb = np.random.normal(0, 2, size=(len(glove_vectors[0]))).astype(np.float32)
    special_vectors = np.vstack([aa, bb])

    word2index = dict()
    vectors = list()
    available_vocab = list()

    # training vocab has 43k words. but not all are in the glove vectors vocab...
    for word in training_vocab:
        available_vocab.append(word)
        try:
            vectors.append(glove_vectors[glove_word2index[word]])
        except KeyError:
            available_vocab.pop()

    # make word2index AFTER
    word2index = {word: idx for idx, word in enumerate(SPECIAL_TOKENS + available_vocab)}
    available_vocab = SPECIAL_TOKENS + available_vocab
    available_vectors = np.vstack([special_vectors, np.array(vectors, dtype=np.float32)])

    index2word = {index: word for word, index in word2index.items()}

    return word2index, index2word, available_vectors.astype(np.float32), available_vocab
