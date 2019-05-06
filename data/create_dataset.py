import argparse
import os
import pickle
import sys

import tensorflow as tf
from tensorflow import logging as logging

from backend.container.generator_container import GeneratorDataContainer
from backend.container.static_container import StaticDataContainer

from data.data_gather import (gather_stack_exchange_data_from_scratch,
                                 gather_stack_exchange_from_file)
from backend.utils import process_glove
from backend.container_fillers.static_fillers import create_static_features
from backend.utils import set_data_to_new_vocab
from tensorflow import logging as logging

logging.set_verbosity(tf.logging.INFO)


XMLS_DICT = {'Badges.xml': None,
             'Comments.xml': 'Text',
             'PostHistory.xml': None,
             'PostLinks.xml': None,
             'Posts.xml': 'Body',
             'Tags.xml': None,
             'Users.xml': None,
             'Votes.xml': None}
STACKS = ['monero.stackexchange.com',
          'movies.stackexchange.com',
          'philosophy.stackexchange.com',
          'politics.stackexchange.com']


def run(args):
    # collect preprocessed data
    input_files = [args.questions, args.non_questions]
    for p in input_files:
        assert os.path.exists(p)

    if not os.path.exists(args.output_file_name) or args.overwrite:

        questions, non_questions = gather_stack_exchange_from_file(*input_files)
        glove_vocab, glove_vectors, glove_word2index = process_glove(glove_file=args.glove_path)

        (word2index,
         index2word,
         vectors,
         training_vocab) = set_data_to_new_vocab(questions,
                                                 non_questions,
                                                 glove_vectors,
                                                 glove_word2index)
        if args.static_data:
            features, labels = create_static_features(questions,
                                                      non_questions,
                                                      word2index,
                                                      size=args.num_samples,
                                                      max_seq_length=args.max_seq_length,
                                                      max_num_questions=args.max_num_questions,
                                                      max_num_elements=args.max_elements,
                                                      randomize_num_questions=args.rand_num_questions)
            data = StaticDataContainer(questions,
                                       non_questions,
                                       training_vocab,
                                       vectors,
                                       word2index,
                                       index2word,
                                       features,
                                       labels,
                                       args.num_samples,
                                       args.mini_num_samples,
                                       args.test_fraction,
                                       args.max_seq_length,
                                       args.max_elements,
                                       args.description
                                       )
            with open(args.output_file_name + '.static.pkl', 'wb+') as container:
                pickle.dump(data, container, protocol=-1)

        if args.gen_data:
            data = GeneratorDataContainer(questions,
                                          non_questions,
                                          training_vocab,
                                          vectors,
                                          word2index,
                                          index2word,
                                          args.max_seq_length,
                                          args.max_elements,
                                          args.max_num_questions,
                                          args.description,
                                          test_size=int(args.test_fraction * 10000),
                                          randomize_num_questions=args.rand_num_questions)
            with open(args.output_file_name + '.generator.pkl', 'wb+') as container:
                pickle.dump(data, container, protocol=-1)

    else:
        logging.warn('pickle files already exist, set overwrite to true if you want to refresh data; its randomly created each time')
        sys.exit()

    return data


if __name__ == "__main__":

    """
    Download stackoverflow data from:
    https://archive.org/details/stackexchange

    ** look for drop down on the right - choose 7z
    - comes in *.tar.7z
    - you can download 7zip from their downloads page for linux
    - you can unzip using bzip2
    - you can run 7z binary, for example: `p7zip_16.02/bin/7z e philosophy.stackexchange.com.7z`
    """

    parser = argparse.ArgumentParser(description='default sets to a tiny toy dataset - normal vocab size, 5 samples')
    parser.add_argument('-g', '--glove-path',         type=str,   default='./glove.6B.50d.txt', help='path to glove file')  # noqa
    parser.add_argument('-q', '--questions',          type=str,   default='./QUESTIONS.txt')  # noqa
    parser.add_argument('-n', '--non-questions',      type=str,   default='./NON_QUESTIONS.txt')  # noqa
    parser.add_argument('-s', '--num-samples',        type=int,   default=3000)  # noqa
    parser.add_argument('-m', '--max-seq-length',     type=int,   default=30)  # noqa
    parser.add_argument('-x', '--max-elements', type=int,   default=3, help='number of questions + answer used in input sequence')  # noqa
    parser.add_argument('-v', '--max-num-questions',  type=int,   default=1, help='max number of questions put in the input sequence')  # noqa
    parser.add_argument('-f', '--output-file-name',   type=str,   default='data_container')  # noqa
    parser.add_argument('-i', '--mini-num-samples',   type=int,   default=64)  # noqa
    parser.add_argument('-t', '--test-fraction',      type=float, default=0.1, help="percent to use for testing")  # noqa
    parser.add_argument('-a', '--rand-num-questions', action='store_true', help='whether or not to specify the number of questions used in input. If false, then random number of questions is used between zero and max_num_questions')  # noqa
    parser.add_argument('-o', '--overwrite',          action='store_true')  # noqa
    parser.add_argument('-e', '--supress-meta',       action='store_false')  # noqa
    parser.add_argument('-z', '--from-scratch',       action='store_true')  # noqa
    parser.add_argument('--static-data', action='store_true')
    parser.add_argument('--gen-data', action='store_true')
    parser.add_argument('-c', '--description', type=str, default=None)
    args = parser.parse_args()

    assert args.num_samples > args.mini_num_samples, "You must select more samples than mini_num_samples"
    assert args.static_data or args.gen_data, 'Need to choose static or generator type dataset'
    if args.static_data:
        assert int(args.num_samples - (args.num_samples * args.test_fraction)) > args.mini_num_samples, "Set test fraction lower or set num samples higher"

    if os.path.exists(args.output_file_name) and not args.overwrite:
        logging.warn(' mini pickle files already exist, set overwrite to true if you want to refresh data; its randomly created each time')
        sys.exit()

    if args.from_scratch:
        directory_str = os.path.join('.', 'stack_exchange', STACKS[2])
        filename = 'Comments.xml'
        _, _ = gather_stack_exchange_data_from_scratch(directory_str, filename, XMLS_DICT, write=True)

    data_obj = run(args)
    if args.supress_meta:
        data_obj.display_metadata()
