import argparse
import os
import random as r
import warnings

import tensorflow as tf
from tensorflow import logging as logging

from data.parse_stack_exchange import (get_texts, parse_xml_doc,
                                          preprocess_a_text, write_data)

logging.set_verbosity(tf.logging.INFO)

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


XMLS = {'Badges.xml': None,
        'Comments.xml': 'Text',
        'PostHistory.xml': None,
        'PostLinks.xml': None,
        'Posts.xml': 'Body',
        'Tags.xml': None,
        'Users.xml': None,
        'Votes.xml': None}
DIRECTORY = 'stack_exchange'
STACK_TYPES = ['monero.stackexchange.com',
               'movies.stackexchange.com',
               'philosophy.stackexchange.com',
               'politics.stackexchange.com']


def gather_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-datatype',     '-d', help='datatype from XMLS above',    type=str, default='Comments.xml')  # noqa
    parser.add_argument('-dont-shuffle', '-s', help='whether to shuffle data',     action='store_false')  # noqa
    parser.add_argument('-stacktype',    '-t', help='which stack exchange to use', type=int, default=2)  # noqa

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = gather_args()

    logging.basicConfig(filename='data.log', level=logging.INFO)
    logging.info('args used: {}'.format(args))

    path = os.path.join(DIRECTORY, STACK_TYPES[args.stacktype], args.datatype)
    docs = parse_xml_doc(path, XMLS[args.datatype])

    preprocessed_docs = [preprocess_a_text(doc) for doc in docs]

    questions = get_texts(preprocessed_docs, get_questions=True)
    non_questions = get_texts(preprocessed_docs, get_questions=False)
    logging.info("Original number of samples in each set: Q: {}, NonQ: {}".format(str(len(questions)), str(len(non_questions))))

    if not args.dont_shuffle:
        logging.info('Data was shuffled')
        r.shuffle(questions)
        r.shuffle(non_questions)

    # equalize list lengths
    non_questions = non_questions[:len(questions)]

    write_data(questions, non_questions)
    logging.info("Final number of questions: {}".format(str(len(questions))))
    logging.info('Data preparation complete')
