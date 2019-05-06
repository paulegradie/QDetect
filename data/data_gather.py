import os
from random import shuffle

import tensorflow as tf
from tensorflow import logging as logging

from data.parse_stack_exchange import (parse_xml_doc, process_text,
                                          remove_html, strip_and_lower,
                                          write_data)
from backend.utils import read_files

logging.set_verbosity(tf.logging.INFO)


def gather_stack_exchange_data_from_scratch(path_to_stack_dir, filename, xmls_dict, write=True):
    path_to_file = os.path.join(path_to_stack_dir, filename)
    dataname_str = xmls_dict[filename]

    docs = parse_xml_doc(path_to_file, dataname_str)

    # preprocessed_docs = [preprocess_a_text(doc) for doc in docs]
    preprocessed = [remove_html(x) for x in docs]
    strip_lower = [strip_and_lower(x) for x in preprocessed]

    prepard_questions = process_text(strip_lower, get_questions=True)
    prepard_non_questions = process_text(strip_lower, get_questions=False)

    max_len = min(len(prepard_questions), len(prepard_non_questions))
    questions = prepard_questions[:max_len]
    non_questions = prepard_non_questions[:max_len]

    shuffle(questions)
    shuffle(non_questions)

    if write:
        write_data(questions, non_questions)

    return questions, non_questions


def gather_stack_exchange_from_file(qfile_st="./data/ALL_QUESTIONS.txt",
                                    nqfile_st="./data/NON_QUESTIONS.txt"):
    return read_files(qfile_st, nqfile_st)
