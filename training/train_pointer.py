import os
import pickle
import sys
from argparse import ArgumentParser
from functools import partial

import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
from tensorflow import logging as logging

from backend.container.arg_container import ArgContainer
from backend.config import (load_estimator_config, load_graph_params,
                            load_input_fn_params)

from inference.eval.eval_input_functions import eval_input_fn
from training.training_input_functions import (
    train_generator_input_fn_v3,
    train_static_input_fn)
from backend.utils import check_response, make_model_dir

from tensorflow.python.training import basic_session_run_hooks

import logging

from inference.eval.pointer_eval_metrics import evaluate_double_pointer_predictions, evaluate_single_pointer_predictions

from training.utils import gather_args

tf.logging.set_verbosity(tf.logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def read_prev_pids(id_file='temp_pid'):
    try:
        with open(id_file, 'r') as pin:
            pid_list = [int(x.strip()) for x in pin.readlines()]
    except IOError:
        pid_list = None
    return pid_list


def kill_prev_pids(pid_list):
    if pid_list:
        try:
            for pid in pid_list:
                os.kill(pid, signal.SIGTERM)
        except Exception as e:
            pass
        return True
    else:
        print('no pids found')
        return False


def train_model(args, model_fn):
    datafile = args.gen_pickle or args.static_pickle
    with open(datafile, 'rb') as pick:
        data = pickle.load(pick)

    # output dir
    _name = '_'.join([args.model_dir, 'mini']) if args.minimode else args.model_dir
    model_dir = make_model_dir(name=_name, overwrite=args.overwrite if isinstance(args, ArgContainer) else False)

    print('Experiment name: {}'.format(model_dir))

    # params & configs
    graph_params = load_graph_params(batch_size=args.batch_size, **data.return_config())
    input_fn_params = load_input_fn_params(
        batch_size=args.batch_size if not args.minimode else 16,
        repeat=-1
    )
    estimator_config = load_estimator_config(save_every=100, log_every=100)

    # load estimator
    classifier = tf.estimator.Estimator(model_fn=model_fn, params=graph_params, config=estimator_config, model_dir=model_dir)

    # setup up input fn partials
    if args.gen_pickle:

        train_input_fn = partial(train_generator_input_fn_v3,  # train_generator_input_fn  - v2 uses multi process
                                 questions=data.questions,
                                 non_questions=data.non_questions,
                                 input_fn_params=input_fn_params,
                                 graph_params=graph_params)
        evaluation_input_fn = partial(eval_input_fn, test_features=data.test_features, test_labels=data.test_labels, batch_size=args.batch_size)
        eval_predictions = partial(evaluate_double_pointer_predictions,
                                   validation_features=data.test_features,
                                   validation_labels=data.test_labels,
                                   index2word=data.index2word)

    else:
        input_features = data.mini_features if args.minimode else data.train_features
        input_labels = data.mini_labels if args.minimode else data.train_labels

        validation_features = data.mini_features if args.minimode else data.test_features
        validation_labels = data.mini_labels if args.minimode else data.test_labels

        train_input_fn = partial(train_static_input_fn,
                                 features=input_features,
                                 labels=input_labels,
                                 input_fn_params=input_fn_params)
        evaluation_input_fn = partial(eval_input_fn, test_features=validation_features, test_labels=validation_labels, batch_size=args.batch_size)
        eval_predictions = partial(evaluate_double_pointer_predictions,
                                   validation_features=validation_features,
                                   validation_labels=validation_labels,
                                   index2word=data.index2word)

    # run estimator
    if 'train' in args.command:
        check_response(model_dir)

    if args.command == 'train':
        classifier.train(steps=args.steps, input_fn=lambda: train_input_fn())

    elif args.command == 'train_and_eval':
        epochs = int(args.steps // args.eval_every)
        for epoch in range(epochs):
            classifier.train(steps=args.eval_every,
                             input_fn=lambda: train_input_fn())
            predictions = classifier.predict(input_fn=lambda: evaluation_input_fn())
            eval_predictions(predictions, show_first=args.show_sample)

    elif args.command == 'train_until_thresh':
        bleu_score = [0]
        while np.mean(bleu_score) < args.threshold:
            classifier.train(steps=args.steps,
                             input_fn=lambda: train_input_fn())
            predictions = classifier.predict(input_fn=lambda: evaluation_input_fn())
            bleu_score = eval_predictions(predictions, show_first=args.show_sample)

    elif args.command == 'eval':
        predictions = classifier.predict(input_fn=lambda: evaluation_input_fn())
        eval_predictions(predictions, show_first=args.show_sample)

    elif args.command == 'debug':
        classifier.train(steps=1, input_fn=lambda: train_input_fn())

    else:
        sys.exit()

    # clean up experiments
    pid_list = read_prev_pids('temp_pid')
    print('pids: {}'.format(pid_list))

    res = kill_prev_pids(pid_list)

    if res:
        os.remove('temp_pid')


    print("Done")
    print("proccesses killed: {}".format(pid_list))

    sys.exit('Training ran successfully.')