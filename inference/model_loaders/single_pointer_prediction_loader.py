
from functools import partial

import numpy as np
import tensorflow as tf
from backend.config import load_estimator_config, load_graph_params
from backend.utils import _prepare_encoder_input_, load_a_pickle
from data.parse_stack_exchange import _regex_, strip_and_lower
from inference.eval.eval_input_functions import eval_input_fn
from inference.eval.pointer_eval_metrics import \
    evaluate_single_pointer_predictions
from inference.prediction.prediction_input_functions import prediction_input_fn


class SinglePointerModelLoader(object):

    def __init__(self, model_fn, data_pickle, model_dir):

        self.model_fn = model_fn
        self.model_dir = model_dir
        self.estimator_config = load_estimator_config()
        self.data = load_a_pickle(data_pickle)

        self.batch_size = 1
        # params & configs

        self.graph_params = load_graph_params(batch_size=self.batch_size, **self.data.return_config())
        self.load_estimator()

    def predict_on_features(self, features):
        input_fn = partial(prediction_input_fn,
                           test_features=(np.array(features[0], dtype=np.int32),
                                          np.array(features[1], dtype=np.int32)),
                           batch_size=self.batch_size)

        prediction_gen = self.classifier.predict(input_fn=lambda: input_fn())
        predictions = [int(start['question_starts'][0]) for start in prediction_gen]

        results = [features[x][strt:] for x, strt in enumerate(predictions)]
        inputs = [" ".join(convert_int2word(self.data.index2word, feature)) for feature in features[0]]

        if len(result) == 1:
            results = zip(result.pop(), inputs.pop())

        return results, inputs

    def predict_single_pointer_string(self, inputs):
        " splits on spaces, can handle a list of sentences "
        assert type(inputs) == list or str
        if isinstance(inputs, str):
            inputs = [inputs]
        encoder_inputs, encoder_input_lengths = list(), list()
        for seq in inputs:

            clean_text = strip_and_lower(_regex_(seq))
            prepared_input, length = _prepare_encoder_input_(clean_text, self.data.max_seq_length_meta, self.data.word2index)
            encoder_inputs.append(prepared_input)
            encoder_input_lengths.append(length)

        input_fn = partial(prediction_input_fn,
                           test_features=(np.array(encoder_inputs, dtype=np.int32),
                                          np.array(encoder_input_lengths, dtype=np.int32)),
                           batch_size=self.batch_size)

        prediction_gen = self.classifier.predict(input_fn=lambda: input_fn())
        predictions = [int(start['question_starts'][0]) for start in prediction_gen]
        results = [" ".join(inputs[idx].strip().split()[strt:]) for idx, strt in enumerate(predictions)]

        if len(results) == 1:
            results = results.pop()

        return results, inputs

    def eval_single_pointer(self, show_first=10):
        " Data Pickle needs to have test_features and test_labels built "
        evaluation_input_fn = partial(eval_input_fn, test_features=self.data.test_features, test_labels=self.data.test_labels, batch_size=self.batch_size)
        eval_sp_predictions = partial(evaluate_single_pointer_predictions,
                                   validation_features=self.data.test_features,
                                   validation_labels=self.data.test_labels,
                                   index2word=self.data.index2word,
                                   show_first=show_first)

        predictions = self.classifier.predict(input_fn=lambda: evaluation_input_fn())
        bleu_score = eval_sp_predictions(predictions)
        return bleu_score


    def load_estimator(self):
        self.classifier = tf.estimator.Estimator(model_fn=self.model_fn,
                                                 params=self.graph_params,
                                                 config=self.estimator_config,
                                                 model_dir=self.model_dir)
