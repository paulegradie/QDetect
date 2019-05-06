
from inference.prediction.prediction_input_functions import prediction_input_fn
from backend.config import load_estimator_config, load_graph_params
from backend.utils import load_a_pickle
import tensorflow as tf
from functools import partial
from inference.eval.eval_input_functions import eval_input_fn
from inference.eval.pointer_eval_metrics import evaluate_double_pointer_predictions
import tensorflow as tf


class DoublePointerModelLoader(object):

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
        """
        features: a data_container set of input features, like data.test_features
            or similar dictionary
        """
        prediction_gen = self.classifier.predict(input_fn=lambda: input_fn(
            test_features=(np.array(features[0], dtype=np.int32), np.array(features[1], dtype=np.int32)),
            batch_size=self.batch_size)
        )
        predictions = [(int(start), int(stop)) for start, stop in prediction_gen]

        results = [features[x][strt:stp] for x, (strt, stp) in enumerate(predictions)]
        inputs = [" ".join(convert_int2word(self.data.index2word, feature)) for feature in features[0]]

        if len(result) == 1:
            results = zip(result.pop(), inputs.pop())

        return results, inputs

    def predict_on_strings(self, inputs):
        """
        intputs: single string or list of strings
        """
        assert type(inputs) == list or str
        if isinstance(inputs, str):
            inputs = [inputs]

        encoder_inputs, encoder_input_lengths = list(), list()
        for seq in inputs:
            clean_text = strip_and_lower(_regex_(seq))
            prepared_input, length = _prepare_encoder_input_(clean_text, self.data.max_seq_length_meta, self.data.word2index)
            encoder_inputs.append(prepared_input)
            encoder_input_lengths.append(length)

        prediction_gen = self.classifier.predict(input_fn=lambda: input_fn(
            test_features=(np.array(encoder_inputs, dtype=np.int32), np.array(encoder_input_lengths, dtype=np.int32)),
            batch_size=self.batch_size)
        )

        predictions = [int(start_stop['question_starts'][0]) for start_stop in prediction_gen]
        results = [" ".join(inputs[idx].strip().split()[strt:]) for idx, strt in enumerate(predictions)]

        if len(results) == 1:
            results = results.pop()

        return results, inputs

    def predict_double_pointer_string(self, inputs):
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

        input_fn = partial(pred_input_fn,
                           test_features=(np.array(encoder_inputs, dtype=np.int32),
                                          np.array(encoder_input_lengths, dtype=np.int32)),
                           batch_size=self.batch_size)

        prediction_gen = self.classifier.predict(input_fn=lambda: input_fn())
        # import pdb; pdb.set_trace()

        predictions = [(int(start_stop['question_starts'][0]), int(start_stop['question_ends'][0])) for start_stop in prediction_gen]
        # preds = [start, stop for]
        # predictions = [(int(start), int(stop)) for start, stop in prediction_gen]
        # preds = list(map(int, predictions[0, [1]]))
        results = [" ".join(inputs[idx].strip().split()[strt:stp]) for idx, (strt, stp) in enumerate(predictions)]

        if len(results) == 1:
            results = results.pop()

        return results, inputs

    def load_estimator(self):
        self.classifier = tf.estimator.Estimator(model_fn=self.model_fn,
                                                 params=self.graph_params,
                                                 config=self.estimator_config,
                                                 model_dir=self.model_dir)
