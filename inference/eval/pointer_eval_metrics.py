
from backend.utils import convert_int2word
from nltk.translate.bleu_score import sentence_bleu
from inference.inference_utils import show_result
import numpy as np


def evaluate_single_pointer_predictions(predictions, validation_features, validation_labels, index2word, show_first=3):

    unrolled_predictions = [int(pred['question_starts'][0]) for pred in predictions]

    # list of word lists
    feat = [convert_int2word(index2word, x) for x in validation_features[0]]

    targets = [single_feature[start[0]:] for single_feature, start in zip(feat, validation_labels[-2])]
    preds = [single_feature[start:] for single_feature, start in zip(feat, unrolled_predictions)]

    scores = list()
    for idx, (pred, target) in enumerate(zip(preds, targets)):

        scores.append(sentence_bleu([target], pred))

        if idx < show_first:
            show_result(idx, feat, target, pred)


    print("\nN={}, AVERAGE BLEU: {}\n\n".format(str(idx), np.mean(scores)))
    return scores


def evaluate_double_pointer_predictions(predictions, validation_features, validation_labels, index2word, show_first=3):

    raw_preds_dict = [p for p in predictions]
    start_predictions = [int(pred['question_starts']) for pred in raw_preds_dict]
    end_predictions = [int(pred['question_ends']) for pred in raw_preds_dict]

    # list of word lists
    feat = [convert_int2word(index2word, x) for x in validation_features[0]]

    targets = [single_feature[start[0]:end[0]] for single_feature, start, end in zip(feat, validation_labels[-2], validation_labels[-1])]
    preds = [single_feature[start:end] for single_feature, start, end in zip(feat, start_predictions, end_predictions)]

    scores = list()
    for idx, (pred, target) in enumerate(zip(preds, targets)):

        scores.append(sentence_bleu([target], pred))

        if idx < show_first:
            show_result(idx, feat, target, pred)

    print("\nN={}, AVERAGE BLEU: {}\n\n".format(str(idx), np.mean(scores)))

    return scores
