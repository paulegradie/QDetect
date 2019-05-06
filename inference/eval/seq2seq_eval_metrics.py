
from backend.utils import convert_int2word
from nltk.translate.bleu_score import sentence_bleu


def evaluate_seq2seq_predictions(predictions, validation_features, validation_labels, index2word, show_first=3):
    scores = list()
    for idx, pred in enumerate(predictions):
        try:
            feat = convert_int2word(index2word, validation_features[0][idx])
            target = convert_int2word(index2word, validation_labels[0][idx])
            pred = convert_int2word(index2word, pred['sentence_tokens'])

            scores.append(sentence_bleu([target], pred))
            if idx < show_first:
                print('\n\n')
                print('SAMPLE: {}'.format(idx + 1))
                print("Orig: {}".format(' '.join(feat)))
                print("Target: {}".format(' '.join(target)))
                print("Pred: {}".format(' '.join(pred)))
        except Exception as e:
            break
    print("\nN={}, AVERAGE BLEU: {}\n\n".format(str(idx), np.mean(scores)))
    return scores
