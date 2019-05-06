def show_result(idx, feat, target, pred):
    print('\n\n')
    print('SAMPLE: {}'.format(idx + 1))
    print("Orig: {}".format(' '.join(feat[idx])))
    print("Target: {}".format(' '.join(target)))
    print("Pred: {}".format(' '.join(pred)))

