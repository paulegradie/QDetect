from argparse import ArgumentParser


def gather_args():

    parser = ArgumentParser(description='default sets to a tiny toy dataset - normal vocab size, 5 samples')
    subparsers = parser.add_subparsers(dest='command', help='options:\n--train\n\n--train_and_eval\n\n--train_until_thresh\n\n use like [script.py train -h]')

    def repeats(parser):

        exclusive_group = parser.add_mutually_exclusive_group()
        exclusive_group.add_argument("-g", '--gen-pickle', type=str, default=None, help="get dataset from generator instead of static dataset")
        exclusive_group.add_argument('-d', '--static-pickle', type=str, default=None, help='get dataset from static file')

        parser.add_argument('-f', '--model-dir', type=str, default='test_directory')
        parser.add_argument('-m', '--minimode', action='store_true', default=False, help='Overrides batch size to be size of toy dataset')
        parser.add_argument('-b', '--batch-size', type=int, default=16)
        parser.add_argument('-o', '--overwrite', action='store_true')
        return parser

    def train_args(parser):
        parser.add_argument('-s', '--steps', type=int, default=1000, help='number of batches to train over')
        return parser

    def eval_args(parser):
        parser.add_argument('-w', '--show-sample', type=int, default=5, help='number of eval samples to show')
        return parser

    train_args(repeats(subparsers.add_parser("train")))

    subparser_train_and_eval = eval_args(train_args(repeats(subparsers.add_parser("train_and_eval"))))
    subparser_train_and_eval.add_argument('-e', '--eval_every', type=float, help='evaluate until mrean-BLEU threshold is met')

    subparser_train_until_threshold = eval_args(train_args(repeats(subparsers.add_parser("train_until_thresh"))))
    subparser_train_until_threshold.add_argument('-t', '--threshold', type=float, help='evaluate until mrean-BLEU threshold is met')

    eval_args(repeats(subparsers.add_parser('eval')))
    train_args(repeats(subparsers.add_parser('debug')))

    args = parser.parse_args()
    return args
