import numpy as np
from random import shuffle
from backend.utils import _make_breaks

def _generate_sequence_(questions, non_questions, max_seq_length, num_questions, max_num_elements):
    """
    - Aim to generate seq shorter than max_seq_length by filtering as data is processed
    """
    assert max_num_elements >= num_questions, 'must have more elements than questions'

    num_non_questions = max_num_elements - num_questions

    # keep track of questions for label making
    pre_choices = np.random.choice(questions, num_questions, replace=False).tolist()
    question_choices = [''.join([x, '<QQQ>']) for x in pre_choices]
    length_of_questions = sum([len(x.split()) for x in question_choices])

    filtered_non_questions = list(filter(lambda x: len(x.split()) <= int((max_seq_length - length_of_questions) / num_non_questions), non_questions))
    shuffle(filtered_non_questions)

    non_question_choices = np.random.choice(filtered_non_questions, num_non_questions, replace=False).tolist()

    # New code
    input_list = [(len(x.split()), x) for x in question_choices] + [(len(x.split()), x) for x in non_question_choices]

    shuffle(input_list)

    # New code
    starts, stops = _make_breaks(input_list)
    input_list = [x[1] for x in input_list]

    input_sequence = ' '.join(input_list).replace('<QQQ>', '')
    target_sequence = ' '.join(
        [x.replace('<QQQ>', '') for x in input_list if '<QQQ>' in x])
    return input_sequence, target_sequence, num_questions, starts, stops
