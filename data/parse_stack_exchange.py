import io
import re
import warnings
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

"""
This module is intended to be used to extract cleaned information from the
stack overflow datasets found at https://archive.org/details/stackexchange

The data is downloaded from the 7z drop down menu to the right.
e.g. https://archive.org/download/stackexchange/philosophy.stackexchange.com.7z

Do not use the meta files with this module.

Example usage:

``Python``
xmls = {'Badges.xml': None,
        'Comments.xml': 'Text',
        'PostHistory.xml': None,
        'PostLinks.xml': None,
        'Posts.xml': Body,
        'Tags.xml': None,
        'Users.xml': None,
        'Votes.xml': None}
# Sorry, don't know what each of these should be yet

datatype = 'Comments.xml'
dataname = xmls[datatype]

path = path/to_unzipped/datatype
docs = parse_xml_doc(path, xmls[datatype])
no_html = [[remove_html(x) for x in docs]
preprocessed_texts = [strip_and_lower(x) for x in no_html]

questions = process_text(preprocessed_texts, get_questions=True)
non_questions = process_text(preprocessed_texts, get_questions=False)

shuffle(questions)
shuffle(non_questions)

# equalize list lengths
non_questions = non_questions[:len(all_questions)]
"""


def parse_xml_doc(file_path, attribute_name='Text'):
    " return a list of all the text sentences from xml "
    tree = ET.parse(file_path)
    root = tree.getroot()
    texts = list()
    count = 0
    for child in root:
        try:
            texts.append(child.attrib[attribute_name])
            count += 1
        except Exception as e:
            print(e)
            pass
    return texts, count


def _regex_(string):
    return re.sub(r'  +', ' ', re.sub(r'[!@#¬$%"…”“^&*+\(\)\"\'_\\\-=,./<>?\[\]\{\}|\*`~;:]+', '', string)).strip()


def remove_html(text):
    text = BeautifulSoup(text).get_text()
    return text


def strip_and_lower(text):
    return text.strip().lower()


def _process_question_(text):
    text = text.replace('?', '<QUESTION><BREAK>')
    text = text.replace('.', '<BREAK>')
    text = text.split('<BREAK>')
    text = list(filter(lambda x: '<QUESTION>' in x, text))
    text = ' '.join(list(filter(lambda x: len(x) > 1, text)))
    text = text.replace('<QUESTION>', '?').strip()
    return text


def process_text(text_list, get_questions=True):
    texts = list()
    if get_questions:
        for text in text_list:
            if '?' in text:
                text = _process_question_(text)
                text = _regex_(text)
                texts.append(text)
    else:
        question_markers = ['who', 'what', 'where', 'why', 'when', 'how', '?', '??', '???']
        for text in text_list:
            if any(marker in text for marker in question_markers):
                continue
            else:
                text = _regex_(text)
                texts.append(text)

    return list(filter(lambda x: len(x.split()) > 2, texts))


def write_data(questions, non_questions, question_name='QUESTIONS.txt', answer_name='NON_QUESTIONS.txt'):
    try:
        with io.open(question_name, 'w+', encoding='utf-8') as out:
            for line in questions:
                if not line.strip():
                    continue
                out.write(''.join([line.strip(), '\n']))

        with io.open(answer_name, 'w+', encoding='utf-8') as out:
            for line in non_questions:
                if not line.strip():
                    continue
                out.write(''.join([line.strip(), '\n']))
    except IOError:
        with io.open(question_name, 'w+', encoding='utf-8') as out:
            for line in questions:
                if not line.strip():
                    continue
                out.write(''.join([str(line.strip()), '\n']))

        with io.open(answer_name, 'w+', encoding='utf-8') as out:
            for line in non_questions:
                if not line.strip():
                    continue
                out.write(''.join([str(line.strip()), '\n']))
