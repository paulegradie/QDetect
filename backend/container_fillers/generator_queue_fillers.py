import io
import os
import signal
import warnings
from collections import OrderedDict
from functools import partial
from multiprocessing import Process, Queue
from random import shuffle
from time import sleep
from backend.container_fillers.sequence_builder import _generate_sequence_
import numpy as np
from tqdm import tqdm
from backend.utils import _prepare_encoder_input_, _prepare_decoder_input_, _prepare_target_sequences_

from backend.special_tokens import SPECIAL_TOKENS

class GeneratorInputV3(object):
    """
    NEW MODEL GENERATOR

    This input function will ALSO return a start position and an end position to identify where the question starts and ends.

    ** V3 results in approximately 70% faster data loading than v1 **
    A sample generator with a side process to fill a que of samples. tensorflow can grab items from the queue generator at will.

    Usage:

    gen_obj = GneratorInputV2(**kwargs)  # This should take max a couple minutes to prefill the queue
    generator = gen_obj.from_queue_generator()

    """
    def __init__(self,
                 Q_size,
                 num_proc,
                 questions,
                 non_questions,
                 word2index,
                 max_seq_length,
                 max_num_questions,
                 max_num_elements,
                 randomize_num_questions=False):

        print('Initializing generator')
        self.global_sleep_clock = 1

        self.queues_length = 0
        self.Q_size = Q_size
        self.data_que = Queue(Q_size)

        self.processes = [Process(target=self.queue_builder,
                                  args=(questions,
                                        non_questions,
                                        word2index,
                                        max_seq_length,
                                        max_num_questions,
                                        max_num_elements,
                                        randomize_num_questions)) for _ in range(num_proc)]
        self.initialized = False

        # cleanup from previous iteration
        prev_pids = self.read_prev_pids()
        self.kill_prev_pids(prev_pids)

        # start the engines
        self.start_processes()
        self.save_pids()

        print('Process started...')
        print('Queues initializing...')
        while not self.data_que.full():
            size = self.queue_size()
            sleep(self.global_sleep_clock)
        self.initialized = True

    def save_pids(self):
        pids = [p.pid for p in self.processes]
        with open('temp_pid', 'w+') as pout:
            for pid in pids:
                pout.write(str(pid) + '\n')

    def read_prev_pids(self):
        try:
            with open('temp_pid', 'r') as pin:
                pid_list = [int(x.strip()) for x in pin.readlines()]
            os.remove('temp_pid')
        except IOError:
            pid_list = None
        return pid_list

    def kill_prev_pids(self, pid_list):
        if pid_list:
            try:
                for pid in pid_list:
                    os.kill(pid, signal.SIGTERM)
            except Exception as e:
                pass
        else:
            pass

    def terminate_processes(self):
        for proc in self.processes:
            proc.terminate()

    def start_processes(self):
        for proc in self.processes:
            proc.start()

    def queue_size(self):
        return self.data_que.qsize()

    def kill_builder(self):
        self.processes.terminate()



    def queue_builder(self,
                      questions,
                      non_questions,
                      word2index,
                      max_seq_length,
                      max_num_questions,
                      max_num_elements,
                      randomize_num_questions):
        " This function gets its own process which will run on the side "

        # to ensure no two questions togther reach the max_seq_length
        filtered_questions = list(filter(lambda x: len(x.split()) <= max_seq_length // max_num_elements, questions))
        shuffle(filtered_questions)

        assert max_num_questions <= max_num_elements - 1, ' need to have at least 1 fewer questions than num element '  # to ensure that there will always be non-questions
        while True:
            if not self.data_que.full():
                # if self.initialized:
                    # print("Queue not full. Repleneshing...")
                if randomize_num_questions:  # if max elements is 3, then we can add 1 or 2 questions.
                        max_num_questions = np.random.randint(1, max_num_elements - 1)

                input_sequence, target_sequence, num_questions, starts_list, stops_list = _generate_sequence_(filtered_questions,
                                                                                                                    non_questions,
                                                                                                                    max_seq_length=max_seq_length,
                                                                                                                    num_questions=max_num_questions,
                                                                                                                    max_num_elements=max_num_elements)

                # if _is_acceptable_(input_sequence, max_seq_length):
                encoder_inputs, encoder_input_lengths = _prepare_encoder_input_(input_sequence, max_seq_length, word2index)
                decoder_inputs, decoder_input_lengths = _prepare_decoder_input_(target_sequence, max_seq_length, word2index)
                target_sequences, target_seq_lengths = _prepare_target_sequences_(target_sequence, max_seq_length, word2index)

                array = partial(np.array, dtype=np.int32)
                features = array(encoder_inputs), array(encoder_input_lengths), array(decoder_inputs), array(decoder_input_lengths)
                labels = array(target_sequences), array(target_seq_lengths), array(num_questions), array(starts_list), array(stops_list)

                self.data_que.put((features, labels))

            else:
                # print('Queue full...Sleeping Again... 5 seconds')
                sleep(self.global_sleep_clock)

    def from_queue_generator(self):
        while True:
            if self.data_que.empty():
                sleep(self.global_sleep_clock)
            else:
                features, labels = self.data_que.get()
                yield features, labels

        self.data_que.close()
        self.data_que.join_thread()

        self.terminate_processes()


