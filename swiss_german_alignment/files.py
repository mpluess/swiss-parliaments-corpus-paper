# https://manual.audacityteam.org/man/importing_and_exporting_labels.html

from typing import List

from .model import AlignedSentence

NONE_TIME_STRING_REPRESENTATION = '-1.0'


def read_sentence_alignment(path_to_alignment) -> List[AlignedSentence]:
    with open(path_to_alignment, encoding='utf-8') as f:
        alignment = []
        last_end_time = None
        for line in f:
            start_time, end_time, sentence = line.strip().split('\t', maxsplit=2)
            start_time = float(start_time)
            end_time = float(end_time)
            if start_time < 0:
                start_time = None
            if end_time < 0:
                end_time = None
            if start_time is not None and end_time is not None:
                assert start_time <= end_time, f'start_time={start_time}, end_time={end_time}'
                if last_end_time is not None:
                    if start_time < last_end_time:
                        print(f'WARNING: overlapping intervals, start_time={start_time} < last_end_time={last_end_time}')
            last_end_time = end_time

            alignment.append(AlignedSentence(sentence, start_time, end_time))

    return alignment


def write_sentence_alignment(sentence_alignment: List[AlignedSentence], path_to_sentence_alignment):
    def transform_start_time(time):
        if time is None:
            return NONE_TIME_STRING_REPRESENTATION
        else:
            return time

    def transform_end_time(time):
        if time is None:
            return NONE_TIME_STRING_REPRESENTATION
        else:
            return time

    with open(path_to_sentence_alignment, 'w', encoding='utf-8') as f:
        for aligned_sentence in sentence_alignment:
            f.write(f'{transform_start_time(aligned_sentence.start_time)}\t{transform_end_time(aligned_sentence.end_time)}\t{aligned_sentence.sentence}\n')
