import glob
import os
from typing import List, Tuple

import pandas as pd
from tinytag import TinyTag

from ..model import Metadata
from swiss_german_alignment.files import read_sentence_alignment
from swiss_german_alignment.model import AlignedSentence
from swiss_german_alignment.stt_output.model import Word
from swiss_german_alignment.stt_output.reader.base import SttOutputReader

MANUAL_ALIGNMENT_SUFFIX = '__manual_alignment.txt'


def read_data_all(
    path_to_corpus_final: str, stt_output_suffix: str, stt_output_reader: SttOutputReader
) -> List[Tuple[List[AlignedSentence], List[Word]]]:

    data = []
    for path_to_truth_alignment in glob.iglob(os.path.join(path_to_corpus_final, f'*{MANUAL_ALIGNMENT_SUFFIX}')):
        path_to_stt_output = path_to_truth_alignment.replace(MANUAL_ALIGNMENT_SUFFIX, stt_output_suffix)
        if os.path.isfile(path_to_stt_output):
            sentence_alignment = read_sentence_alignment(path_to_truth_alignment)
            stt_output = stt_output_reader.read(path_to_stt_output)
            data.append((sentence_alignment, stt_output))
        else:
            print(f'{path_to_truth_alignment}: not all necessary files found')

    return data


def read_data_split(
    path_to_corpus_final: str, path_to_corpus_raw: str, path_to_split: str, stt_output_suffix: str, stt_output_reader: SttOutputReader
) -> Tuple[
    Tuple[List[Tuple[List[AlignedSentence], List[Word]]], List[Metadata]],
    Tuple[List[Tuple[List[AlignedSentence], List[Word]]], List[Metadata]]
]:
    train_set = _read_split(os.path.join(path_to_split, 'train.csv'))
    test_set = _read_split(os.path.join(path_to_split, 'test.csv'))
    data_train = []
    data_test = []
    metadata_train = []
    metadata_test = []
    for path_to_truth_alignment in glob.iglob(os.path.join(path_to_corpus_final, f'*{MANUAL_ALIGNMENT_SUFFIX}')):
        path_to_stt_output = path_to_truth_alignment.replace(MANUAL_ALIGNMENT_SUFFIX, stt_output_suffix)
        if os.path.isfile(path_to_stt_output):
            corpus_part_name = os.path.basename(path_to_truth_alignment).replace(MANUAL_ALIGNMENT_SUFFIX, '')
            truth_sentence_alignment = read_sentence_alignment(path_to_truth_alignment)
            stt_output = stt_output_reader.read(path_to_stt_output)
            path_to_flac = os.path.join(path_to_corpus_raw, f'{corpus_part_name}.flac')
            assert os.path.isfile(path_to_flac)
            duration_seconds = TinyTag.get(path_to_flac).duration
            if corpus_part_name in train_set:
                data_train.append((truth_sentence_alignment, stt_output))
                metadata_train.append(Metadata(duration_seconds, corpus_part_name))
            elif corpus_part_name in test_set:
                data_test.append((truth_sentence_alignment, stt_output))
                metadata_test.append(Metadata(duration_seconds, corpus_part_name))
            else:
                print(f'{corpus_part_name}: neither included in train nor in test set')
        else:
            print(f'{path_to_truth_alignment}: not all necessary files found')

    return (data_train, metadata_train), (data_test, metadata_test)


def _read_split(path_to_split):
    df = pd.read_csv(path_to_split, encoding='utf-8')
    return set(df['name'])
