import json
from typing import List

from .base import SttOutputReader
from ..model import Word


class AmazonSttOutputReader(SttOutputReader):
    def read(self, path_to_stt_output: str) -> List[Word]:
        with open(path_to_stt_output, encoding='utf-8') as f:
            stt_output_orig = json.load(f)

        stt_output = []
        for word_dict in stt_output_orig['results']['items']:
            if word_dict['type'] != 'punctuation':
                word = word_dict['alternatives'][0]['content']
                start_time = float(word_dict['start_time'])
                end_time = float(word_dict['end_time'])
                confidence = float(word_dict['alternatives'][0]['confidence'])
                stt_output.append(Word(word, start_time, end_time, confidence))

        return stt_output
