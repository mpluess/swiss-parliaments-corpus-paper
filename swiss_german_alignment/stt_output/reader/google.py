import json
from typing import List

from .base import SttOutputReader
from ..model import Word


class GoogleSttOutputReader(SttOutputReader):
    def read(self, path_to_stt_output: str) -> List[Word]:
        def time_to_float(time_string) -> float:
            return float(time_string.replace('s', ''))

        with open(path_to_stt_output, encoding='utf-8') as f:
            stt_output_orig = json.load(f)

        stt_output = []
        for result in stt_output_orig['results']:
            for word_dict in result['alternatives'][0]['words']:
                word = word_dict['word']
                start_time = time_to_float(word_dict['startTime'])
                end_time = time_to_float(word_dict['endTime'])
                confidence = result['alternatives'][0]['confidence']
                stt_output.append(Word(word, start_time, end_time, confidence))

        return stt_output
