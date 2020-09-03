from typing import List, Union, Tuple

from .stt_output.model import Word


class AlignedSentence:
    def __init__(self, sentence: str, start_time: Union[float, None], end_time: Union[float, None]):
        self.sentence = sentence
        self.start_time = start_time
        self.end_time = end_time


def data_to_data_sentence_only(
    data: List[Tuple[List[AlignedSentence], List[Word]]]
) -> List[Tuple[List[str], List[Word]]]:
    return [
        (
            [aligned_sentence.sentence for aligned_sentence in sentence_alignment],
            stt_output
        )
        for sentence_alignment, stt_output in data
    ]
