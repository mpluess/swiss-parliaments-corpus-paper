from collections import defaultdict, namedtuple
import math
import statistics
from typing import List, Union, Tuple

from Bio import Align
import pandas as pd

from .alignment import ALPHABET_LATIN_1, create_aligner_global
from .iou_estimator import IouEstimator
from .metrics import calculate_metrics
from .model import AlignedSentence, data_to_data_sentence_only
from .stt_output.model import Word
from .text_processing import preprocess_transcript_for_alignment


class SentenceAligner:
    def __init__(
        self,
        create_aligner_function=create_aligner_global, alphabet=ALPHABET_LATIN_1,
        match_score=0.03875752471676385, mismatch_score=-1.0,
        truth_left_open_gap_score=-0.5038367052042227, truth_internal_open_gap_score=-1.0, truth_right_open_gap_score=-0.43980186690399603,
        truth_left_extend_gap_score=-0.2440180768676541, truth_internal_extend_gap_score=-0.4817146150129493, truth_right_extend_gap_score=-0.2594102766979399,
        stt_left_open_gap_score=-1.0, stt_internal_open_gap_score=-0.7698209478188247, stt_right_open_gap_score=-0.9815365376036425,
        stt_left_extend_gap_score=-0.25266456311369834, stt_internal_extend_gap_score=-0.7698209478188247, stt_right_extend_gap_score=-0.5619337177636895,

        do_length_ratio_full_transcript_filtering_while_fitting=True,
        length_ratio_full_transcript_min=0.167,
        length_ratio_full_transcript_max=6.0,
        fit_time_correction=True,
        fit_iou_estimator=False,
        iou_threshold=0.7,
        iou_estimator_optimize_hyperparams=True
    ):
        self.create_aligner_function = create_aligner_function
        self.alphabet = alphabet
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.truth_left_open_gap_score = truth_left_open_gap_score
        self.truth_internal_open_gap_score = truth_internal_open_gap_score
        self.truth_right_open_gap_score = truth_right_open_gap_score
        self.truth_left_extend_gap_score = truth_left_extend_gap_score
        self.truth_internal_extend_gap_score = truth_internal_extend_gap_score
        self.truth_right_extend_gap_score = truth_right_extend_gap_score
        self.stt_left_open_gap_score = stt_left_open_gap_score
        self.stt_internal_open_gap_score = stt_internal_open_gap_score
        self.stt_right_open_gap_score = stt_right_open_gap_score
        self.stt_left_extend_gap_score = stt_left_extend_gap_score
        self.stt_internal_extend_gap_score = stt_internal_extend_gap_score
        self.stt_right_extend_gap_score = stt_right_extend_gap_score
        self.do_length_ratio_full_transcript_filtering_while_fitting = do_length_ratio_full_transcript_filtering_while_fitting
        self.length_ratio_full_transcript_min = length_ratio_full_transcript_min
        self.length_ratio_full_transcript_max = length_ratio_full_transcript_max
        self.fit_time_correction = fit_time_correction
        self.fit_iou_estimator = fit_iou_estimator
        self.iou_threshold = iou_threshold
        self.iou_estimator_optimize_hyperparams = iou_estimator_optimize_hyperparams

        self.aligner = self.create_aligner_function(
            self.alphabet,
            match_score=self.match_score,
            mismatch_score=self.mismatch_score,
            target_left_open_gap_score=self.truth_left_open_gap_score,
            target_internal_open_gap_score=self.truth_internal_open_gap_score,
            target_right_open_gap_score=self.truth_right_open_gap_score,
            target_left_extend_gap_score=self.truth_left_extend_gap_score,
            target_internal_extend_gap_score=self.truth_internal_extend_gap_score,
            target_right_extend_gap_score=self.truth_right_extend_gap_score,
            query_left_open_gap_score=self.stt_left_open_gap_score,
            query_internal_open_gap_score=self.stt_internal_open_gap_score,
            query_right_open_gap_score=self.stt_right_open_gap_score,
            query_left_extend_gap_score=self.stt_left_extend_gap_score,
            query_internal_extend_gap_score=self.stt_internal_extend_gap_score,
            query_right_extend_gap_score=self.stt_right_extend_gap_score
        )

        self.start_time_correction_ = None
        self.end_time_correction_ = None
        self.iou_estimator_ = None

    def fit(self, data: List[Tuple[List[AlignedSentence], List[Word]]]):
        if self.fit_time_correction:
            data_pred, _, _ = self.predict(
                data_to_data_sentence_only(data),
                self.do_length_ratio_full_transcript_filtering_while_fitting, False, False
            )
            start_time_diffs = []
            end_time_diffs = []
            for (sentence_alignment_true, _), sentence_alignment_pred in zip(data, data_pred):
                for aligned_sentence_true, aligned_sentence_pred in zip(sentence_alignment_true, sentence_alignment_pred):
                    if aligned_sentence_true.start_time is not None and aligned_sentence_true.end_time is not None and aligned_sentence_pred.start_time is not None and aligned_sentence_pred.end_time is not None:
                        start_time_diffs.append(aligned_sentence_true.start_time - aligned_sentence_pred.start_time)
                        end_time_diffs.append(aligned_sentence_true.end_time - aligned_sentence_pred.end_time)
            # TODO optimize on dev set instead of median?
            self.start_time_correction_ = statistics.median(start_time_diffs)
            self.end_time_correction_ = statistics.median(end_time_diffs)
            print('Time offsets fitted')

        if self.fit_iou_estimator:
            data_pred, dfs_alignment_info, _ = self.predict(
                data_to_data_sentence_only(data),
                self.do_length_ratio_full_transcript_filtering_while_fitting, self.fit_time_correction, False
            )
            dfs_alignment_info_filtered = []
            for (sentence_alignment_true, _), sentence_alignment_pred, df_alignment_info in zip(data, data_pred, dfs_alignment_info):
                if df_alignment_info is not None:
                    _, metrics_raw = calculate_metrics(sentence_alignment_true, sentence_alignment_pred)
                    assert len(df_alignment_info) == len(metrics_raw['ious'])
                    df_alignment_info['iou'] = metrics_raw['ious']
                    dfs_alignment_info_filtered.append(df_alignment_info)
            df_alignment_info = pd.concat(dfs_alignment_info_filtered, ignore_index=True)
            self.iou_estimator_ = IouEstimator(self.iou_estimator_optimize_hyperparams).fit(df_alignment_info)
            print('IOU estimator fitted')

        return self

    def predict(
        self,
        data_sentence_only: List[Tuple[List[str], List[Word]]],
        do_length_ratio_full_transcript_filtering=True, do_time_correction=True, do_iou_estimate_filtering=False
    ) -> Tuple[List[List[AlignedSentence]], List[Union[pd.DataFrame, None]], List[pd.Series]]:
        data_pred = []
        dfs_alignment_info = []
        iou_estimate_series_list = []
        for i, (truth_sentences, stt_output) in enumerate(data_sentence_only):
            sentence_alignment, df_alignment_info, iou_estimate_series = self.predict_one(
                truth_sentences, stt_output,
                do_length_ratio_full_transcript_filtering, do_time_correction, do_iou_estimate_filtering
            )
            data_pred.append(sentence_alignment)
            dfs_alignment_info.append(df_alignment_info)
            iou_estimate_series_list.append(iou_estimate_series)

        return data_pred, dfs_alignment_info, iou_estimate_series_list

    def predict_one(
        self,
        truth_sentences: List[str], stt_output: List[Word],
        do_length_ratio_full_transcript_filtering, do_time_correction, do_iou_estimate_filtering,
        time_range_start=0.0, time_range_end=math.inf
    ) -> Tuple[List[AlignedSentence], Union[pd.DataFrame, None], pd.Series]:
        assert not do_length_ratio_full_transcript_filtering or self.do_length_ratio_full_transcript_filtering_while_fitting
        assert not do_time_correction or self.fit_time_correction
        assert not do_iou_estimate_filtering or self.fit_iou_estimator

        truth_sentences_preprocessed = [preprocess_transcript_for_alignment(sentence, self.alphabet) for sentence in truth_sentences]
        truth_transcript_preprocessed = ' '.join(truth_sentences_preprocessed)

        stt_transcript_preprocessed = ''
        stt_index = 0
        stt_index_to_word_info_mapping = dict()
        for word in stt_output:
            if word.start_time >= time_range_start and word.end_time <= time_range_end:
                word_preprocessed = preprocess_transcript_for_alignment(word.word, self.alphabet)
                appendix = word_preprocessed + ' '
                stt_transcript_preprocessed += appendix
                new_stt_offset = stt_index + len(appendix)
                for i in range(stt_index, new_stt_offset):
                    stt_index_to_word_info_mapping[i] = {
                        'word': Word(word_preprocessed, word.start_time, word.end_time, word.confidence),
                        'start_index': stt_index,
                        'end_index': new_stt_offset - 1,
                    }
                stt_index = new_stt_offset

        char_alignment = self._create_char_alignment(
            truth_transcript_preprocessed, stt_transcript_preprocessed,
            do_length_ratio_full_transcript_filtering
        )
        if char_alignment is not None:
            sentence_alignment, df_alignment_info = self._transform_char_to_sentence_alignment(
                char_alignment, truth_transcript_preprocessed, stt_transcript_preprocessed,
                truth_sentences, truth_sentences_preprocessed,
                stt_index_to_word_info_mapping,
                do_time_correction
            )
        else:
            sentence_alignment, df_alignment_info = None, None

        iou_estimate_series = None
        if sentence_alignment is None:
            sentence_alignment = [AlignedSentence(truth_sentence, None, None) for truth_sentence in truth_sentences]
        else:
            assert df_alignment_info is not None
            if self.iou_estimator_ is not None:
                iou_estimate_series = self.iou_estimator_.predict(df_alignment_info)

            if do_iou_estimate_filtering:
                sentence_alignment = self.filter_by_iou_estimate(
                    sentence_alignment, df_alignment_info,
                    iou_estimate_series=iou_estimate_series,
                )

        if iou_estimate_series is None:
            iou_estimate_series = pd.Series([None] * len(sentence_alignment))
        assert len(sentence_alignment) == len(iou_estimate_series)
        return sentence_alignment, df_alignment_info, iou_estimate_series

    def filter_by_iou_estimate(self, sentence_alignment, df_alignment_info, iou_estimate_series=None, iou_threshold=None):
        assert self.iou_estimator_ is not None
        assert len(sentence_alignment) == len(df_alignment_info)

        if iou_estimate_series is None:
            iou_estimate_series = self.iou_estimator_.predict(df_alignment_info)
        sentence_alignment_filtered = []
        for iou_estimate_is_na, iou_estimate, aligned_sentence in zip(
            iou_estimate_series.isna(),
            iou_estimate_series,
            sentence_alignment
        ):
            if iou_estimate_is_na or iou_estimate <= (iou_threshold if iou_threshold is not None else self.iou_threshold):
                sentence_alignment_filtered.append(AlignedSentence(aligned_sentence.sentence, None, None))
            else:
                sentence_alignment_filtered.append(aligned_sentence)

        return sentence_alignment_filtered

    def _create_char_alignment(
        self, truth_transcript_preprocessed: str, stt_transcript_preprocessed: str, do_length_ratio_full_transcript_filtering: bool
    ) -> Union[Align.PairwiseAlignment, None]:
        if len(truth_transcript_preprocessed) == 0 or len(stt_transcript_preprocessed) == 0:
            return None

        length_ratio = len(truth_transcript_preprocessed) / len(stt_transcript_preprocessed)
        if do_length_ratio_full_transcript_filtering and (length_ratio < self.length_ratio_full_transcript_min or length_ratio > self.length_ratio_full_transcript_max):
            return None
        else:
            truth_transcript_list = list(truth_transcript_preprocessed)
            stt_transcript_list = list(stt_transcript_preprocessed)
            assert len(truth_transcript_preprocessed) == len(truth_transcript_list)
            assert len(stt_transcript_preprocessed) == len(stt_transcript_list)
            alignments = self.aligner.align(truth_transcript_list, stt_transcript_list)
            # Try-except rather than len() > 0 because len() can result in an OverflowError
            try:
                alignment = alignments[0]
            except IndexError:
                alignment = None

            return alignment

    def _transform_char_to_sentence_alignment(
            self,
            char_alignment: Align.PairwiseAlignment, truth_transcript_preprocessed: str, stt_transcript_preprocessed: str,
            truth_sentences_orig, truth_sentences_preprocessed,
            stt_index_to_word_info_mapping,
            do_time_correction
    ) -> Tuple[Union[List[AlignedSentence], None], Union[pd.DataFrame, None]]:

        def interpolate(truth_start, truth_end, stt_start, stt_end):
            truth_diff = truth_end - truth_start
            stt_diff = stt_end - stt_start
            truth_index_stt_index_tuples = []
            for i, truth_index in enumerate(range(truth_start, truth_end)):
                stt_index = round(stt_start + i*stt_diff/truth_diff)
                assert stt_index <= stt_end
                truth_index_stt_index_tuples.append((truth_index, stt_index))

            return truth_index_stt_index_tuples

        truth_indices, stt_indices = char_alignment.aligned
        assert len(truth_indices) == len(stt_indices)
        if len(truth_indices) == 0:
            return None, None
        truth_index_to_stt_index = dict()
        last_truth_end = 0
        last_stt_end = stt_indices[0][0]
        truth_spans_aligned_to_gap = []
        for (truth_start, truth_end), (stt_start, stt_end) in zip(truth_indices, stt_indices):
            assert truth_end > truth_start
            assert truth_end - truth_start == stt_end - stt_start
            assert last_truth_end <= truth_start
            assert last_stt_end <= stt_start
            for truth_index, stt_index in zip(range(truth_start, truth_end), range(stt_start, stt_end)):
                assert truth_index not in truth_index_to_stt_index
                truth_index_to_stt_index[truth_index] = stt_index
            if last_truth_end < truth_start:
                truth_spans_aligned_to_gap.append((last_truth_end, truth_start, last_stt_end, stt_start))
            last_truth_end = truth_end
            last_stt_end = stt_end
        assert truth_indices[-1][1] <= len(truth_transcript_preprocessed)
        assert stt_indices[-1][1] <= len(stt_transcript_preprocessed)
        if truth_indices[-1][1] < len(truth_transcript_preprocessed):
            truth_spans_aligned_to_gap.append((
                truth_indices[-1][1], len(truth_transcript_preprocessed),
                stt_indices[-1][1], stt_indices[-1][1]
            ))
        for truth_start, truth_end, stt_start, stt_end in truth_spans_aligned_to_gap:
            for truth_index, stt_index in interpolate(truth_start, truth_end, stt_start, stt_end):
                assert truth_index not in truth_index_to_stt_index
                truth_index_to_stt_index[truth_index] = min(stt_index, len(stt_transcript_preprocessed) - 1)
        assert len(truth_index_to_stt_index) == len(truth_transcript_preprocessed)

        WordInfoTuple = namedtuple('WordInfo', ['start_time', 'end_time', 'word', 'stt_confidence'])

        def word_info_to_tuple(word_info):
            return WordInfoTuple(
                start_time=word_info['word'].start_time,
                end_time=word_info['word'].end_time,
                word=word_info['word'].word,
                stt_confidence=word_info['word'].confidence
            )

        last_end_time = 0.0
        last_start_index = 0
        sentence_alignment = []
        alignment_info = defaultdict(list)
        assert len(truth_sentences_orig) == len(truth_sentences_preprocessed)
        if do_time_correction:
            assert self.start_time_correction_ is not None and self.end_time_correction_ is not None
        for truth_sentence_orig, truth_sentence_preprocessed in zip(truth_sentences_orig, truth_sentences_preprocessed):
            if len(truth_sentence_preprocessed) > 0:
                start_index_truth = truth_transcript_preprocessed.find(truth_sentence_preprocessed, last_start_index)
                if start_index_truth == -1:
                    print(f'WARNING: could not find sentence {truth_sentence_preprocessed} from starting position {last_start_index}')
                    return None, None
                end_index_truth = start_index_truth + len(truth_sentence_preprocessed) - 1

                word_info_start = stt_index_to_word_info_mapping[truth_index_to_stt_index[start_index_truth]]
                word_info_end = stt_index_to_word_info_mapping[truth_index_to_stt_index[end_index_truth]]

                start_time = word_info_start['word'].start_time
                if do_time_correction:
                    start_time += self.start_time_correction_
                start_time = max(start_time, last_end_time)
                end_time = word_info_end['word'].end_time
                if do_time_correction:
                    end_time += self.end_time_correction_
                end_time = max(start_time, end_time)

                stt_sentence_preprocessed = stt_transcript_preprocessed[word_info_start['start_index']:word_info_end['end_index']]

                stt_word_info_tuples = {
                    word_info_to_tuple(stt_index_to_word_info_mapping[i])
                    for i in range(
                        truth_index_to_stt_index[start_index_truth],
                        truth_index_to_stt_index[end_index_truth] + 1
                    )
                }

                alignment_info['truth_length'].append(len(truth_sentence_preprocessed))
                alignment_info['stt_length'].append(len(stt_sentence_preprocessed))
                alignment_info['score'].append(self.aligner.score(truth_sentence_preprocessed, stt_sentence_preprocessed))
                alignment_info['stt_confidence'].append(
                    statistics.mean([word_info_tuple.stt_confidence for word_info_tuple in stt_word_info_tuples])
                    if len(stt_word_info_tuples) > 0
                    else None
                )
                alignment_info['truth_string'].append(truth_sentence_preprocessed)
                alignment_info['stt_string'].append(stt_sentence_preprocessed)
                alignment_info['audio_duration'].append(end_time - start_time)

                sentence_alignment.append(AlignedSentence(truth_sentence_orig, start_time, end_time))

                last_start_index = start_index_truth
                last_end_time = end_time
            else:
                alignment_info['truth_length'].append(0)
                alignment_info['stt_length'].append(0)
                alignment_info['score'].append(None)
                alignment_info['stt_confidence'].append(0.0)
                alignment_info['truth_string'].append('')
                alignment_info['stt_string'].append('')
                alignment_info['audio_duration'].append(0.0)

                sentence_alignment.append(AlignedSentence(truth_sentence_orig, None, None))

        df_alignment_info = pd.DataFrame(alignment_info)
        return sentence_alignment, df_alignment_info
