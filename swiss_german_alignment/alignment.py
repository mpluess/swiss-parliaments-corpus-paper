# Assumption in downstream processing: returned alignments are global, i.e. include full transcript and STT strings.

from typing import Set

from Bio import Align


ALPHABET_ASCII = {chr(i) for i in range(128)}
ALPHABET_LATIN_1 = {chr(i) for i in range(256)}


def create_aligner_global(
    alphabet: Set[str],
    match_score: float,
    mismatch_score: float,
    target_left_open_gap_score: float, target_internal_open_gap_score: float, target_right_open_gap_score: float,
    target_left_extend_gap_score: float, target_internal_extend_gap_score: float, target_right_extend_gap_score: float,
    query_left_open_gap_score: float, query_internal_open_gap_score: float, query_right_open_gap_score: float,
    query_left_extend_gap_score: float, query_internal_extend_gap_score: float, query_right_extend_gap_score: float
) -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner()
    aligner.alphabet = sorted(alphabet)
    aligner.mode = 'global'

    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score
    aligner.target_left_open_gap_score = target_left_open_gap_score
    aligner.target_internal_open_gap_score = target_internal_open_gap_score
    aligner.target_right_open_gap_score = target_right_open_gap_score
    aligner.target_left_extend_gap_score = target_left_extend_gap_score
    aligner.target_internal_extend_gap_score = target_internal_extend_gap_score
    aligner.target_right_extend_gap_score = target_right_extend_gap_score
    aligner.query_left_open_gap_score = query_left_open_gap_score
    aligner.query_internal_open_gap_score = query_internal_open_gap_score
    aligner.query_right_open_gap_score = query_right_open_gap_score
    aligner.query_left_extend_gap_score = query_left_extend_gap_score
    aligner.query_internal_extend_gap_score = query_internal_extend_gap_score
    aligner.query_right_extend_gap_score = query_right_extend_gap_score

    return aligner
