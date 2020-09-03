import statistics
from typing import Any, List, Tuple

from .model import AlignedSentence


def calculate_all_metrics(
    data_true: List[Tuple[List[AlignedSentence], Any]], data_pred: List[List[AlignedSentence]], duration_audio_seconds: float
):
    aligned_sentences_true = []
    aligned_sentences_pred = []
    for (sentence_alignment_true, _), sentence_alignment_pred in zip(data_true, data_pred):
        for aligned_sentence_true, aligned_sentence_pred in zip(sentence_alignment_true, sentence_alignment_pred):
            aligned_sentences_true.append(aligned_sentence_true)
            aligned_sentences_pred.append(aligned_sentence_pred)

    metrics, metrics_raw = calculate_metrics(
        aligned_sentences_true, aligned_sentences_pred,
        duration_audio_seconds=duration_audio_seconds,
    )
    assert metrics is not None and metrics_raw is not None
    return metrics, metrics_raw


def calculate_metrics(
    truth_sentence_alignment: List[AlignedSentence], predicted_sentence_alignment: List[AlignedSentence],
    duration_audio_seconds: float = None,
):
    def calculate_ious(sentence_alignment_1: List[AlignedSentence], sentence_alignment_2: List[AlignedSentence]):
        ious = []
        for aligned_sentence_1, aligned_sentence_2 in zip(sentence_alignment_1, sentence_alignment_2):
            start_time_1 = aligned_sentence_1.start_time
            start_time_2 = aligned_sentence_2.start_time
            end_time_1 = aligned_sentence_1.end_time
            end_time_2 = aligned_sentence_2.end_time
            if start_time_1 is None or start_time_2 is None:
                ious.append(None)
            else:
                intersection = min(end_time_1, end_time_2) - max(start_time_1, start_time_2)
                intersection = 0.0 if intersection < 0.0 else intersection
                union = max(end_time_1, end_time_2) - min(start_time_1, start_time_2)
                assert union > 0.0
                iou = intersection / union
                ious.append(iou)

        return ious

    def calculate_precision_recall(truth_sentence_alignment: List[AlignedSentence], predicted_sentence_alignment: List[AlignedSentence]):
        TP = TN = FP = FN = 0
        for aligned_sentence_truth, aligned_sentence_pred in zip(truth_sentence_alignment, predicted_sentence_alignment):
            start_time_truth = aligned_sentence_truth.start_time
            start_time_pred = aligned_sentence_pred.start_time
            if start_time_truth is not None and start_time_pred is not None:
                TP += 1
            elif start_time_truth is None and start_time_pred is None:
                TN += 1
            elif start_time_truth is None and start_time_pred is not None:
                FP += 1
            elif start_time_truth is not None and start_time_pred is None:
                FN += 1

        precision, recall = _precision_recall_from_counts(TP, FP, FN)
        return precision, recall, (TP, TN, FP, FN)

    if len(truth_sentence_alignment) != len(predicted_sentence_alignment):
        print(f'WARNING: length mismatch: {len(truth_sentence_alignment)} != {len(predicted_sentence_alignment)}')
        return None, None
    for aligned_sentence_1, aligned_sentence_2 in zip(truth_sentence_alignment, predicted_sentence_alignment):
        if aligned_sentence_1.sentence != aligned_sentence_2.sentence:
            print(f'WARNING: sentence mismatch: {aligned_sentence_1.sentence} != {aligned_sentence_2.sentence}')

    ious = calculate_ious(truth_sentence_alignment, predicted_sentence_alignment)
    precision, recall, (TP, TN, FP, FN) = calculate_precision_recall(truth_sentence_alignment, predicted_sentence_alignment)

    duration_truth_seconds = sum(
        (s.end_time - s.start_time if s.end_time is not None and s.start_time is not None else 0.0)
        for s in truth_sentence_alignment
    )
    duration_predicted_seconds = sum(
        (s.end_time - s.start_time if s.end_time is not None and s.start_time is not None else 0.0)
        for s in predicted_sentence_alignment
    )

    metrics = {
        'iou mean': _aggregate_ious(statistics.mean, ious),
        'iou median': _aggregate_ious(statistics.median, ious),
        'sentence precision': precision,
        'sentence recall': recall,
        'duration ratio predicted to truth': duration_predicted_seconds / duration_truth_seconds if duration_truth_seconds != 0.0 else 0.0,
    }
    if duration_audio_seconds is not None:
        metrics['duration ratio predicted to audio'] = duration_predicted_seconds / duration_audio_seconds if duration_audio_seconds != 0.0 else 0.0
        metrics['duration ratio truth to audio'] = duration_truth_seconds / duration_audio_seconds if duration_audio_seconds != 0.0 else 0.0
    metrics_raw = {
        'ious': ious,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
    }
    return metrics, metrics_raw


def print_metrics(metrics):
    for k, v in sorted(metrics.items(), key=lambda t: t[0]):
        v_string = f'{v}' if v is None else f'{v:.4f}'
        print(f'{k}: {v_string}')


def _aggregate_ious(f, ious):
    ious = [iou for iou in ious if iou is not None]
    if len(ious) > 0:
        return f(ious)
    else:
        return None


def _precision_recall_from_counts(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else None
    recall = TP / (TP + FN) if (TP + FN) > 0 else None

    return precision, recall
