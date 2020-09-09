from swiss_german_alignment.files import write_sentence_alignment
from swiss_german_alignment.sentence_aligner import SentenceAligner
from swiss_german_alignment.stt_output.reader.amazon import AmazonSttOutputReader
from swiss_german_alignment.text_processing import preprocess_transcript_for_sentence_split, split_to_sentences

PATH_TO_SENTENCE_ALIGNER = 'models/sentence_aligner__i4ds_alignment_corpus__amazon_transcribe.pickle'

PATH_TO_AMAZON_TRANSCRIBE_OUTPUT = 'data/amazon_transcribe_output.json'
PATH_TO_TRANSCRIPT = 'data/transcript.txt'

sentence_aligner = SentenceAligner.load(PATH_TO_SENTENCE_ALIGNER)

with open(PATH_TO_TRANSCRIPT, encoding='utf-8') as f:
    transcript = f.read()
    transcript = preprocess_transcript_for_sentence_split(transcript)
    truth_sentences = split_to_sentences(transcript)

stt_output = AmazonSttOutputReader().read(PATH_TO_AMAZON_TRANSCRIBE_OUTPUT)

sentence_alignment, _, _ = sentence_aligner.predict_one(
    truth_sentences, stt_output,
    do_length_ratio_full_transcript_filtering=False,
    do_time_correction=False,
    do_iou_estimate_filtering=False,
)
write_sentence_alignment(sentence_alignment, 'sentence_alignment.txt')
