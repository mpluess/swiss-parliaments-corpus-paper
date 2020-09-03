import re
from typing import Set

import spacy

WHITESPACE_REGEX = re.compile(r'[ \t]+')

NLP = spacy.load('de_core_news_sm', disable=['ner'])


def preprocess_transcript_for_alignment(transcript: str, alphabet: Set[str]):
    transcript = ''.join([char for char in transcript if char in alphabet])
    transcript = transcript.strip()

    return transcript


def preprocess_transcript_for_sentence_split(transcript):
    transcript = transcript.replace('-\n', '')
    transcript = transcript.replace(' \n', ' ')
    transcript = transcript.replace('\n', ' ')
    transcript = transcript.replace('\t', ' ')

    transcript = WHITESPACE_REGEX.sub(' ', transcript)
    transcript = transcript.strip()

    return transcript


def split_to_sentences(transcript):
    doc = NLP(transcript)

    return [sent.text for sent in doc.sents]
