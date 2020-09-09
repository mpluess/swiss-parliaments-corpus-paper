# swiss-parliaments-corpus-paper
Code for the paper "Swiss Parliaments Corpus, an Automatically Aligned Swiss German Speech to Standard German Text Corpus".  
TODO link

## Example
To run an example alignment of data/transcript.txt to data/audio.flac, follow these steps:
- Create a Python 3.7 environment
- Clone the repository
- Change your working directory to the repo directory
- pip install -r requirements.txt
- python -m spacy download de_core_news_sm
- python example.py

The output will be written to sentence_alignment.txt. You can compare it with the expected output (data/sentence_alignment_expected_output.txt).
