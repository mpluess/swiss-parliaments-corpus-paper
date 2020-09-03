class Word:
    def __init__(self, word: str, start_time: float, end_time: float, confidence: float):
        self.word = word
        self.start_time = start_time
        self.end_time = end_time
        self.confidence = confidence

    def __repr__(self):
        return f'Word(word={self.word}, start_time={self.start_time}, end_time={self.end_time}, confidence={self.confidence})'
