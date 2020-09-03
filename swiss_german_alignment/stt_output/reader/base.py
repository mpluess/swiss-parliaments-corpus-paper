from abc import ABC, abstractmethod
from typing import List

from ..model import Word


class SttOutputReader(ABC):
    @abstractmethod
    def read(self, path_to_stt_output: str) -> List[Word]:
        pass
