from abc import ABC, abstractmethod


class Kernel(ABC):
    @abstractmethod
    def compute(self, X, center, sigma):
        raise NotImplementedError()
