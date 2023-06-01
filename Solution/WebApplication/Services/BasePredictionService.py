from abc import ABC, abstractmethod

class BasePredictionService(ABC):

    @abstractmethod
    def process_prediction_request(self, request):
        pass
