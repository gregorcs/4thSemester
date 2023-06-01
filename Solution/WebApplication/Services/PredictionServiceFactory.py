from .drinking_prediction_service import DrinkingPredictionService
from .smoking_prediction_service import SmokingPredictionService

def create_prediction_service(savedModel):
    if model_type == 'drinking':
        return DrinkingPredictionService(savedModel)
    elif model_type == 'smoking':
        return SmokingPredictionService(savedModel)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")
