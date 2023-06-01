from Solution.WebApplication.Services.BasePredictionService import BasePredictionService
from sklearn.preprocessing import LabelEncoder
import joblib
import os


class DrinkingPredictionService(BasePredictionService):

    def __init__(self, preprocess_input_data, prepare_input_array, predict, columns_to_encode, savedModel):
        self.preprocess_input_data = preprocess_input_data
        self.prepare_input_array = prepare_input_array
        self.predict = predict
        self.columns_to_encode = columns_to_encode
        self.savedModel = savedModel
        self.label_encoders = self.load_label_encoders(columns_to_encode)

    def process_prediction_request(self, input_data):
        print(input_data)
        if not self.is_input_valid(input_data):
            return {'error': 'Invalid input data'}, 400

        input_data = self.preprocess_input_data(input_data, self.columns_to_encode, self.label_encoders)
        input_array = self.prepare_input_array(input_data)
        prediction_label = self.predict(self.savedModel, input_array)

        return {'prediction': prediction_label}, 200

    def is_input_valid(self, input_data):
        required_keys = set(self.columns_to_encode)

        for data in input_data:
            if not isinstance(data, dict):
                return False

            input_keys = set(data.keys())
            if not required_keys.issubset(input_keys):
                return False

        return True

    def load_label_encoders(self, columns):
        parent_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        encoders = {}
        for column in columns:
            file_path = os.path.join('C:\\Users\\Gregor.Csatlos\\Downloads\\OKCupidProject\\Solution\\', 'Training',
                                     'TrainedModels', 'DrinkingPredictionRepository', f'label_encoder_{column}.pkl')
            print(file_path)
            encoders[column] = joblib.load(file_path)
        open(file_path, 'rb')
        return encoders
