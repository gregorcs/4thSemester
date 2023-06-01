import os
import joblib
import pandas as pd

# todo most paths use the same parent folder, could be refactored

# paths
parent_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
model_path = os.path.join(parent_directory, 'Training', 'TrainedModels', 'DrinkingPredictionRepository_RandomForest',
                          'RandomForest_Model.pkl')
scaler_path = os.path.join(parent_directory, 'Training', 'TrainedModels', 'DrinkingPredictionRepository_RandomForest',
                           'scaler.pkl')

# pre-trained models
savedModel = joblib.load(model_path)
savedScaler = joblib.load(scaler_path)

columns_to_encode_drinking = ['age', 'body_type', 'diet', 'drugs', 'education', 'ethnicity', 'job', 'location',
                              'offspring', 'orientation', 'religion', 'sex']


def load_label_encoders(columns):
    encoders = {}
    for column in columns:
        file_path = os.path.join(parent_directory, 'Training', 'TrainedModels',
                                 'DrinkingPredictionRepository_RandomForest', f'label_encoder_{column}.pkl')
        encoders[column] = joblib.load(file_path)
    return encoders


label_encoders = load_label_encoders(columns_to_encode_drinking)


def preprocess_input_data(input_data_list, columns_to_encode, label_encoders):
    df = pd.DataFrame(input_data_list)

    for column_name in columns_to_encode:
        df[column_name] = label_encoders[column_name].transform(df[column_name])

    return df


def prepare_input_array(df):
    input_array = df.values

    scaled_input = savedScaler.transform(input_array)
    return scaled_input


def predict(model, input_array):
    prediction = model.predict(input_array)
    return prediction[0]
