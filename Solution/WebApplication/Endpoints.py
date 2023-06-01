from flask import Blueprint, render_template, request, jsonify
from SmokingPredictionLogic import preprocess_input_data as preprocess_input_data_smoking, \
    prepare_input_array as prepare_input_array_smoking, predict as predict_smoking, \
    columns_to_encode_smoking, savedModel as savedModel_smoking
from DrinkingPredictionLogic import preprocess_input_data as preprocess_input_data_drinking, \
    prepare_input_array as prepare_input_array_drinking, predict as predict_drinking, \
    columns_to_encode_drinking, savedModel as savedModel_drinking
from Solution.WebApplication.Services.DrinkingPredictionService import DrinkingPredictionService
from Solution.WebApplication.Services.SmokingPredictionService import SmokingPredictionService

endpoints = Blueprint('endpoints', __name__)

smoking_prediction_service = SmokingPredictionService(preprocess_input_data_smoking,
                                                      prepare_input_array_smoking,
                                                      predict_smoking,
                                                      columns_to_encode_smoking,
                                                      savedModel_smoking)

drinking_prediction_service = DrinkingPredictionService(preprocess_input_data_drinking,
                                                        prepare_input_array_drinking,
                                                        predict_drinking,
                                                        columns_to_encode_drinking,
                                                        savedModel_drinking)

@endpoints.route('/predict-smoking', methods=['POST'])
def predict_smoking():
    if request.is_json:
        input_data = request.get_json()
        response, status_code = smoking_prediction_service.process_prediction_request(input_data)
        return jsonify(response), status_code
    else:
        return jsonify({'error': 'Request must be JSON'}), 400


@endpoints.route('/predict-drinking', methods=['POST'])
def predict_drinking():
    if request.is_json:
        input_data = request.get_json()
        response, status_code = drinking_prediction_service.process_prediction_request(input_data)
        return jsonify(response), status_code
    else:
        return jsonify({'error': 'Request must be JSON'}), 400


@endpoints.route('/DrinkingPrediction', methods=['GET'])
def drinking_prediction():
    return render_template('DrinkingPrediction.html')
