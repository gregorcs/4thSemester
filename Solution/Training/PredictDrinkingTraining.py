import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib


def encode_columns(data, columns):
    label_encoders = {}
    for column in columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data, label_encoders


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division='warn')
    print(f"Accuracy of {model_name}: {accuracy:.2f}")
    print(f"{model_name} classification report:\n{class_report}")
    return model


def save_model(model, model_name, label_encoders, scaler):
    os.makedirs(f'TrainedModels/DrinkingPredictionRepository_{model_name}', exist_ok=True)
    joblib.dump(model, f"TrainedModels/DrinkingPredictionRepository_{model_name}/{model_name}_Model.pkl")
    joblib.dump(scaler, f'TrainedModels/DrinkingPredictionRepository_{model_name}/scaler.pkl')
    for column, encoder in label_encoders.items():
        joblib.dump(encoder, f'TrainedModels/DrinkingPredictionRepository_{model_name}/label_encoder_{column}.pkl')


def main():
    data = pd.read_csv('cleaned_data.csv')
    columns_to_encode = ['age', 'body_type', 'diet', 'drugs', 'education', 'ethnicity', 'job', 'location',
                         'offspring', 'orientation', 'religion', 'sex']

    data, label_encoders = encode_columns(data, columns_to_encode)
    X = data[columns_to_encode]
    y = data['drinks']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=300, random_state=300)
    rf_model = train_and_evaluate_model(rf_model, X_train_smote, y_train_smote, X_test, y_test, 'RandomForest')
    save_model(rf_model, 'RandomForest', label_encoders, scaler)

    xgb_model = xgb.XGBClassifier(random_state=42, objective='multi:softmax')
    param_grid = {'n_estimators': [100, 300, 500], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7], }
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_smote, y_train_smote)
    best_xgb_model = grid_search.best_estimator_
    best_xgb_model = train_and_evaluate_model(best_xgb_model, X_train_smote, y_train_smote, X_test, y_test, 'XGBoost')
    save_model(best_xgb_model, 'XGBoost', label_encoders, scaler)


if __name__ == "__main__":
    main()
