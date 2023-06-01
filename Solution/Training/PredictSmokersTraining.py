import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
import joblib

def encode_columns(data, columns):
    label_encoders = {}
    for column in columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data, label_encoders

def apply_oversampling(X_train, y_train):
    over_sampler = RandomOverSampler(random_state=42)
    X_train_oversampled, y_train_oversampled = over_sampler.fit_resample(X_train, y_train)
    return X_train_oversampled, y_train_oversampled

def train_and_evaluate_models(models, X_train, y_train, X_test, y_test, model_dir):
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, zero_division='warn')
        print(f"Accuracy of {name}: {accuracy:.2f}")
        print(f"{name} classification report:\n{class_report}")
        joblib.dump(model, os.path.join(model_dir, f"{name.replace(' ', '_')}_Model.pkl"))

def main():
    data = pd.read_csv('cleaned_data.csv')
    columns_to_encode = ['age', 'body_type', 'diet', 'drinks', 'drugs', 'education', 'ethnicity', 'job', 'location',
                         'offspring', 'orientation', 'religion', 'sex']

    # Data Preprocessing
    data, label_encoders = encode_columns(data, columns_to_encode)
    X = data[columns_to_encode]
    y = data['smokes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_oversampled, y_train_oversampled = apply_oversampling(X_train, y_train)

    models = [
        ('Logistic Regression', LogisticRegression(max_iter=2000)),
        ('Random Forest', RandomForestClassifier(n_estimators=150, random_state=150))
    ]
    model_dir = "TrainedModels/SmokingPredictionRepository"
    os.makedirs(model_dir, exist_ok=True)
    train_and_evaluate_models(models, X_train_oversampled, y_train_oversampled, X_test, y_test, model_dir)

    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

    for column, encoder in label_encoders.items():
        joblib.dump(encoder, os.path.join(model_dir, f'label_encoder_{column}.pkl'))

if __name__ == "__main__":
    main()
