import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

data = pd.read_csv('cleaned_data.csv')
label_encoder = LabelEncoder()

columns_to_encode = ['body_type', 'diet', 'drugs', 'ethnicity', 'job', 'location', 'offspring',
                     'orientation', 'religion', 'sex', 'education']

for column in columns_to_encode:
    data[column] = label_encoder.fit_transform(data[column])

print(data.head())

X = data[['age', 'body_type', 'diet', 'drugs', 'education', 'ethnicity', 'job', 'location',
          'offspring', 'orientation', 'religion', 'sex']]
y = data['drinks']

y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

xgb_model = xgb.XGBClassifier(random_state=42, objective='multi:softmax')

param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
}

grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_smote, y_train_smote)

best_xgb_model = grid_search.best_estimator_
y_pred = best_xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of XGBoost: {accuracy}")

class_report = classification_report(y_test, y_pred, zero_division='warn')
print(class_report)
