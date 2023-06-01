import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

data = pd.read_csv('cleaned_data.csv')
label_encoder = LabelEncoder()

columns_to_encode = ['body_type', 'diet', 'drinks', 'drugs', 'ethnicity', 'job', 'location', 'offspring', 'orientation', 'religion', 'sex', 'education']

for column in columns_to_encode:
    data[column] = label_encoder.fit_transform(data[column])

X = data[['age', 'body_type', 'diet', 'drinks', 'drugs', 'education', 'ethnicity', 'income', 'job', 'location', 'offspring', 'orientation', 'religion', 'sex']].values
y = data['smokes'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

_, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Accuracy: {accuracy:.2f}")

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

class_report = classification_report(y_test, y_pred, zero_division='warn')
print(class_report)
