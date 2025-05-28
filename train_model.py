import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load data
df = pd.read_csv('patient_data.csv')  # Ensure this CSV is in the same folder
df.rename(columns={'C': 'Gender'}, inplace=True)

# Identify categorical columns
categorical_columns = ['Gender', 'History', 'Patient', 'TakeMedication',
                       'Severity', 'BreathShortness', 'VisualChanges',
                       'NoseBleeding', 'Whendiagnoused', 'ControlledDiet']

# Encode categorical columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Separate features and target
X = df.drop('Stages', axis=1)
y = df['Stages']

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open('model.pkl', 'wb'))
print("Model saved as model.pkl")
