import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load and preprocess data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(patient_data.csv)
    df.rename(columns={'C':'Gender'}, inplace=True)
    
    # Columns to encode
    columns = ['Gender','Age','History','Patient','TakeMedication',
              'Severity','BreathShortness','VisualChanges',
              'NoseBleeding','Whendiagnoused','Systolic',
              'Diastolic','ControlledDiet','Stages']
    
    # Label encoding
    label_encoder = LabelEncoder()
    for col in columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    return df

# Train and save model
def train_and_save_model(df):
    X = df.drop('Stages', axis=1)
    y = df['Stages']
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    random_forest = RandomForestClassifier()
    random_forest.fit(x_train, y_train)
    
    # Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(random_forest, f)
    
    return random_forest

if __name__ == "__main__":
    # Update this path to your dataset location
    data_path = 'patient_data.csv'  
    df = load_and_preprocess_data(data_path)
    model = train_and_save_model(df)
    print("Model trained and saved as model.pkl")
