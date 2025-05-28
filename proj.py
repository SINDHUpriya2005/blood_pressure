import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

def load_and_preprocess_data(patient_data.csv):
    try:
        df = pd.read_csv(patient_data.csv)
        df.rename(columns={'C':'Gender'}, inplace=True)
        
        columns = ['Gender','Age','History','Patient','TakeMedication',
                 'Severity','BreathShortness','VisualChanges',
                 'NoseBleeding','Whendiagnoused','Systolic',
                 'Diastolic','ControlledDiet','Stages']
        
        label_encoder = LabelEncoder()
        for col in columns:
            df[col] = label_encoder.fit_transform(df[col])
        
        return df
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None

def train_and_save_model(df):
    X = df.drop('Stages', axis=1)
    y = df['Stages']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    data_path = BASE_DIR / "data" / "patient_data.csv"
    
    print(f"Loading data from: {data_path}")
    
    if not data_path.exists():
        print("Error: Data file not found!")
        print("Current directory contents:")
        for item in BASE_DIR.iterdir():
            print(f" - {item.name}")
    else:
        df = load_and_preprocess_data(patient_data.csv)
        if df is not None:
            model = train_and_save_model(df)
            print("Model trained and saved as model.pkl")
            print(f"Model accuracy: {model.score(*train_test_split(X, y, test_size=0.2, random_state=42))}")
