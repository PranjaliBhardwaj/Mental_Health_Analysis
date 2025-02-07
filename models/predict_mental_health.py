import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


with open("models/mental_health_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

def predict_mental_health(symptoms):
    
df = pd.read_csv("data/survey.csv")
    feature_columns = df.drop(columns=["treatment"]).columns

    
    input_data = pd.DataFrame([symptoms], columns=feature_columns)
    
    # Standardize input	scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)

  
    prediction = model.predict(input_scaled)
    return "Requires Treatment" if prediction[0] == 1 else "No Treatment Needed"

if __name__ == "__main__":
    
    test_input = {"Age": 25, "Gender": "Male", "Self-employed": "No", ...}  # Fill other fields
    print(predict_mental_health(test_input))
