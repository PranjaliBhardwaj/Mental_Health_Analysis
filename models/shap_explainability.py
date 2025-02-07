import shap
import pickle
import pandas as pd
import numpy as np

# Load model
with open("models/mental_health_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load dataset
df = pd.read_csv("data/survey.csv")
X = df.drop(columns=["treatment"])

# Create SHAP Explainer
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Visualize SHAP summary plot
shap.summary_plot(shap_values, X)
