import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
data = pd.read_csv("medicine_data.csv")

# Encode text columns
le_medicine = LabelEncoder()
le_disease = LabelEncoder()

le_safety = LabelEncoder()

data["medicine"] = le_medicine.fit_transform(data["medicine"])
data["disease"] = le_disease.fit_transform(data["disease"])

data["safety"] = le_safety.fit_transform(data["safety"])

X = data[["medicine", "age", "disease", ]]
y = data["safety"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model and encoders
pickle.dump(model, open("medicine_model.pkl", "wb"))
pickle.dump(le_medicine, open("medicine_encoder.pkl", "wb"))
pickle.dump(le_disease, open("disease_encoder.pkl", "wb"))

pickle.dump(le_safety, open("safety_encoder.pkl", "wb"))

print(" AI Model Trained & Saved Successfully")