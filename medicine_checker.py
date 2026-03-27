import pickle

# Load model and encoders
model = pickle.load(open("medicine_model.pkl", "rb"))
le_medicine = pickle.load(open("medicine_encoder.pkl", "rb"))
le_disease = pickle.load(open("disease_encoder.pkl", "rb"))

le_safety = pickle.load(open("safety_encoder.pkl", "rb"))

print("💊 AI Medicine Safety Checker\n")

medicine = input("Enter Medicine Name: ")
age = int(input("Enter Age: "))
disease = input("Enter Disease (or None): ")


try:
    medicine_encoded = le_medicine.transform([medicine])[0]
except:
    print("❌ Medicine not found in database")
    exit()

disease_encoded = le_disease.transform([disease])[0]


prediction = model.predict([[medicine_encoded, age, disease_encoded, 
result = le_safety.inverse_transform(prediction)[0]

print("\n🔍 Safety Result:")
if result == "Safe":
    print("✅ SAFE to use (within normal dosage)")
elif result == "Caution":
    print("⚠ USE WITH CAUTION – consult doctor")
else:
    print("❌ UNSAFE – Do NOT use")

print("\n⚠ Disclaimer: This is an advisory AI system, not medical diagnosis.")