import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# 1️⃣ Load CLEAN dataset
data = pd.read_csv("datasets/mumbai_house_price.csv")

# 2️⃣ Encode location
location_encoder = LabelEncoder()
data["location_encoded"] = location_encoder.fit_transform(data["location"])

# 3️⃣ Features & Target
X = data[["bhk", "area", "location_encoded"]]
y = data["price_lakhs"]

# 4️⃣ Train model
model = LinearRegression()
model.fit(X, y)

# 5️⃣ Save model & encoder
pickle.dump(model, open("models/house_price_mumbai.pkl", "wb"))
pickle.dump(location_encoder, open("models/location_encoder.pkl", "wb"))

print("✅ House Price Prediction Model Trained & Saved Successfully")
