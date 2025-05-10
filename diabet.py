from flask import Flask, request, jsonify
from flask_cors import CORS  # Yangi import
import pickle
import numpy as np

# Flask ilovasini yaratish
app = Flask(__name__)
CORS(app)  # Barcha domenlardan so‘rovlarni qabul qilishga ruxsat beradi

# Modelni yuklash
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# API endpoint: Bashorat qilish
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([data['features']])
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

# Backendni ishga tushirish
if __name__ == '__main__':
    app.run(debug=True)
