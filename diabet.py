from flask import Flask, request, jsonify
import pickle
import numpy as np

# Flask ilovasini yaratish
app = Flask(__name__)

# Modelni yuklash
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)


# API endpoint: Bashorat qilish
@app.route('/predict', methods=['POST'])
def predict():
    # So'rovda yuborilgan ma'lumotlarni olish
    data = request.get_json(force=True)

    # Ma'lumotlar formati (masalan, yangi ma'lumotlar uchun 8 ta xususiyat)
    features = np.array([data['features']])  # request body'dan ma'lumot olish

    # Modelni ishlatib bashorat qilish
    prediction = model.predict(features)

    # Natijani JSON formatda qaytarish
    return jsonify({'prediction': prediction.tolist()})


# Backendni ishga tushirish
if __name__ == '__main__':
    app.run(debug=True)
