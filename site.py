from flask import Flask, request, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Инициализация Flask
app = Flask(__name__)

# Загружаем обученную модель
model = load_model("pneumonia_cnn_model.h5")

# Функция предсказания
def predict_pneumonia(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image) / 255.0  # Нормализация
    image = np.expand_dims(image, axis=0)  # Добавляем размерность batch
    prediction = model.predict(image)[0][0]  # Получаем вероятность
    return "Пневмония" if prediction > 0.5 else "Здоровые легкие"

# Главная страница
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = "uploads/" + file.filename
            file.save(file_path)
            result = predict_pneumonia(file_path)
            return render_template("index.html", prediction=result, filename=file.filename)

    return render_template("index.html", prediction=None)

# Запуск приложения
if __name__ == "__main__":
    app.run(debug=True)
