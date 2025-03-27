from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pandas as pd
import re
import joblib  # Import joblib để load preprocessor

app = Flask(__name__)

# Load model đã huấn luyện
model = tf.keras.models.load_model('laptop_price_prediction_model.keras')

# Load preprocessor đã lưu
preprocessor = joblib.load('preprocessor.joblib')

# Load dữ liệu gốc để lấy các giá trị duy nhất
data = pd.read_csv('laptop_prices.csv')
data = data.dropna()

# Lấy các giá trị duy nhất cho các cột categorical
brands = sorted(data['Brand'].unique().tolist())
processors = sorted(data['Processor'].unique().tolist())
gpus = sorted(data['GPU'].unique().tolist())
resolutions = sorted(data['Resolution'].unique().tolist())
operating_systems = sorted(data['Operating System'].unique().tolist())

# Hàm tiền xử lý dữ liệu đầu vào
def preprocess_input(data):
    # Tạo DataFrame từ dữ liệu đầu vào
    input_df = pd.DataFrame([data])

    # Trích xuất Storage_GB từ cột Storage
    def extract_storage_gb(storage_str):
        match = re.search(r'(\d+)(?:GB|TB)', storage_str)
        if match:
            size = int(match.group(1))
            if 'TB' in storage_str:
                size *= 1024
            return size
        else:
            return None

    input_df['Storage_GB'] = input_df['Storage'].apply(extract_storage_gb).astype(float)
    input_df = input_df.drop('Storage', axis=1)

    # Tiền xử lý dữ liệu sử dụng preprocessor đã load
    X = preprocessor.transform(input_df)  # Sử dụng transform, không phải fit_transform

    return X

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        data = {
            'Brand': request.form['brand'],
            'Processor': request.form['processor'],
            'RAM (GB)': float(request.form['ram']),
            'Storage': request.form['storage'],
            'GPU': request.form['gpu'],
            'Screen Size (inch)': float(request.form['screen_size']),
            'Resolution': request.form['resolution'],
            'Operating System': request.form['os'],
            'Battery Life (hours)': float(request.form['battery_life']),
            'Weight (kg)': float(request.form['weight'])
        }

        # Tiền xử lý và dự đoán
        input_data = preprocess_input(data)
        prediction = model.predict(input_data)[0][0]

    return render_template('index.html', prediction=prediction, brands=brands, processors=processors, gpus=gpus, resolutions=resolutions, operating_systems=operating_systems)

if __name__ == '__main__':
    app.run(debug=True)