from flask import Flask, render_template, request  # Import các module từ Flask để tạo ứng dụng web
import tensorflow as tf  # Import TensorFlow để load mô hình đã huấn luyện
import numpy as np  # Import NumPy để làm việc với mảng
import pandas as pd  # Import Pandas để làm việc với dữ liệu dạng bảng
import re  # Import re để làm việc với biểu thức chính quy
import joblib  # Import joblib để load preprocessor

app = Flask(__name__)  # Tạo một instance của Flask, gán cho biến app

# Load model đã huấn luyện
model = tf.keras.models.load_model('laptop_price_prediction_model.keras')  # Load mô hình từ file .keras

# Load preprocessor đã lưu
preprocessor = joblib.load('preprocessor.joblib')  # Load preprocessor từ file .joblib

# Load dữ liệu gốc để lấy các giá trị duy nhất
data = pd.read_csv('laptop_prices.csv')  # Đọc dữ liệu từ file CSV
data = data.dropna()  # Loại bỏ các hàng có giá trị NaN

# Lấy các giá trị duy nhất cho các cột categorical
brands = sorted(data['Brand'].unique().tolist())  # Lấy danh sách các thương hiệu duy nhất, sắp xếp và chuyển thành list
processors = sorted(data['Processor'].unique().tolist())  # Lấy danh sách các bộ xử lý duy nhất, sắp xếp và chuyển thành list
gpus = sorted(data['GPU'].unique().tolist())  # Lấy danh sách các GPU duy nhất, sắp xếp và chuyển thành list
resolutions = sorted(data['Resolution'].unique().tolist())  # Lấy danh sách các độ phân giải duy nhất, sắp xếp và chuyển thành list
operating_systems = sorted(data['Operating System'].unique().tolist())  # Lấy danh sách các hệ điều hành duy nhất, sắp xếp và chuyển thành list

# Hàm tiền xử lý dữ liệu đầu vào
def preprocess_input(data):
    # Tạo DataFrame từ dữ liệu đầu vào
    input_df = pd.DataFrame([data])  # Tạo DataFrame từ dữ liệu đầu vào (dạng dictionary)

    # Trích xuất Storage_GB từ cột Storage
    def extract_storage_gb(storage_str):
        match = re.search(r'(\d+)(?:GB|TB)', storage_str)  # Tìm số và đơn vị (GB hoặc TB) trong chuỗi Storage
        if match:
            size = int(match.group(1))  # Lấy số từ kết quả tìm kiếm
            if 'TB' in storage_str:
                size *= 1024  # Chuyển đổi TB sang GB
            return size  # Trả về dung lượng lưu trữ (GB)
        else:
            return None  # Trả về None nếu không tìm thấy

    input_df['Storage_GB'] = input_df['Storage'].apply(extract_storage_gb).astype(float)  # Áp dụng hàm trích xuất và chuyển sang float
    input_df = input_df.drop('Storage', axis=1)  # Loại bỏ cột Storage gốc

    # Tiền xử lý dữ liệu sử dụng preprocessor đã load
    X = preprocessor.transform(input_df)  # Sử dụng transform, không phải fit_transform, vì preprocessor đã được fit trên dữ liệu huấn luyện

    return X  # Trả về dữ liệu đã tiền xử lý

@app.route('/', methods=['GET', 'POST'])  # Định nghĩa route '/' cho cả phương thức GET và POST
def index():
    prediction = None  # Khởi tạo biến prediction là None
    if request.method == 'POST':  # Nếu phương thức là POST (dữ liệu được gửi từ form)
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
        input_data = preprocess_input(data)  # Tiền xử lý dữ liệu đầu vào
        prediction = model.predict(input_data)[0][0]  # Dự đoán giá và lấy giá trị đầu tiên từ kết quả dự đoán

    return render_template('index.html', prediction=prediction, brands=brands, processors=processors, gpus=gpus, resolutions=resolutions, operating_systems=operating_systems)  # Render template index.html, truyền dữ liệu vào template

if __name__ == '__main__':
    app.run(debug=True)  # Chạy ứng dụng Flask ở chế độ debug