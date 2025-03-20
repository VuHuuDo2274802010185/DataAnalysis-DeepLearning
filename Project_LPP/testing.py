import pandas as pd
from tensorflow.keras.models import load_model
from data_preprocessing import extract_storage_gb  # Import hàm extract_storage_gb
import joblib  # Import joblib để tải preprocessor

# Tải mô hình đã lưu
model = load_model('laptop_price_prediction_model.keras')

# Tải preprocessor đã lưu
preprocessor = joblib.load('preprocessor.pkl')

# Dữ liệu mới
new_data = pd.DataFrame({
    'RAM (GB)': [16],
    'Storage': ['512GB'],
    'Screen Size (inch)': [15.6],
    'Battery Life (hours)': [10],
    'Weight (kg)': [1.8],
    'Brand': ['Dell'],
    'Processor': ['Intel Core i7'],
    'GPU': ['NVIDIA GTX 1650'],
    'Resolution': ['1920x1080'],
    'Operating System': ['Windows 10']
})

# Tiền xử lý dữ liệu mới
new_data['Storage_GB'] = new_data['Storage'].apply(extract_storage_gb).astype(float)
new_data = new_data.drop('Storage', axis=1)  # Loại bỏ cột 'Storage' gốc
new_data_processed = preprocessor.transform(new_data)  # Áp dụng bộ tiền xử lý

# Dự đoán giá laptop
predicted_price = model.predict(new_data_processed)
print(f"Predicted Price: {predicted_price[0]}")