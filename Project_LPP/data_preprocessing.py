import pandas as pd # Import thư viện pandas để làm việc với dữ liệu dạng bảng
from sklearn.model_selection import train_test_split # Import hàm train_test_split để chia dữ liệu thành tập huấn luyện và tập kiểm tra
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Import StandardScaler để chuẩn hóa dữ liệu số và OneHotEncoder để mã hóa dữ liệu phân loại
from sklearn.compose import ColumnTransformer # Import ColumnTransformer để kết hợp các bước tiền xử lý
import re # Import thư viện re để làm việc với biểu thức chính quy
import logging # Import thư viện logging để ghi log

def extract_storage_gb(storage_str):
    """Trích xuất dung lượng lưu trữ (GB) từ chuỗi."""
    match = re.search(r'(\d+)(?:GB|TB)', storage_str) # Tìm số và đơn vị (GB hoặc TB) trong chuỗi
    if match: # Nếu tìm thấy
        size = int(match.group(1)) # Lấy số từ kết quả tìm kiếm
        if 'TB' in storage_str: # Nếu đơn vị là TB
            size *= 1024  # Chuyển đổi TB sang GB
        return size # Trả về dung lượng lưu trữ (GB)
    else:
        return None # Trả về None nếu không tìm thấy

def load_and_preprocess_data(filepath):
    """Đọc và tiền xử lý dữ liệu từ file CSV."""
    logging.info(f"Loading data from {filepath}") # Ghi log thông báo đang đọc dữ liệu
    data = pd.read_csv(filepath) # Đọc dữ liệu từ file CSV
    data = data.dropna() # Loại bỏ các hàng có giá trị NaN

    # Sử lý nhãn 
    data['Storage_GB'] = data['Storage'].apply(extract_storage_gb).astype(float) # Trích xuất dung lượng lưu trữ (GB) từ cột 'Storage'
    data = data.drop('Storage', axis=1) # Loại bỏ cột 'Storage' gốc

    numerical_features = ['RAM (GB)', 'Storage_GB', 'Screen Size (inch)', 'Battery Life (hours)', 'Weight (kg)'] # Danh sách các cột số
    categorical_features = ['Brand', 'Processor', 'GPU', 'Resolution', 'Operating System'] # Danh sách các cột phân loại
    target = 'Price ($)' # Tên cột mục tiêu

    numerical_transformer = StandardScaler() # Khởi tạo StandardScaler
    categorical_transformer = OneHotEncoder(handle_unknown='ignore') # Khởi tạo OneHotEncoder

    preprocessor = ColumnTransformer( # Kết hợp các bước tiền xử lý
        transformers=[
            ('num', numerical_transformer, numerical_features), # Áp dụng StandardScaler cho các cột số
            ('cat', categorical_transformer, categorical_features) # Áp dụng OneHotEncoder cho các cột phân loại
        ])

    X = preprocessor.fit_transform(data.drop(target, axis=1)) # Áp dụng tiền xử lý cho dữ liệu đặc trưng
    y = data[target].values # Lấy dữ liệu mục tiêu

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    return X_train, X_test, y_train, y_test, preprocessor # Trả về dữ liệu đã tiền xử lý và preprocessor