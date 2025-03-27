import tensorflow as tf # Import thư viện TensorFlow
from tensorflow import keras # Import Keras API từ TensorFlow
from tensorflow.keras import layers # Import các lớp mạng nơ-ron từ Keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # Import các callbacks để kiểm soát quá trình huấn luyện
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Import các hàm đánh giá mô hình
import numpy as np # Import thư viện NumPy để làm việc với mảng
import logging # Import thư viện logging để ghi log

def build_and_train_model(X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """Xây dựng và huấn luyện mô hình học sâu."""
    logging.info("Building and training the model") # Ghi log thông báo đang xây dựng và huấn luyện mô hình
    model = keras.Sequential([ # Tạo mô hình *** Sequential ***
    # mô hình được sử dụng là một Mạng Nơ-ron Nhân tạo (Artificial Neural Network - ANN) được xây dựng bằng Keras (thuộc TensorFlow).
        layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)), # Lớp ẩn đầu tiên với 256 nơ-ron và hàm kích hoạt ReLU
        layers.Dropout(0.3), # Lớp Dropout để tránh overfitting
        layers.Dense(128, activation='relu'), # Lớp ẩn thứ hai với 128 nơ-ron và hàm kích hoạt ReLU
        layers.Dropout(0.3), # Lớp Dropout
        layers.Dense(64, activation='relu'), # Lớp ẩn thứ ba với 64 nơ-ron và hàm kích hoạt ReLU
        layers.Dense(1) # Lớp đầu ra với 1 nơ-ron (cho hồi quy)
    ])

    model.compile(optimizer='adam', loss='mse') # Biên dịch mô hình với optimizer Adam và hàm mất mát MSE
    # Sử dụng Adam, một thuật toán tối ưu hóa hiệu quả và phổ biến trong học sâu.
    # Sử dụng mean squared error (MSE) để đo lường sai số giữa giá trị dự đoán và giá trị thực tế.
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # Khởi tạo EarlyStopping
    model_checkpoint = ModelCheckpoint('Project_LPP/best_laptop_model.keras', save_best_only=True) # Khởi tạo ModelCheckpoint
    # Lưu mô hình tốt nhất vào file best_laptop_model.keras.

    # Validation đánh giá mô hình trên tập kiểm tra sau mỗi epoch.
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, # Huấn luyện mô hình
                        validation_split=0.2, verbose=1,
                        callbacks=[early_stopping, model_checkpoint])

    return model, history # Trả về mô hình và lịch sử huấn luyện

def evaluate_model(model, X_test, y_test):
    """Đánh giá mô hình và xuất kết quả."""
    logging.info("Evaluating the model") # Ghi log thông báo đang đánh giá mô hình
    y_pred = model.predict(X_test).flatten() # Dự đoán giá laptop trên tập kiểm tra

    mse = mean_squared_error(y_test, y_pred) # Tính MSE: Sai số bình phương trung bình.
    mae = mean_absolute_error(y_test, y_pred) # Tính MAE Sai số tuyệt đối trung bình.
    rmse = np.sqrt(mse) # Tính RMSE: Căn bậc hai của MSE.
    r2 = r2_score(y_test, y_pred) # Tính R-squared: Đo lường mức độ phù hợp của mô hình với dữ liệu.

    logging.info(f'Mean Squared Error: {mse}') # Ghi log MSE
    logging.info(f'Mean Absolute Error: {mae}') # Ghi log MAE
    logging.info(f'Root Mean Squared Error: {rmse}') # Ghi log RMSE
    logging.info(f'R-squared: {r2}') # Ghi log R-squared

    return mse, mae, rmse, r2 # Trả về các chỉ số đánh giá

# Mô hình này được thiết kế để giải quyết bài toán hồi quy (regression)
# cụ thể là dự đoán giá laptop dựa trên các đặc trưng đầu vào.