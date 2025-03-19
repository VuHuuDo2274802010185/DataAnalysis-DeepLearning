import tensorflow as tf # Import thư viện TensorFlow
from tensorflow import keras # Import Keras API từ TensorFlow
from tensorflow.keras import layers # Import các lớp mạng nơ-ron từ Keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # Import các callbacks để kiểm soát quá trình huấn luyện
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Import các hàm đánh giá mô hình
import numpy as np # Import thư viện NumPy để làm việc với mảng
import logging # Import thư viện logging để ghi log
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def build_and_train_model(X_train, y_train, X_test, y_test, callbacks=None):
    """
    Xây dựng và huấn luyện mô hình.
    :param X_train: Dữ liệu huấn luyện
    :param y_train: Nhãn huấn luyện
    :param X_test: Dữ liệu kiểm tra
    :param y_test: Nhãn kiểm tra
    :param callbacks: Danh sách các callback (mặc định là None)
    :return: Mô hình đã huấn luyện và lịch sử huấn luyện
    """
    # Xây dựng mô hình
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer
    ])

    # Compile mô hình
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    # Huấn luyện mô hình
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks  # Truyền callbacks vào đây
    )

    return model, history

def evaluate_model(model, X_test, y_test):
    """Đánh giá mô hình và xuất kết quả."""
    logging.info("Evaluating the model") # Ghi log thông báo đang đánh giá mô hình
    y_pred = model.predict(X_test).flatten() # Dự đoán giá laptop trên tập kiểm tra

    mse = mean_squared_error(y_test, y_pred) # Tính MSE
    mae = mean_absolute_error(y_test, y_pred) # Tính MAE
    rmse = np.sqrt(mse) # Tính RMSE
    r2 = r2_score(y_test, y_pred) # Tính R-squared

    logging.info(f'Mean Squared Error: {mse}') # Ghi log MSE
    logging.info(f'Mean Absolute Error: {mae}') # Ghi log MAE
    logging.info(f'Root Mean Squared Error: {rmse}') # Ghi log RMSE
    logging.info(f'R-squared: {r2}') # Ghi log R-squared

    return mse, mae, rmse, r2 # Trả về các chỉ số đánh giá