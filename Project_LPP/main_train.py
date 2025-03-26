import logging
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess_data
from model import build_and_train_model, evaluate_model
from utils import plot_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def remove_outliers(df, target, threshold=3):
    """Loại bỏ các điểm ngoại lệ dựa trên z-score."""
    z_scores = np.abs((df[target] - df[target].mean()) / df[target].std())
    return df[z_scores < threshold]

def plot_loss(history):
    """Vẽ biểu đồ loss trong quá trình huấn luyện."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    """Hàm chính để chạy toàn bộ quy trình Training."""
    try:
        # Xác định đường dẫn tương đối đến file dữ liệu
        data_filepath = os.path.join(os.getcwd(), 'laptop_prices.csv')

        logging.info(f"Loading data from: {data_filepath}")

        # Load dữ liệu và loại bỏ điểm ngoại lệ
        data = pd.read_csv(data_filepath)
        data = data.dropna()
        data = remove_outliers(data, 'Price ($)')

        # Tiền xử lý dữ liệu
        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(data_filepath)

        # Huấn luyện mô hình
        model, history = build_and_train_model(X_train, y_train, X_test, y_test)

        # Đánh giá mô hình
        mse, mae, rmse, r2 = evaluate_model(model, X_test, y_test)

        # Vẽ biểu đồ loss
        plot_loss(history)

        # Dự đoán và vẽ biểu đồ dự đoán
        y_pred = model.predict(X_test).flatten()
        plot_predictions(y_test, y_pred)

        # Lưu mô hình và preprocessor
        model.save('laptop_price_prediction_model.keras')
        joblib.dump(preprocessor, 'preprocessor.joblib')

        logging.info("Model training and evaluation completed successfully.")

    except FileNotFoundError:
        logging.error(f"File 'laptop_prices.csv' not found in the current directory: {os.getcwd()}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()