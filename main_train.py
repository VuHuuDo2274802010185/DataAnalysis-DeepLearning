import logging  # Import thư viện logging để ghi log thông tin, lỗi
import os  # Import thư viện os để làm việc với đường dẫn file
import joblib  # Import thư viện joblib để lưu và tải mô hình, preprocessor
import pandas as pd  # Import thư viện pandas để làm việc với dữ liệu dạng bảng
import numpy as np  # Import thư viện numpy để làm việc với mảng và các phép toán số học
import matplotlib.pyplot as plt  # Import thư viện matplotlib để vẽ biểu đồ
from data_preprocessing import load_and_preprocess_data  # Import hàm load_and_preprocess_data từ file data_preprocessing.py
from model import build_and_train_model, evaluate_model  # Import hàm build_and_train_model, evaluate_model từ file model.py.
from utils import plot_predictions  # Import hàm plot_predictions từ file utils.py.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Cấu hình logging: ghi log ở mức INFO, định dạng thời gian, mức độ, và thông báo.

def remove_outliers(df, target, threshold=3):
    """Loại bỏ các điểm ngoại lệ dựa trên z-score."""
    z_scores = np.abs((df[target] - df[target].mean()) / df[target].std())  # Tính z-score cho cột mục tiêu.
    return df[z_scores < threshold]  # Trả về DataFrame chỉ chứa các hàng có z-score nhỏ hơn ngưỡng.

def plot_loss(history):
    """Vẽ biểu đồ loss trong quá trình huấn luyện."""
    plt.figure(figsize=(10, 6))  # Tạo hình vẽ với kích thước 10x6.
    plt.plot(history.history['loss'], label='Training Loss')  # Vẽ loss trên tập huấn luyện.
    plt.plot(history.history['val_loss'], label='Validation Loss')  # Vẽ loss trên tập kiểm tra.
    plt.title('Training and Validation Loss')  # Đặt tiêu đề cho biểu đồ.
    plt.xlabel('Epoch')  # Đặt nhãn trục x.
    plt.ylabel('Loss')  # Đặt nhãn trục y.
    plt.legend()  # Hiển thị chú thích.
    plt.show()  # Hiển thị biểu đồ.

def main():
    """Hàm chính để chạy toàn bộ quy trình Training."""
    try:
        # Xác định đường dẫn tương đối đến file dữ liệu
        data_filepath = os.path.join(os.getcwd(), 'laptop_prices.csv')  # Tạo đường dẫn đến file laptop_prices.csv trong thư mục hiện tại.

        logging.info(f"Loading data from: {data_filepath}")  # Ghi log thông báo đang tải dữ liệu.

        # Load dữ liệu và loại bỏ điểm ngoại lệ
        data = pd.read_csv(data_filepath)  # Đọc dữ liệu từ file CSV.
        data = data.dropna()  # Loại bỏ các hàng có giá trị NaN.
        data = remove_outliers(data, 'Price ($)')  # Loại bỏ các điểm ngoại lệ trong cột 'Price ($)'.

        # Tiền xử lý dữ liệu
        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(data_filepath)  # Tiền xử lý dữ liệu bằng hàm load_and_preprocess_data.

        # Huấn luyện mô hình
        model, history = build_and_train_model(X_train, y_train, X_test, y_test)  # Huấn luyện mô hình bằng hàm build_and_train_model.

        # Đánh giá mô hình
        mse, mae, rmse, r2 = evaluate_model(model, X_test, y_test)  # Đánh giá mô hình bằng hàm evaluate_model.

        # Vẽ biểu đồ loss
        plot_loss(history)  # Vẽ biểu đồ loss trong quá trình huấn luyện.

        # Dự đoán và vẽ biểu đồ dự đoán
        y_pred = model.predict(X_test).flatten()  # Dự đoán giá trị trên tập kiểm tra.
        plot_predictions(y_test, y_pred)  # Vẽ biểu đồ so sánh giá trị thực tế và giá trị dự đoán.

        # Lưu mô hình và preprocessor
        model.save('laptop_price_prediction_model.keras')  # Lưu mô hình đã huấn luyện.
        joblib.dump(preprocessor, 'preprocessor.joblib')  # Lưu preprocessor đã sử dụng.

        logging.info("Model training and evaluation completed successfully.")  # Ghi log thông báo hoàn thành huấn luyện và đánh giá mô hình.

    except FileNotFoundError:
        logging.error(f"File 'laptop_prices.csv' not found in the current directory: {os.getcwd()}")  # Ghi log lỗi nếu file không tìm thấy.
    except Exception as e:
        logging.error(f"An error occurred: {e}")  # Ghi log lỗi nếu có lỗi xảy ra trong quá trình chạy.

if __name__ == "__main__":
    main()  # Gọi hàm main nếu file được chạy trực tiếp.