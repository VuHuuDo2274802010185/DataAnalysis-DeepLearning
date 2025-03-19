import logging
import os
from data_preprocessing import load_and_preprocess_data
from model import build_and_train_model, evaluate_model
from utils import plot_loss, plot_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Hàm chính để chạy toàn bộ quy trình Training"""
    try:
        # Xác định đường dẫn tương đối đến file dữ liệu
        data_filepath = os.path.join(os.getcwd(), 'laptop_prices.csv')

        logging.info(f"Loading data from: {data_filepath}")

        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(data_filepath)

        model, history = build_and_train_model(X_train, y_train, X_test, y_test)

        mse, mae, rmse, r2 = evaluate_model(model, X_test, y_test)

        plot_loss(history)

        y_pred = model.predict(X_test).flatten()

        plot_predictions(y_test, y_pred)

        model.save('laptop_price_prediction_model.keras')

        logging.info("Model training and evaluation completed successfully.")

    except FileNotFoundError:
        logging.error(f"File 'laptop_prices.csv' not found in the current directory: {os.getcwd()}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()