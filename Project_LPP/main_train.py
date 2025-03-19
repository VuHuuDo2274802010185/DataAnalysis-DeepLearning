import logging
import os
from tkinter import Tk, Label, Text, Button, filedialog
from data_preprocessing import load_and_preprocess_data
from model import build_and_train_model, evaluate_model
from utils import plot_loss, plot_predictions
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.callbacks import Callback

# Cấu hình logging để ghi log vào GUI
class TkinterHandler(logging.Handler):
    """Custom logging handler to redirect logs to tkinter Text widget."""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        log_message = self.format(record)
        self.text_widget.insert("end", log_message + "\n")
        self.text_widget.see("end")  # Tự động cuộn xuống cuối

# Custom callback để hiển thị log từng epoch lên GUI
class TrainingLogger(Callback):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        message = f"Epoch {epoch + 1}: "
        message += ", ".join([f"{key}={value:.4f}" for key, value in logs.items()])
        self.text_widget.insert("end", message + "\n")
        self.text_widget.see("end")  # Tự động cuộn xuống cuối

def main():
    """Hàm chính để chạy toàn bộ quy trình Training"""
    try:
        # Tạo cửa sổ tkinter
        root = Tk()
        root.title("Laptop Price Prediction - Training Results")
        root.geometry("800x600")

        # Giao diện tkinter
        Label(root, text="Laptop Price Prediction - Training Results", font=("Arial", 16)).pack(pady=10)

        Button(root, text="Select Data File", command=lambda: select_file()).pack(pady=5)

        Label(root, text="Log Output:", font=("Arial", 12)).pack(anchor="w", padx=10)
        log_output = Text(root, height=10, width=80)
        log_output.pack(pady=5)

        Label(root, text="Evaluation Metrics:", font=("Arial", 12)).pack(anchor="w", padx=10)
        metrics_output = Text(root, height=5, width=40)
        metrics_output.pack(pady=5)

        # Cấu hình logging để ghi log vào Text widget
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        tkinter_handler = TkinterHandler(log_output)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        tkinter_handler.setFormatter(formatter)
        logger.addHandler(tkinter_handler)

        # Chọn file dữ liệu
        def select_file():
            filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
            if filepath:
                run_training(filepath)

        # Hàm chạy training
        def run_training(data_filepath):
            try:
                logging.info(f"Loading data from: {data_filepath}")
                X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(data_filepath)

                # Tạo callback để log từng epoch
                training_logger = TrainingLogger(log_output)

                # Train model
                model, history = build_and_train_model(
                    X_train, y_train, X_test, y_test, callbacks=[training_logger]
                )

                # Đánh giá mô hình
                mse, mae, rmse, r2 = evaluate_model(model, X_test, y_test)

                # Hiển thị các chỉ số đánh giá
                metrics_output.insert("end", f"MSE: {mse:.4f}\n")
                metrics_output.insert("end", f"MAE: {mae:.4f}\n")
                metrics_output.insert("end", f"RMSE: {rmse:.4f}\n")
                metrics_output.insert("end", f"R2 Score: {r2:.4f}\n")

                logging.info("Metrics displayed successfully.")

                # Hiển thị biểu đồ loss
                fig_loss = plot_loss(history)
                canvas_loss = FigureCanvasTkAgg(fig_loss, master=root)
                canvas_loss.get_tk_widget().pack()

                # Hiển thị biểu đồ dự đoán
                y_pred = model.predict(X_test).flatten()
                fig_pred = plot_predictions(y_test, y_pred)
                canvas_pred = FigureCanvasTkAgg(fig_pred, master=root)
                canvas_pred.get_tk_widget().pack()

                # Lưu mô hình
                model.save('laptop_price_prediction_model.keras')
                logging.info("Model training and evaluation completed successfully.")
            except FileNotFoundError:
                logging.error(f"File 'laptop_prices.csv' not found in the current directory: {os.getcwd()}")
            except Exception as e:
                logging.error(f"An error occurred: {e}")

        root.mainloop()

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()