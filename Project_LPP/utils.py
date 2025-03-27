import os
import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(y_test, y_pred, save_path=None, figsize=(10, 6)):
    """
    Vẽ đồ thị dự đoán so với thực tế.
    
    Args:
        y_test (list | np.ndarray): Giá trị thực tế (mảng hoặc danh sách số).
        y_pred (list | np.ndarray): Giá trị dự đoán (mảng hoặc danh sách số).
        save_path (str, optional): Đường dẫn để lưu đồ thị (nếu cần). Mặc định là None.
        figsize (tuple, optional): Kích thước của đồ thị. Mặc định là (10, 6).
    
    Raises:
        ValueError: Nếu y_test hoặc y_pred rỗng hoặc không có cùng kích thước.
        TypeError: Nếu y_test hoặc y_pred không phải là danh sách hoặc mảng số.
    """
    # Kiểm tra kiểu dữ liệu
    if not isinstance(y_test, (list, np.ndarray)) or not isinstance(y_pred, (list, np.ndarray)):
        raise TypeError("y_test và y_pred phải là danh sách hoặc mảng số.")
    if len(y_test) == 0 or len(y_pred) == 0:
        raise ValueError("Dữ liệu y_test hoặc y_pred không được rỗng.")
    if len(y_test) != len(y_pred):
        raise ValueError("Dữ liệu y_test và y_pred phải có cùng kích thước.")

    # Chuyển đổi sang numpy array nếu cần
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Vẽ đồ thị
    plt.figure(figsize=figsize)
    plt.scatter(y_test, y_pred, alpha=0.7, label="Predictions")  # Vẽ scatter plot
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label="Ideal Fit")  # Đường y=x
    plt.xlabel("Giá thực tế")
    plt.ylabel("Giá dự đoán")
    plt.title("Giá thực tế vs. Giá dự đoán")
    plt.legend()
    plt.grid(True)  # Thêm lưới để dễ đọc

    # Lưu đồ thị nếu cần
    if save_path:
        # Kiểm tra xem thư mục có tồn tại không
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)  # Tạo thư mục nếu chưa tồn tại
        plt.savefig(save_path)
        print(f"Đồ thị dự đoán đã được lưu tại: {save_path}")
    
    plt.show()