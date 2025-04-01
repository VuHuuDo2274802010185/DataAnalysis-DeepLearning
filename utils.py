import os  # Import thư viện os để làm việc với đường dẫn file
import numpy as np  # Import thư viện numpy để làm việc với mảng và các phép toán số học
import matplotlib.pyplot as plt  # Import thư viện matplotlib để vẽ biểu đồ

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
    if not isinstance(y_test, (list, np.ndarray)) or not isinstance(y_pred, (list, np.ndarray)):  # Kiểm tra nếu y_test hoặc y_pred không phải là list hoặc numpy array
        raise TypeError("y_test và y_pred phải là danh sách hoặc mảng số.")  # Ném ngoại lệ TypeError nếu kiểu dữ liệu không hợp lệ
    if len(y_test) == 0 or len(y_pred) == 0:  # Kiểm tra nếu y_test hoặc y_pred rỗng
        raise ValueError("Dữ liệu y_test hoặc y_pred không được rỗng.")  # Ném ngoại lệ ValueError nếu dữ liệu rỗng
    if len(y_test) != len(y_pred):  # Kiểm tra nếu kích thước của y_test và y_pred không bằng nhau
        raise ValueError("Dữ liệu y_test và y_pred phải có cùng kích thước.")  # Ném ngoại lệ ValueError nếu kích thước không bằng nhau

    # Chuyển đổi sang numpy array nếu cần
    y_test = np.array(y_test)  # Chuyển đổi y_test sang numpy array
    y_pred = np.array(y_pred)  # Chuyển đổi y_pred sang numpy array

    # Vẽ đồ thị
    plt.figure(figsize=figsize)  # Tạo hình vẽ với kích thước được chỉ định
    plt.scatter(y_test, y_pred, alpha=0.7, label="Predictions")  # Vẽ scatter plot: giá trị thực tế vs giá trị dự đoán, độ trong suốt 0.7, nhãn "Predictions"
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label="Ideal Fit")  # Vẽ đường y=x (đường lý tưởng), màu đỏ, nhãn "Ideal Fit"
    plt.xlabel("Giá thực tế")  # Đặt nhãn trục x
    plt.ylabel("Giá dự đoán")  # Đặt nhãn trục y
    plt.title("Giá thực tế vs. Giá dự đoán")  # Đặt tiêu đề cho đồ thị
    plt.legend()  # Hiển thị chú thích
    plt.grid(True)  # Thêm lưới để dễ đọc

    # Lưu đồ thị nếu cần
    if save_path:  # Nếu save_path được cung cấp
        # Kiểm tra xem thư mục có tồn tại không
        save_dir = os.path.dirname(save_path)  # Lấy đường dẫn thư mục từ save_path
        if save_dir and not os.path.exists(save_dir):  # Nếu save_dir tồn tại và không có thư mục nào tồn tại
            os.makedirs(save_dir)  # Tạo thư mục nếu chưa tồn tại
        plt.savefig(save_path)  # Lưu đồ thị tại đường dẫn được chỉ định
        print(f"Đồ thị dự đoán đã được lưu tại: {save_path}")  # In thông báo lưu thành công
    
    plt.show()  # Hiển thị đồ thị