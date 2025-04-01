import pandas as pd  # Import thư viện pandas để làm việc với dữ liệu dạng bảng
import matplotlib.pyplot as plt  # Import thư viện matplotlib để vẽ biểu đồ
import seaborn as sns  # Import thư viện seaborn để vẽ biểu đồ đẹp hơn
import tkinter as tk  # Import thư viện tkinter để tạo giao diện đồ họa
from tkinter import scrolledtext  # Import widget ScrolledText để hiển thị văn bản cuộn

def show_data_info(df, text_widget):
    """Hiển thị thông tin DataFrame trong một widget văn bản."""
    text_widget.insert(tk.END, "Thông tin ban đầu của DataFrame:\n")  # Chèn tiêu đề vào widget văn bản
    text_widget.insert(tk.END, str(df.info()) + "\n")  # Chèn thông tin DataFrame vào widget văn bản
    text_widget.insert(tk.END, "\nSố lượng giá trị null trước khi xử lý:\n")  # Chèn tiêu đề vào widget văn bản
    text_widget.insert(tk.END, str(df.isnull().sum()) + "\n")  # Chèn số lượng giá trị null vào widget văn bản

    # Giải thích:
    # df.info() cung cấp thông tin tổng quan về DataFrame, bao gồm:
    # - Số lượng dòng và cột
    # - Tên cột
    # - Số lượng giá trị không null trong mỗi cột
    # - Kiểu dữ liệu của mỗi cột
    # df.isnull().sum() hiển thị số lượng giá trị null (giá trị thiếu) trong mỗi cột.

def show_invalid_values(df, numeric_cols, text_widget):
    """Hiển thị các giá trị không hợp lệ trong các cột số."""
    for col in numeric_cols:  # Lặp qua từng cột số
        invalid_values = df[pd.to_numeric(df[col], errors='coerce').isnull()]  # Tìm các giá trị không hợp lệ (NaN) sau khi chuyển đổi
        if not invalid_values.empty:  # Nếu có giá trị không hợp lệ
            text_widget.insert(tk.END, f"\nCác giá trị không hợp lệ trong cột '{col}':\n")  # Chèn tiêu đề vào widget văn bản
            text_widget.insert(tk.END, str(invalid_values[col]) + "\n")  # Chèn các giá trị không hợp lệ vào widget văn bản

    # Giải thích:
    # Đoạn code này xử lý các cột số.
    # - Loại bỏ các ký tự không phải số bằng cách thay thế chúng bằng chuỗi rỗng.
    # - Chuyển đổi cột sang kiểu số bằng pd.to_numeric, với errors='coerce' để chuyển các giá trị không hợp lệ thành NaN.
    # - In ra các giá trị không hợp lệ (NaN) sau khi chuyển đổi.

def show_null_counts(df, text_widget, stage):
    """Hiển thị số lượng giá trị null sau khi xử lý."""
    text_widget.insert(tk.END, f"\nSố lượng giá trị null sau khi xử lý {stage}:\n")  # Chèn tiêu đề vào widget văn bản
    text_widget.insert(tk.END, str(df.isnull().sum()) + "\n")  # Chèn số lượng giá trị null vào widget văn bản

    # Giải thích:
    # df.dropna(subset=numeric_cols) loại bỏ các hàng có giá trị null trong các cột số.
    # df.isnull().sum() hiển thị số lượng giá trị null còn lại sau khi xử lý.

def show_correlation_info(correlation_matrix, text_widget):
    """Hiển thị thông tin tương quan."""
    text_widget.insert(tk.END, "\nTương quan với Price ($):\n")  # Chèn tiêu đề vào widget văn bản
    text_widget.insert(tk.END, str(correlation_matrix['Price ($)'].sort_values(ascending=False)) + "\n")  # Chèn tương quan với Price ($) vào widget văn bản
    high_correlation_features = correlation_matrix['Price ($)'].sort_values(ascending=False)  # Lấy các thuộc tính có tương quan cao
    high_correlation_features = high_correlation_features[abs(high_correlation_features) > 0.5].drop('Price ($)')  # Lọc các thuộc tính có tương quan tuyệt đối > 0.5
    text_widget.insert(tk.END, "\nCác thuộc tính có tương quan cao với Price ($):\n")  # Chèn tiêu đề vào widget văn bản
    text_widget.insert(tk.END, str(high_correlation_features) + "\n")  # Chèn các thuộc tính có tương quan cao vào widget văn bản

    # Giải thích:
    # target_correlation = correlation_matrix['Price ($)'].sort_values(ascending=False) sắp xếp các thuộc tính theo độ tương quan với 'Price ($)' giảm dần.
    # high_correlation_features lọc các thuộc tính có độ tương quan tuyệt đối lớn hơn 0.5.

def analyze_data():
    """Phân tích dữ liệu và hiển thị kết quả trong giao diện tkinter."""
    df = pd.read_csv('laptop_prices.csv')  # Đọc dữ liệu từ file CSV

    text_widget.delete(1.0, tk.END)  # Xóa nội dung hiện tại của widget văn bản

    show_data_info(df, text_widget)  # Hiển thị thông tin DataFrame ban đầu

    numeric_cols = ['RAM (GB)', 'Storage', 'Screen Size (inch)', 'Battery Life (hours)', 'Weight (kg)', 'Price ($)']  # Danh sách các cột số
    for col in numeric_cols:  # Lặp qua từng cột số
        df[col] = df[col].astype(str).str.replace(r'[^\d\.]+', '', regex=True)  # Loại bỏ ký tự không phải số
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Chuyển sang kiểu số, lỗi thành NaN

    show_invalid_values(df, numeric_cols, text_widget)  # Hiển thị các giá trị không hợp lệ

    df = df.dropna(subset=numeric_cols)  # Loại bỏ các hàng có giá trị null trong các cột số
    show_null_counts(df, text_widget, "cột số")  # Hiển thị số lượng giá trị null sau khi xử lý cột số

    categorical_cols = ['Brand', 'Processor', 'GPU', 'Operating System', 'Resolution']  # Danh sách các cột phân loại
    for col in categorical_cols:  # Lặp qua từng cột phân loại
        df[col] = df[col].fillna('Unknown')  # Điền giá trị null bằng 'Unknown'
    show_null_counts(df, text_widget, "cột phân loại")  # Hiển thị số lượng giá trị null sau khi xử lý cột phân loại

    correlation_matrix = df[numeric_cols].corr()  # Tính ma trận tương quan giữa các cột số

    # Hiển thị heatmap trong một cửa sổ riêng biệt
    plt.figure(figsize=(10, 8))  # Tạo hình vẽ với kích thước 10x8
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")  # Vẽ heatmap ma trận tương quan
    plt.title('Ma trận tương quan giữa các thuộc tính số')  # Đặt tiêu đề cho biểu đồ
    plt.show()  # Hiển thị biểu đồ

    # Giải thích:
    # correlation_matrix = df[numeric_cols].corr() tính toán ma trận tương quan giữa các cột số.
    # sns.heatmap() vẽ heatmap để trực quan hóa ma trận tương quan.
    # - annot=True hiển thị giá trị tương quan trong các ô.
    # - cmap='coolwarm' sử dụng bảng màu coolwarm.
    # - fmt=".2f" định dạng giá trị tương quan thành 2 chữ số thập phân.

    show_correlation_info(correlation_matrix, text_widget)  # Hiển thị thông tin tương quan

    # Hiển thị scatter plots trong một cửa sổ riêng biệt
    high_correlation_features = correlation_matrix['Price ($)'].sort_values(ascending=False)  # Lấy các thuộc tính có tương quan cao
    high_correlation_features = high_correlation_features[abs(high_correlation_features) > 0.5].drop('Price ($)')  # Lọc các thuộc tính có tương quan tuyệt đối > 0.5
    for feature in high_correlation_features.index:  # Lặp qua từng thuộc tính có tương quan cao
        plt.figure(figsize=(8, 6))  # Tạo hình vẽ với kích thước 8x6
        sns.scatterplot(x=feature, y='Price ($)', data=df)  # Vẽ scatter plot
        plt.title(f'Scatter plot: {feature} vs Price ($)')  # Đặt tiêu đề cho biểu đồ
        plt.show()  # Hiển thị biểu đồ

    # Giải thích:
    # sns.scatterplot() vẽ biểu đồ phân tán để xem mối quan hệ giữa thuộc tính và 'Price ($)'.

    # Hiển thị boxplots trong một cửa sổ riêng biệt
    for col in categorical_cols:  # Lặp qua từng cột phân loại
        plt.figure(figsize=(10, 6))  # Tạo hình vẽ với kích thước 10x6
        sns.boxplot(x=col, y='Price ($)', data=df)  # Vẽ boxplot
        plt.title(f'Box plot: {col} vs Price ($)')  # Đặt tiêu đề cho biểu đồ
        plt.xticks(rotation=45, ha='right')  # Xoay nhãn trục x 45 độ
        plt.show()  # Hiển thị biểu đồ

    # Giải thích:
    # sns.boxplot() vẽ biểu đồ hộp để xem sự phân phối của 'Price ($)' theo từng thuộc tính phân loại.
    # plt.xticks(rotation=45, ha='right') xoay nhãn trục x 45 độ để dễ đọc.

# Tạo cửa sổ tkinter
root = tk.Tk()
root.title("Phân tích dữ liệu laptop")

# Tạo widget văn bản cuộn
text_widget = scrolledtext.ScrolledText(root, width=80, height=30)
text_widget.pack(padx=10, pady=10)

# Gọi hàm phân tích dữ liệu ngay khi chạy chương trình
analyze_data()

root.mainloop()  # Bắt đầu vòng lặp sự kiện tkinter