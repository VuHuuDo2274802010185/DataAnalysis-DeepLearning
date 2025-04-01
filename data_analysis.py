import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import scrolledtext

def show_data_info(df, text_widget):
    """Hiển thị thông tin DataFrame trong một widget văn bản."""
    text_widget.insert(tk.END, "Thông tin ban đầu của DataFrame:\n")
    text_widget.insert(tk.END, str(df.info()) + "\n")
    text_widget.insert(tk.END, "\nSố lượng giá trị null trước khi xử lý:\n")
    text_widget.insert(tk.END, str(df.isnull().sum()) + "\n")

def show_invalid_values(df, numeric_cols, text_widget):
    """Hiển thị các giá trị không hợp lệ trong các cột số."""
    for col in numeric_cols:
        invalid_values = df[pd.to_numeric(df[col], errors='coerce').isnull()]
        if not invalid_values.empty:
            text_widget.insert(tk.END, f"\nCác giá trị không hợp lệ trong cột '{col}':\n")
            text_widget.insert(tk.END, str(invalid_values[col]) + "\n")

def show_null_counts(df, text_widget, stage):
    """Hiển thị số lượng giá trị null sau khi xử lý."""
    text_widget.insert(tk.END, f"\nSố lượng giá trị null sau khi xử lý {stage}:\n")
    text_widget.insert(tk.END, str(df.isnull().sum()) + "\n")

def show_correlation_info(correlation_matrix, text_widget):
    """Hiển thị thông tin tương quan."""
    text_widget.insert(tk.END, "\nTương quan với Price ($):\n")
    text_widget.insert(tk.END, str(correlation_matrix['Price ($)'].sort_values(ascending=False)) + "\n")
    high_correlation_features = correlation_matrix['Price ($)'].sort_values(ascending=False)
    high_correlation_features = high_correlation_features[abs(high_correlation_features) > 0.5].drop('Price ($)')
    text_widget.insert(tk.END, "\nCác thuộc tính có tương quan cao với Price ($):\n")
    text_widget.insert(tk.END, str(high_correlation_features) + "\n")

def analyze_data():
    """Phân tích dữ liệu và hiển thị kết quả trong giao diện tkinter."""
    df = pd.read_csv('laptop_prices.csv')

    text_widget.delete(1.0, tk.END)  # Xóa nội dung hiện tại của widget văn bản

    show_data_info(df, text_widget)

    numeric_cols = ['RAM (GB)', 'Storage', 'Screen Size (inch)', 'Battery Life (hours)', 'Weight (kg)', 'Price ($)']
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(r'[^\d\.]+', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    show_invalid_values(df, numeric_cols, text_widget)

    df = df.dropna(subset=numeric_cols)
    show_null_counts(df, text_widget, "cột số")

    categorical_cols = ['Brand', 'Processor', 'GPU', 'Operating System', 'Resolution']
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    show_null_counts(df, text_widget, "cột phân loại")

    correlation_matrix = df[numeric_cols].corr()

    # Hiển thị heatmap trong một cửa sổ riêng biệt
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Ma trận tương quan giữa các thuộc tính số')
    plt.show()

    show_correlation_info(correlation_matrix, text_widget)

    # Hiển thị scatter plots trong một cửa sổ riêng biệt
    high_correlation_features = correlation_matrix['Price ($)'].sort_values(ascending=False)
    high_correlation_features = high_correlation_features[abs(high_correlation_features) > 0.5].drop('Price ($)')
    for feature in high_correlation_features.index:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=feature, y='Price ($)', data=df)
        plt.title(f'Scatter plot: {feature} vs Price ($)')
        plt.show()

    # Hiển thị boxplots trong một cửa sổ riêng biệt
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col, y='Price ($)', data=df)
        plt.title(f'Box plot: {col} vs Price ($)')
        plt.xticks(rotation=45, ha='right')
        plt.show()

# Tạo cửa sổ tkinter
root = tk.Tk()
root.title("Phân tích dữ liệu laptop")

# Tạo widget văn bản cuộn
text_widget = scrolledtext.ScrolledText(root, width=80, height=30)
text_widget.pack(padx=10, pady=10)

# Gọi hàm phân tích dữ liệu ngay khi chạy chương trình
analyze_data()

root.mainloop()