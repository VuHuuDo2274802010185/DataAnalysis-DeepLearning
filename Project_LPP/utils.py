import matplotlib.pyplot as plt # Import thư viện matplotlib để vẽ đồ thị

def plot_loss(history):
    """Vẽ đồ thị loss trong quá trình huấn luyện."""
    plt.figure(figsize=(10, 6)) # Tạo figure
    plt.plot(history.history['loss'], label='Training Loss') # Vẽ đồ thị loss trên tập huấn luyện
    plt.plot(history.history['val_loss'], label='Validation Loss') # Vẽ đồ thị loss trên tập validation
    plt.xlabel("Epochs") # Đặt nhãn trục x
    plt.ylabel("Loss") # Đặt nhãn trục y
    plt.title("Training and Validation Loss") # Đặt tiêu đề
    plt.legend() # Hiển thị legend
    plt.show() # Hiển thị đồ thị

def plot_predictions(y_test, y_pred):
    """Vẽ đồ thị dự đoán so với thực tế."""
    plt.figure(figsize=(10, 6)) # Tạo figure
    plt.scatter(y_test, y_pred) # Vẽ scatter plot của giá thực tế và giá dự đoán
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Vẽ đường y=x
    plt.xlabel("Giá thực tế") # Đặt nhãn trục x
    plt.ylabel("Giá dự đoán") # Đặt nhãn trục y
    plt.title("Giá thực tế vs. Giá dự đoán") # Đặt tiêu đề
    plt.show() # Hiển thị đồ thị