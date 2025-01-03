import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data = pd.read_csv('C:\workspace\python-project\lamtron\Dữ-liệu-khử-nhiễu.csv')  # Thay bằng đường dẫn file thực tế
# thuộc tính đánh giá 
continuous_columns = [ 'Visibility (km)']
# mô hình đánh giá sự tương quan giữa thuộc tính với nhãn mục tiêu (Weather Type)
plt.figure(figsize=(20, 15))
for i, column in enumerate(continuous_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='Weather Type', y=column, data=data, palette="Set2")
    plt.title(f"{column} vs Weather Type")
    plt.xlabel("Weather Type")
    plt.ylabel(column)
plt.tight_layout()
plt.show()
