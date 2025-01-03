import pandas as pd

# Đọc file CSV vào DataFrame
file_path = 'C:\\Users\\nguye\\Downloads\\dữ-liệu-machine-learning.csv'
data = pd.read_csv(file_path)

data_filtered = data[(data['Temperature'] >= -20) & (data['Temperature'] <= 50) & (data['Humidity'] <= 100) & (data['Precipitation (%)'] <= 100) & (data['Atmospheric Pressure'] <= 1050) & (data['Atmospheric Pressure'] >= 950) & (data['UV Index'] <= 11)]

# Lưu lại kết quả vào file CSV mới
data_filtered.to_csv('C:\\Users\\nguye\\Downloads\\Dữ-liệu-khử-nhiễu.csv', index=False)

# In ra dữ liệu đã được lọc
print(data_filtered)



