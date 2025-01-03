import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Đọc file dữ liệu
file_path = 'c:\workspace\python-project\lamtron\Dữ-liệu-khử-nhiễu.csv'  # Đường dẫn tới file của bạn
data = pd.read_csv(file_path)

# 2. Kiểm tra dữ liệu
print(data.head())

# 3. Chọn các cột đặc trưng cần dùng
features = data[['Wind Speed']]  # Thay column1, column2 bằng tên cột của bạn

# 4. Chuẩn hóa dữ liệu (khuyến khích)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 5. Tính WCSS cho các giá trị k khác nhau
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# 6. Vẽ đồ thị Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()
