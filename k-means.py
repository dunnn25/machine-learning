import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# Đọc dữ liệu từ file CSV
data = pd.read_csv(r'C:\workspace\python-project\lamtron\Dữ-liệu-khử-nhiễu.csv')  # Thay bằng đường dẫn file thực tế

def kmeans_discretize(data, column, n_clusters, cluster_names=None):
    """
    Rời rạc hóa một cột bằng k-means và in thông tin cụm.
    
    Parameters:
    - data: DataFrame gốc.
    - column: Tên cột cần rời rạc hóa.
    - n_clusters: Số cụm k-means.
    - cluster_names: Danh sách tên cụm (nếu có). Độ dài phải bằng số cụm.
    
    Returns:
    - Series chứa các nhãn cụm được gán tên.
    """
    if cluster_names is None:
        cluster_names = [f"Cluster {i}" for i in range(n_clusters)]
    elif len(cluster_names) != n_clusters:
        raise ValueError("Số lượng tên cụm phải bằng số cụm (n_clusters).")
    
    values = data[[column]].dropna().values  # Lấy giá trị không NaN từ cột
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(values)
    labels = kmeans.predict(values)
    centroids = kmeans.cluster_centers_.flatten()
    
    # Sắp xếp cụm theo giá trị trung tâm
    sorted_centroids = np.argsort(centroids)
    label_mapping = {old_label: cluster_names[new_label] for new_label, old_label in enumerate(sorted_centroids)}
    
    # Gán nhãn cụm đã sắp xếp
    sorted_labels = pd.Series(index=data.index, data=None)
    sorted_labels.loc[data[[column]].dropna().index] = [label_mapping[label] for label in labels]
    
    # In thông tin cụm
    print(f"\nCột '{column}' - Trung tâm cụm (sorted):")
    for i, center in enumerate(centroids[sorted_centroids]):
        print(f"{cluster_names[i]}: Trung tâm = {center:.2f}")
    
    # In giá trị lớn nhất và nhỏ nhất trong từng cụm
    print(f"\nGiá trị lớn nhất và nhỏ nhất trong từng cụm của cột '{column}':")
    for i, cluster in enumerate(sorted_centroids):
        cluster_data = data.loc[sorted_labels[sorted_labels == cluster_names[i]].index, column]
        min_value = cluster_data.min()
        max_value = cluster_data.max()
        print(f"{cluster_names[i]}: Giá trị nhỏ nhất = {min_value:.2f}, Giá trị lớn nhất = {max_value:.2f}")
    
    return sorted_labels

def discretize_data(data):
    discretized_data = data.copy()
    
    # Rời rạc hóa Temperature bằng k-means với 4 cụm
    discretized_data['Temperature'] = kmeans_discretize(
        discretized_data, 
        'Temperature', 
        n_clusters=4, 
        cluster_names=['Very Low', 'Low', 'High', 'Very High']
    )
    
    # Rời rạc hóa Precipitation (%) bằng k-means với 4 cụm
    discretized_data['Precipitation (%)'] = kmeans_discretize(
        discretized_data, 
        'Precipitation (%)', 
        n_clusters=4, 
        cluster_names=['Dry', 'Light Rain', 'Moderate Rain', 'Heavy Rain']
    )
    
    # Rời rạc hoá Humidity bằng k-means với 4 cụm
    discretized_data['Humidity'] = kmeans_discretize(
        discretized_data, 
        'Humidity', 
        n_clusters=4, 
        cluster_names=['Very Low', 'Low', 'High', 'Very High']
    )
    
    # Rời rạc hoá Atmospheric Pressure bằng k-means với 4 cụm
    discretized_data['Atmospheric Pressure'] = kmeans_discretize(
        discretized_data, 
        'Atmospheric Pressure', 
        n_clusters=4, 
        cluster_names=['Low Pressure', 'Normal Pressure', 'High Pressure', 'Very High Pressure']
    )
    
    # Rời rạc hoá UV Index bằng k-means với 4 cụm
    discretized_data['UV Index'] = kmeans_discretize(
        discretized_data, 
        'UV Index', 
        n_clusters=4, 
        cluster_names=['Low', 'Moderate', 'High', 'Extreme']
    )
    
    return discretized_data

# Áp dụng rời rạc hóa
discretized_data = discretize_data(data)

# Hiển thị vài dòng đầu của dữ liệu đã rời rạc hóa
print("\nDữ liệu đã rời rạc hóa:")
print(discretized_data.head())

# Lưu dữ liệu đã rời rạc hóa vào file mới
# discretized_data.to_csv('du_lieu_roi_rac3.csv', index=False)
# print("Dữ liệu đã lưu vào file 'du_lieu_roi_rac3.csv'")
