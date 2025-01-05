import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dữ liệu
file_path = "D:/ML/BTL/Phan-cum-kmeans1_reduced.csv"
df = pd.read_csv(file_path)

# Encode các cột phân loại
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Tách dữ liệu thành X và y
X = df.drop(columns=['Weather Type'])
y = df['Weather Type']

# Tách dữ liệu thành tập huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hiển thị kích thước
print("Số dòng huấn luyện:", X_train.shape[0])
print("Số dòng kiểm thử:", X_test.shape[0])

# Xây dựng mô hình cây quyết định với thuật toán ID3 (sử dụng entropy)
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(criterion='entropy')  # ID3 sử dụng entropy để tính gain
    model.fit(X_train, y_train)
    return model

# Đánh giá mô hình
def evaluate_model(model, X_test, y_test):
    # Dự đoán nhãn của tập test
    y_pred = model.predict(X_test)

    # Tính toán độ chính xác
    accuracy = accuracy_score(y_test, y_pred)

    # In ra các chỉ số đánh giá khác
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Huấn luyện mô hình
model = train_decision_tree(X_train, y_train)

# Đánh giá mô hình
evaluate_model(model, X_test, y_test)

# Vẽ cây quyết định
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=label_encoders['Weather Type'].classes_,
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.show()
