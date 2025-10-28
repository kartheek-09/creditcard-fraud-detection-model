
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv('/content/drive/MyDrive/fraudTest.csv.zip')
df
df.info()
df.describe().T
df.isnull()
df.isnull().sum()

# Handle date/time column
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_hour'] = df['trans_date_trans_time'].dt.hour
df['trans_dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
df = df.drop(columns=['trans_date_trans_time'])

# Select features and target
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

print(X)
print(y)

categorical_features = X.select_dtypes(include=['object']).columns


for feature in categorical_features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature].astype(str))
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

print(X_train)
print(X_test)
scaler = StandardScaler()
numerical_features = X.select_dtypes(include=['number']).columns

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Initialize and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy:{accuracy}")