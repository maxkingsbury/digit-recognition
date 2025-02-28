import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
import joblib
from sklearn.utils import shuffle

# Load MNIST data
mnist = fetch_openml('mnist_784')

# Get the data and labels as numpy arrays
X = mnist.data.values
y = mnist.target.values

# Resize images to 16x16 using numpy
X_resized = np.array([image.reshape(28, 28)[0:16, 0:16].flatten() for image in X])

# Shuffle data
X_resized, y = shuffle(X_resized, y)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_resized, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(knn_model, 'model/knn_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

accuracy = knn_model.score(X_test, y_test)
print(f'Accuracy of KNN model: {accuracy * 100}')

print("Model and scaler have been saved!")

# After training the KNN model, print sample predictions
y_pred = knn_model.predict(X_test_scaled)
print("Sample predictions:", y_pred[:20])
print("Actual labels:     ", y_test[:20])