import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.datasets import load_digits

# Load a sample dataset (you can replace this with your own dataset)
digits = load_digits()
X = digits.data
y = digits.target

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# Save the model and scaler
with open('model/knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")