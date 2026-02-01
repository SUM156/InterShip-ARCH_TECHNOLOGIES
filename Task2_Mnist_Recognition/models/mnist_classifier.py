import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. Load MNIST Data
print("Fetching MNIST data... (this might take a moment)")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]

# 2. Preprocessing: Normalize pixel values (0-255) to (0-1)
X = X / 255.0

# 3. Split Data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Save the Model
if not os.path.exists('models'):
    os.makedirs('models')
joblib.dump(model, 'models/mnist_digit_model.pkl')
print("\nSuccess: Model saved in 'models/' folder.")

# 7. Show a sample prediction
plt.imshow(X_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted Digit: {y_pred[0]}")
plt.axis('off')
plt.show()