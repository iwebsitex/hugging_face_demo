# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Generate some example data (a simple linear relationship)
X = np.array([[1], [2], [3], [4], [5]])  # Feature (e.g., size of the house)
y = np.array([100, 200, 300, 400, 500])  # Target (e.g., price of the house)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
print("Predictions on test set:", predictions)

# Save the trained model using joblib
joblib.dump(model, "linear_regression_model.joblib")
