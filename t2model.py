import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'advertising.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display information about the dataset
print("\nDataset info:")
print(data.info())

# Display basic statistics of the dataset
print("\nDataset statistics:")
print(data.describe())

# Preprocess the data (if necessary)
# Assuming the dataset has columns 'TV', 'Radio', 'Newspaper', and 'Sales'

# Split the data into features and target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot the true vs predicted sales
plt.scatter(y_test, y_pred, label='Predicted Sales')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

# Generate the next 10 sets of values for predictions
new_data = pd.DataFrame({
    'TV': np.linspace(0, 300, 10),
    'Radio': np.linspace(0, 50, 10),
    'Newspaper': np.linspace(0, 100, 10)
})

# Make predictions using the trained model
new_predictions = model.predict(new_data)

# Display the predictions with red dots on the graph
plt.scatter(new_predictions, new_predictions, color='red', label='New Predictions', marker='o')

plt.legend()
plt.show()

# Print the new predictions
print("Predictions for the next 10 sets of values:")
print(new_predictions)
