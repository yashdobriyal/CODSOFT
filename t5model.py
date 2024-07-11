import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
file_path = 'IMDb Movies India.csv'
data = pd.read_csv(file_path, encoding='latin1')

# Selecting relevant features
data = data[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Rating']]

# Drop rows with missing values
data.dropna(inplace=True)

# Preprocessing and feature engineering
# OneHotEncoding categorical features
categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ])

# Define the model pipeline for Gradient Boosting Regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))  # Using default parameters
])

# Separate features and target
X = data.drop('Rating', axis=1)
y = data['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print("\nCross-Validation Scores (MSE) - Gradient Boosting Regressor:", -cv_scores)
print("Average Cross-Validation Score (MSE) - Gradient Boosting Regressor:", -cv_scores.mean())

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nGradient Boosting Regressor Model Performance:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Display first few predictions vs actual ratings
predictions_vs_actual = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
print("\nFirst few predictions vs actual ratings:")
print(predictions_vs_actual.head())
