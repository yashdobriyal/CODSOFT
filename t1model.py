import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
file_path = 'path_to_your_csv_file/tested.csv'  # replace with your file path
titanic_df = pd.read_csv(file_path)

# Handle missing values
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
titanic_df['Fare'] = titanic_df['Fare'].fillna(titanic_df['Fare'].median())  # Alternative approach to avoid the warning
titanic_df.drop(columns=['Cabin'], inplace=True)

# Encode 'Sex' column
titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked' column
embarked_encoded = pd.get_dummies(titanic_df['Embarked'], prefix='Embarked')
titanic_df = pd.concat([titanic_df, embarked_encoded], axis=1)
titanic_df.drop(columns=['Embarked'], inplace=True)

# Select relevant features for the model
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
X = titanic_df[features]
y = titanic_df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived'])

# Print the results with descriptive labels
print(f'Accuracy: {accuracy}\n')
print('Confusion Matrix:')
print(pd.DataFrame(conf_matrix, index=['Actual Not Survived', 'Actual Survived'], columns=['Predicted Not Survived', 'Predicted Survived']))
print('\nClassification Report:')
print(class_report)
