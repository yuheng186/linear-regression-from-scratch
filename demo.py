"""
Demo script to test the custom Linear Regression model.

- Loads a salary dataset
- Trains the LinearRegression model
- Evaluates performance (MSE, MAE, R2)
- Visualizes the results
"""
from linear_regression_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Salary_dataset.csv')

# EDA
shape = df.shape
print(f'Dataset shape: {shape}')

# Print the first 5 rows
print(df.head())

# Check for missing values
print(f'Missing values:\n{df.isnull().sum()}')

# Split dataset into x and y
x = df['YearsExperience'].values.reshape(-1, 1)
y = df['Salary'].values

# Split into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model 
LR = LinearRegression(learning_rate=0.01, epochs=3000)

# Training
LR.fit(x_train, y_train)

# Prediction
y_pred = LR.predict(x_test)

# Evaluation
mse = LR.mse(y_test, y_pred)
mae = LR.mae(y_test, y_pred)
r2 = LR.r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')
print(f'R2 Score: {r2:.4f}')

# Visualisation
plt.scatter(x_test, y_test, color='blue', label='Actual')
plt.scatter(x_test, y_pred, color='red', label='Predicted')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Actual vs Predicted Salary')
plt.plot(x_test, y_pred, color='green', linewidth=2, label='Regression Line')
plt.legend()
plt.show()
