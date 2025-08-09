"""
Linear Regression Model

This implements a simple linear regression model from scratch using NumPy.
It includes:
- Calculation of loss functions (MSE, MAE, R2 Score)
- Training the model via gradient descent
- Making predictions

It is designed for demonstration purposes to understand how linear regression works under the hood.
"""
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.0001, epochs=1000):
        """
        Initialize the Linear Regression model with hyperparameters.

        Parameters:
        learning_rate (float): Learning rate of gradient descent.
        epochs (int): Number of iterations for training.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None 
        self.bias = None 

    def mse(self, y, y_pred):
        """
        Calculate Mean Squared Error (MSE).

        Parameters:
        y (array-like): True values.
        y_pred (array-like): Predicted values.

        Returns:
        float: MSE value.
        """
        return np.mean((y - y_pred) ** 2)

    def mae(self, y, y_pred):
        """
        Calculate Mean Absolute Error (MAE).

        Parameters:
        y (array-like): True values.
        y_pred (array-like): Predicted values.

        Returns:
        float: MAE value.
        """
        return np.mean(np.abs(y - y_pred))

    def r2_score(self, y, y_pred):
        """
        Calculate R-Squared (R2 Score).

        Parameters:
        y (array-like): True values.
        y_pred (array-like): Predicted values.

        Returns:
        float: R² score.
        """
        ss_residual = np.sum((y - y_pred) ** 2) 
        ss_total = np.sum((y - np.mean(y)) ** 2)       
        return 1 - (ss_residual/ss_total) 

    def fit(self, x_train, y_train):
        """
        Train the Linear Regression model using gradient descent.

        Parameters:
        x_train (array): Training features.
        y_train (array): Training target values.

        Returns:
        tuple: Final weights and bias after training.
        """
        n_observations, n_features = x_train.shape
        self.weights = np.zeros(n_features) # Initialize weights for all features
        self.bias = 0

        # Model: y_pred = w·x + b
        # Error: y - y_pred = y - (w·x + b)
        # Loss: MSE = (1/n) * Σ(y - (w·x + b))²
        # Update weights and bias in every epoch using gradient descent
        for _ in range(self.epochs):
            dw = np.zeros(n_features)
            db = 0

            y_pred = np.dot(x_train, self.weights) + self.bias

            error = y_train - y_pred

            dw = (-2/n_observations) * x_train.T.dot(error) 
            db = (-2/n_observations) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, x_test):
        """
        Predict target values using trained model.
        
        Parameters:
        x_test (array): Input for prediction.

        Returns:
        array: Predicted values.
        """
        return np.dot(x_test, self.weights) + self.bias 
