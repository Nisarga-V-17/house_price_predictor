# House Price Predictor using Linear Regression

from sklearn.linear_model import LinearRegression
import numpy as np

# Training data (House size in sq.ft vs Price)
size = np.array([[500], [800], [1000], [1200], [1500]])
price = np.array([100000, 150000, 200000, 240000, 300000])

# Create model
model = LinearRegression()

# Train model
model.fit(size, price)

# Predict new price
new_size = np.array([[1100]])
predicted_price = model.predict(new_size)

print("Predicted price:", predicted_price[0])