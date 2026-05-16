import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("sales_data.csv")

# Features and target
X = df[["Day"]]
y = df["Sales"]

# Create model
model = LinearRegression()
model.fit(X, y)

# Predict future
future_days = [[11], [12], [13]]
predictions = model.predict(future_days)

print("Future Predictions:", predictions)

# Plot
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Day")
plt.ylabel("Sales")
plt.title("Sales Prediction")
plt.show()