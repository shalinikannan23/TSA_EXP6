## DEVELOPED BY: K SHALINI
## REGISTER NO: 212222240095
## DATE:


# Ex.No: 6               HOLT WINTERS METHOD

### AIM:
To forecast sales using the Holt-Winters method and calculate the Test and Final Predictions.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:


```python
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'coin_Bitcoin.csv'  # Path to the uploaded dataset
data = pd.read_csv(file_path)

# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Group data by date and resample it to month-end frequency ('ME') based on the 'Close' price
monthly_data = data.resample('ME', on='Date').sum()

# Plot the time series data (using 'Close' prices)
plt.figure(figsize=(10, 5))
plt.plot(monthly_data['Close'], label='Monthly Close Prices')
plt.title('Monthly Bitcoin Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Split data into training and testing sets (80% for training, 20% for testing)
train_size = int(len(monthly_data) * 0.8)
train, test = monthly_data['Close'][:train_size], monthly_data['Close'][train_size:]

# Fit the Holt-Winters model on training data
model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12)
fit = model.fit()

# Make predictions on the test set
predictions = fit.forecast(len(test))

# Calculate RMSE for the test set predictions
rmse = np.sqrt(mean_squared_error(test, predictions))
print(f'Test RMSE: {rmse}')

# Fit Holt-Winters model on the entire dataset for future forecasting
final_model = ExponentialSmoothing(monthly_data['Close'], trend="add", seasonal="add", seasonal_periods=12)
final_fit = final_model.fit()

# Make future predictions (for 12 months)
future_steps = 12
final_forecast = final_fit.forecast(steps=future_steps)

# Plotting Test Predictions and Final Predictions
plt.figure(figsize=(12, 6))

# Plot Test Predictions
plt.subplot(1, 2, 1)
plt.plot(monthly_data.index[:train_size], train, label='Training Data', color='blue')
plt.plot(monthly_data.index[train_size:], test, label='Test Data', color='green')
plt.plot(monthly_data.index[train_size:], predictions, label='Predictions', color='orange')
plt.title('Test Predictions')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Plot Final Predictions
plt.subplot(1, 2, 2)
plt.plot(monthly_data.index, monthly_data['Close'], label='Original Close Prices', color='blue')
# Plot future forecast (use 'ME' frequency)
plt.plot(pd.date_range(start=monthly_data.index[-1] + pd.DateOffset(months=1), periods=future_steps, freq='ME'), 
         final_forecast, label='Final Forecast', color='orange')
plt.title('Final Predictions')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

plt.tight_layout()
plt.show()

```

### OUTPUT:

TEST AND FINAL PREDICTION:

<img heigth=10% width=50% src="https://github.com/user-attachments/assets/d9dcc7cb-1227-42c3-b444-730c6c8a7627">

### RESULT:
Thus, the program to forecast sales using the Holt-Winters method and calculate the Test and Final Prediction is executed successfully.
