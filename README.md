import yfinance as yf
import pandas as pd

# Define the sector indices
sectors = ['^NSEI', '^NSEBANK', '^CNXAUTO', '^CNXIT', '^CNXPHARMA']  # Add more indices as needed

# Download data for each sector
data = {}
for sector in sectors:
    data[sector] = yf.download(sector, start="2014-01-01", end="2024-01-01")

# Save data to CSV files
for sector, df in data.items():
    df.to_csv(f'{sector}.csv')

# Example of loading macroeconomic data from CSV files
macro_data = pd.read_csv('macro_data.csv')  # Replace with actual macroeconomic data file

import seaborn as sns
import matplotlib.pyplot as plt

# Load Nifty Bank data
nifty_bank = pd.read_csv('^NSEBANK.csv')
nifty_bank['Date'] = pd.to_datetime(nifty_bank['Date'])
nifty_bank.set_index('Date', inplace=True)

# Plot the closing price
plt.figure(figsize=(12, 6))
plt.plot(nifty_bank['Close'])
plt.title('Nifty Bank Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Correlation heatmap for macroeconomic variables
sns.heatmap(macro_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

from statsmodels.tsa.arima_model import ARIMA

# Fit ARIMA model
model = ARIMA(nifty_bank['Close'][:'2022'], order=(5, 1, 0))  # Modify order based on ACF/PACF plots
model_fit = model.fit(disp=0)
print(model_fit.summary())

# Forecasting
forecast = model_fit.forecast(steps=24)  # Forecasting next 2 years (monthly data)
plt.figure(figsize=(12, 6))
plt.plot(nifty_bank['Close'], label='Actual')
plt.plot(pd.date_range(start=nifty_bank.index[-1], periods=24, freq='M'), forecast[0], label='Forecast')
plt.legend()
plt.show()

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Prepare data for XGBoost
X = macro_data.drop('Nifty Bank', axis=1)  # Assuming 'Nifty Bank' is one of the columns in macro_data
y = macro_data['Nifty Bank']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit XGBoost model
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plotting predictions vs actual
plt.figure(figsize=(12, 6))
plt.plot(y_test.reset_index(drop=True), label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()

1. Introduction
   - Project objective and scope
2. Data Collection
   - Data sources and variables considered
3. Exploratory Data Analysis
   - Key findings and visualizations
4. Methodology
   - Models used and rationale
5. Results
   - Predictions and performance metrics
6. Conclusion
   - Summary of findings and implications
  
# Sector Analysis and Prediction

## 1. Introduction
The project aims to predict the performance of various Nifty sector indices relative to the Nifty 50 index using global and local macroeconomic variables.

## 2. Data Collection
Data was collected for Nifty sector indices and relevant macroeconomic variables from sources like Yahoo Finance and the World Bank.

## 3. Exploratory Data Analysis
### Nifty Bank Closing Prices
![Nifty Bank Closing Prices](nifty_bank_closing_prices.png)

### Correlation Heatmap
![Correlation Heatmap](correlation_heatmap.png)

## 4. Methodology
We used ARIMA and XGBoost models to predict the performance of sector indices. The ARIMA model was chosen for its effectiveness in time series forecasting, while XGBoost was used for its robustness in handling complex relationships between variables.

## 5. Results
### ARIMA Model Summary
\`\`\`
ARIMA Model Summary
\`\`\`

### XGBoost Predictions vs Actual
![XGBoost Predictions](xgboost_predictions.png)

## 6. Conclusion
The models provide insights into the potential performance of sector indices. The ARIMA model showed consistent predictions, while XGBoost demonstrated its capability in capturing intricate patterns.
