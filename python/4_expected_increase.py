import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r"D:\Documents\DataAnalysis\Housing_Market_France\csv_files\housing_market_france_processed.csv")

# Separate features and target variable
X = df[['year', 'department']]
y = df['average_price_per_year_department']

# Encode departments using one-hot encoding
X_encoded = pd.get_dummies(X, columns=['department'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions for 2023

# First, create a dataframe with the year 2023 and all departments
departments_2023 = pd.DataFrame({'year': [2023] * len(df['department'].unique()),
                                 'department': df['department'].unique()})

# Encode departments using one-hot encoding
departments_2023_encoded = pd.get_dummies(departments_2023, columns=['department'], drop_first=True)

# Predict average prices for 2023
predictions_2023 = model.predict(departments_2023_encoded)

# Add predicted average prices to the departments dataframe
departments_2023['predicted_average_price'] = predictions_2023

# Calculate the increase in average price compared to 2022
data_2022 = df[df['year'] == 2022][['department', 'average_price_per_year_department']]
departments_2023 = pd.merge(departments_2023, data_2022, on='department', how='left')
departments_2023['price_increase'] = departments_2023['predicted_average_price'] - departments_2023['average_price_per_year_department']

# Sort departments by price increase
departments_2023_sorted = departments_2023.sort_values(by='price_increase', ascending=False)

# Display the departments with the biggest increase in average price
print("Top 10 departments with the biggest increase in average price for 2023:")
print(departments_2023_sorted.head(10))

# Verification: Check if the model's predictions are reasonable

# Calculate root mean squared error (RMSE) on the test set
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\nRoot Mean Squared Error (RMSE) on test set:", rmse)

# Check the coefficient of determination (R^2) on the test set
root_mean_squared_error = model.score(X_test, y_test)
print("Coefficient of determination (R^2) on test set:", root_mean_squared_error)