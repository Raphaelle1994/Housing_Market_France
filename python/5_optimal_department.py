# Optimal department : What are the three departments with an average house price below the national average expected to see the biggest increase in house price? 

#We'll use all our previous code and logic to find departments that meet all our critera:
  #-  are below the national average (with a bonus point if they are among the 5 least expensive departments)
  #-  have seen a constant increase (from 2020 to 2021 and 2021 to 2022)
  #-  are expected to have a bigger increase than average by our model 


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r"D:\Documents\DataAnalysis\Housing_Market_France\csv_files\housing_market_france_processed.csv")

# First we will find all the departments that are below the national mean 

# Calculate the average price across all departments and years
average_price = df["average_price_per_year_department"].astype(float).mean().round(2)
print("Average price for all departments in France regardless of year:", average_price)

# Filter departments below the average price
below_average_departments = df[df["average_price_per_year_department"].astype(float) < average_price]

# Group by department and calculate the average price for each
average_prices_by_department = below_average_departments.groupby("department")["average_price_per_year_department"].mean().round(2)
sorted_departments = average_prices_by_department.sort_values(ascending=False)

# Print department names and their average prices
print("Departments with average price below the overall average price:")
for department, avg_price in sorted_departments.items():
    print(f"{department}: {avg_price}")

# From these departments, figure out which ones have seen a constant increase (2018 to 2022)

# Calculate the average price change year to year for each department

average_price_change_2018_to_2019 = df[df['year'] == 2019].groupby('department')['average_price_per_year_department'].mean() - df[df['year'] == 2018].groupby('department')['average_price_per_year_department'].mean()
average_price_change_2019_to_2020 = df[df['year'] == 2020].groupby('department')['average_price_per_year_department'].mean() - df[df['year'] == 2019].groupby('department')['average_price_per_year_department'].mean()
average_price_change_2020_to_2021 = df[df['year'] == 2021].groupby('department')['average_price_per_year_department'].mean() - df[df['year'] == 2020].groupby('department')['average_price_per_year_department'].mean()
average_price_change_2021_to_2022 = df[df['year'] == 2022].groupby('department')['average_price_per_year_department'].mean() - df[df['year'] == 2021].groupby('department')['average_price_per_year_department'].mean()

# Combine the results into a single dataframe
average_price_change = pd.concat([average_price_change_2018_to_2019, average_price_change_2019_to_2020, average_price_change_2020_to_2021, average_price_change_2021_to_2022], axis=1)
average_price_change.columns = ['average_price_change_2018_to_2019','average_price_change_2019_to_2020','average_price_change_2020_to_2021', 'average_price_change_2021_to_2022']

# Reset index to ensure 'department' becomes a regular column
average_price_change.reset_index(inplace=True)

# Filter departments where prices didn't increase for any given year
decreased_departments_2018_to_2019 = average_price_change[average_price_change['average_price_change_2018_to_2019'] <= 0]['department']
decreased_departments_2019_to_2020 = average_price_change[average_price_change['average_price_change_2019_to_2020'] <= 0]['department']
decreased_departments_2020_to_2021 = average_price_change[average_price_change['average_price_change_2020_to_2021'] <= 0]['department']
decreased_departments_2021_to_2022 = average_price_change[average_price_change['average_price_change_2021_to_2022'] <= 0]['department']

# Combine the departments where prices didn't increase from either period
decreased_departments = decreased_departments_2018_to_2019.tolist() + decreased_departments_2020_to_2021.tolist() + decreased_departments_2020_to_2021.tolist() + decreased_departments_2021_to_2022.tolist() 

# Filter the decreased departments from sorted_departments
sorted_decreased_departments_2018_to_2019 = decreased_departments_2018_to_2019[decreased_departments_2018_to_2019.isin(sorted_departments.index)]
sorted_decreased_departments_2019_to_2020 = decreased_departments_2019_to_2020[decreased_departments_2019_to_2020.isin(sorted_departments.index)]
sorted_decreased_departments_2020_to_2021 = decreased_departments_2020_to_2021[decreased_departments_2020_to_2021.isin(sorted_departments.index)]
sorted_decreased_departments_2021_to_2022 = decreased_departments_2021_to_2022[decreased_departments_2021_to_2022.isin(sorted_departments.index)]

# Combine the departments where prices didn't increase from any period
sorted_decreased_departments = (
    sorted_decreased_departments_2018_to_2019.tolist() +
    sorted_decreased_departments_2019_to_2020.tolist() +
    sorted_decreased_departments_2020_to_2021.tolist() +
    sorted_decreased_departments_2021_to_2022.tolist()
)

# Check if there are any departments where prices didn't increase
if not sorted_decreased_departments:
    print("Among departments below the national average, housing prices increased in all French departments from 2020 to 2021 or from 2021 to 2022.")
else:
    print("Among departments below the national average, housing prices did not increase in the following departments during either period.")
    print(sorted_decreased_departments)
    for department in sorted_decreased_departments:
        print(f"\nPrices in {department}:")
        department_data = df[df['department'] == department]
        print(department_data[['year', 'average_price_per_year_department']])
    print("Among departments below the national average, housing prices increased consistently in:")
    print(sorted_departments[~sorted_departments.index.isin(sorted_decreased_departments)])
    for department in sorted_departments[~sorted_departments.index.isin(sorted_decreased_departments)].index:
        print(f"\nPrices in {department}:")
        department_data = df[df['department'] == department]
        print(department_data[['year', 'average_price_per_year_department']])

# We now have filtered our results and removed all departments whose price is above the national average, as well as the departrments that didn't see a consistent increase in price. 
# We can add one more optional condition and only take interest into departments who have been predicted to increase the most according to our previosu model.

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

# Get the intersection of departments from sorted_departments and departments_2023_sorted.head(10)
departments_of_interest = sorted_departments.index.intersection(departments_2023_sorted.head(10)['department'])                                  

# Filter out the departments that are also not in sorted_decreased_departments
departments_of_interest = [dep for dep in departments_of_interest if dep not in sorted_decreased_departments]                                           

if not departments_2023_sorted[departments_2023_sorted['department'].isin(departments_of_interest)].empty:
    print("The departments below the national average, with constant increase since 2018 and predicted to be in the top 10 of increase in 2023 are:")
    print(departments_2023_sorted[departments_2023_sorted['department'].isin(departments_of_interest)].head(10))
else:
    print("No departments meet all the criteria.")