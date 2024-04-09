# Housing Market Trends in France 2018-2022

## Introduction

### Objective:
🏠 The objective of this project is to analyze housing market trends in France and build a predictive model to estimate housing prices based on various features. This analysis aims to assist potential buyers, sellers, and real estate investors in understanding market dynamics in different regions of France to make informed decisions.

### Questions:
This project aims to answer the following questions:
1. What are the least and most expensive French departments regarding the housing market?
2. What is the national median and average regarding house prices?
3. Are housing prices expected to increase in all French departments?
4. What departments are expected to see the biggest increase in their housing prices?
5. What are the three departments with an average house price below the national average expected to see the biggest increase in house price?

### Tools Used:
For this project, the following key tools were utilized:
- **SQL:** For data processing and cleaning, extracting relevant information from a large dataset.
- **PostgreSQL:** Used as a database management system.
- **Visual Studio Code:** Primary tool for both database management and use of SQL and Python.
- **Python:** Used for Exploratory Data Analysis, visualization, sharing insights gathered during the analysis, and implementing regression algorithms to predict housing prices.
- **Git & GitHub:** For collaborative sharing and project tracking.

## Project Execution

### Data Collection
Data used in this project was sourced from [Data.gouv](https://www.data.gouv.fr/fr/datasets/statistiques-dvf/), containing information on the price per square meter of houses in France. The dataset was chosen due to its relevance, credibility, cleanliness, and public availability.

### Data Cleaning using SQL
🔗[Link to SQL Query](https://github.com/Raphaelle1994/Housing_Market_France/blob/main/sql_queries/1_cleaning_data)

The SQL queries were used to clean the dataset by addressing issues such as NULL values and inconsistencies. The query ensured that only relevant data was extracted for analysis. 

``` sql
SELECT 
    SUBSTRING(year_month, 1, 4) AS year,
    department,
    ROUND(AVG(average_price_house), 2) AS average_price_per_year_department,
    SUM(amount_houses_sold) AS total_houses_sold
FROM (
      SELECT 
        department,
        year_month,
        amount_houses_sold,
        average_price_house
      FROM (
            SELECT 
              libelle_geo AS department,
              annee_mois AS year_month,
              nb_ventes_maison as amount_houses_sold,
              moy_prix_m2_maison as average_price_house
            FROM 
              housing_facts_france
            WHERE 
              moy_prix_m2_maison IS NOT NULL
              AND echelle_geo = 'departement'
              )
              )
WHERE 
    SUBSTRING(year_month, 1, 4) BETWEEN '2018' AND '2022'
GROUP BY 
    SUBSTRING(year_month, 1, 4),
    department
ORDER BY 
    year, 
    department
```

### Exploring the Data and Creating Visualizations using Python
🔗[Link to Python](https://github.com/Raphaelle1994/Housing_Market_France/tree/main/python)

Exploratory Data Analysis (EDA) was performed using Python, focusing on identifying the most and least expensive departments regardless of the year and calculating national median and average prices. Visualizations such as bar plots were created to present the findings effectively. 

### Finding If the Housing Price Increased in France from 2018 to 2022
Statistical analysis was conducted to determine if housing prices increased in all French departments from 2018 to 2022. The analysis included identifying departments with consistent price increases and visualizing the percentage change in housing prices for selected departments.

- Find and visualise the least and most expensive French departments 

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:\Documents\DataAnalysis\Housing_Market_France\csv_files\housing_market_france_processed.csv")

# Finding the most and least expensive departments regardless of the year - Grouping by department and calculating the mean price
department_avg_price = df.groupby('department')['average_price_per_year_department'].mean()

# Step one, finding the 5 most expensive departments : 

# Sort the departments by average price in descending order
sorted_departments_desc = department_avg_price.sort_values(ascending=False)

# Get the top 5 most expensive departments
top_5_most_expensive = sorted_departments_desc.head(5).round(2)

print("Top 5 most expensive departments:")
print(top_5_most_expensive)

# Step two, finding the 5 least expensive departments : 

# Sorting the departments by average price in ascending order
sorted_departments_asc = department_avg_price.sort_values(ascending=True)

# Get the top 5 most expensive departments
top_5_least_expensive = sorted_departments_asc.head(5).round(2)

print("Top 5 least expensive departments:")
print(top_5_least_expensive)


# Creating bar plots

# Convert the top 5 most and least expensive to DataFrame
top_5_most_expensive_df = top_5_most_expensive.reset_index()
top_5_least_expensive_df = top_5_least_expensive.reset_index()

# Create a bar plot for top 5 most expensive departments
plt.figure(figsize=(10, 6))
sns.barplot(x='average_price_per_year_department', y='department', data=top_5_most_expensive_df)
plt.title('Top 5 Most Expensive Departments')
plt.xlabel('Average Price per square meter')
plt.ylabel('Department')
plt.tight_layout()
plt.show()

# Create a bar plot for top 5 least expensive departments
plt.figure(figsize=(10, 6))
sns.barplot(x='average_price_per_year_department', y='department', data=top_5_least_expensive_df)
plt.title('Top 5 Least Expensive Departments')
plt.xlabel('Average Price per square meter')
plt.ylabel('Department')
plt.tight_layout()
plt.show()
```

- Find the median, mean and the departments furthest from mean

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:\Documents\DataAnalysis\Housing_Market_France\csv_files\housing_market_france_processed.csv")

# Calculating mean price per square meter per department
mean_price_by_department = df.groupby('department')['average_price_per_year_department'].mean().round(2)

print ('The mean price for each department is')
print(mean_price_by_department)

# Calculating mean price per square meter in France
mean_price_france = df['average_price_per_year_department'].mean().round(2)

print ('The global mean price in France is')
print(mean_price_france)

# Calculate the median price
median_price_france = df['average_price_per_year_department'].median().round(2)

print ('The global median price in France is')
print(median_price_france)

# Create deviation bar graph for 5 most expensive and 5 least expensive departments 

# Sort DataFrame by 'average_price_per_year_department'

sorted_df = mean_price_by_department.reset_index().sort_values(by='average_price_per_year_department')

# Extract the top 5 and bottom 5 departments
top_5_expensive = sorted_df.tail(5)
bottom_5_expensive = sorted_df.head(5)

print('top 5')
print(top_5_expensive)
print('bottom 5')
print(bottom_5_expensive)

# Calculate deviation from global mean price
top_5_expensive['deviation'] = top_5_expensive['average_price_per_year_department'] - mean_price_france
bottom_5_expensive['deviation'] = bottom_5_expensive['average_price_per_year_department'] - mean_price_france

# Combine the two dataframes
combined_df = pd.concat([top_5_expensive, bottom_5_expensive])

# Plot the bar graph
plt.figure(figsize=(10, 6))
plt.bar(combined_df['department'], combined_df['deviation'], color=['red' if deviation > 0 else 'blue' for deviation in combined_df['deviation']])
plt.xlabel('Department')
plt.ylabel('Deviation from Global Mean Price')
plt.title('Deviation of Average Price from Global Mean Price for Top 5 Most Expensive and Bottom 5 Least Expensive Departments')
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='black', linewidth=0.5)  
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
````

- Checking id the housing prices increased consistently every year in France, and if this is true for all departments

```python

#Did the housing price increase in France from 2018 to 2022? What about individual departments?

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame
df = pd.read_csv(r"D:\Documents\DataAnalysis\Housing_Market_France\csv_files\housing_market_france_processed.csv")

# Finding the national average price per year and plotting a simple bar chart        
average_price_by_year = df.groupby('year')['average_price_per_year_department'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=average_price_by_year, x='year', y='average_price_per_year_department')
plt.title('Average Price per Square Meter in France')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.show()

# Finding if all departments followed the national trend and had their price increase consistently from 2018 to 2020
# Group by 'year' and 'department', then calculate the mean
average_price_by_year_department = df.groupby(['year', 'department'])['average_price_per_year_department'].mean().unstack()

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

# Check if there are any departments where prices didn't increase
if not decreased_departments:
    print("Housing prices increased in all French departments year too year between 2018 and 2022.")
else:
    print("Housing prices didn't increase for at least one given year in the following departments (2018-2022):")
    print(decreased_departments)
    for department in decreased_departments:
        print(f"\nPrices in {department}:")
        department_data = df[df['department'] == department]
        print(department_data[['year', 'average_price_per_year_department']])

# Filter out departments with consistent price increases over the years
filtered_departments = df[~df['department'].isin(decreased_departments)]['department'].unique()

# Select the top 5 departments with the biggest increase in housing prices over the period from 2018 to 2022
top_increasing_departments = (
    average_price_change.nlargest(5, 'average_price_change_2018_to_2019')['department'].tolist() +
    average_price_change.nlargest(5, 'average_price_change_2019_to_2020')['department'].tolist() +
    average_price_change.nlargest(5, 'average_price_change_2020_to_2021')['department'].tolist() +
    average_price_change.nlargest(5, 'average_price_change_2021_to_2022')['department'].tolist()
)

# Calculate the mean price for all departments for each year
mean_prices_all_departments = df.groupby('year')['average_price_per_year_department'].mean()

# Visualize the percentage change in housing prices for selected departments:
plt.figure(figsize=(12, 8))
for department in decreased_departments + top_increasing_departments:
    if department in filtered_departments:
        department_data = df[df['department'] == department]
        percent_change = (department_data.groupby('year')['average_price_per_year_department'].mean().pct_change() * 100).fillna(0)
        plt.plot(percent_change.index, percent_change.values, marker='o', label=department)
percent_change_mean = (mean_prices_all_departments.pct_change() * 100).fillna(0)

plt.plot(percent_change_mean.index, percent_change_mean.values, marker='o', color='black', linestyle='--', label='Mean (All Departments)')
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5) 
plt.title('Percentage Change in Housing Prices in selected departments')
plt.xlabel('Year')
plt.ylabel('Percentage Change')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
plt.grid(True)
plt.tight_layout()
plt.show()

```


### Using Machine Learning to Predict Expected Increases in Price
A Linear Regression model was implemented to predict housing prices for 2023 based on historical data. The model's predictions were evaluated using metrics such as Root Mean Squared Error (RMSE) and the coefficient of determination (R^2).

```python

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

```

The top 10 departments returned are the following: 

|   year | department           | predicted_average_price | average_price_per_year_department | price_increase |
|-------:|:---------------------|------------------------:|----------------------------------:|----------------|
|   2023 | Hautes-Pyrenees      |                 2002.86 |                           1721.25 |          281.61 |
|   2023 | Haute-Marne          |                 1295.80 |                           1014.75 |          281.05 |
|   2023 | Indre                |                 1398.49 |                           1128.92 |          269.57 |
|   2023 | Ardennes             |                 1532.26 |                           1278.42 |          253.84 |
|   2023 | Territoire de Belfort|                 2069.79 |                           1816.17 |          253.62 |
|   2023 | Correze              |                 1701.49 |                           1459.67 |          241.82 |
|   2023 | Meuse                |                 1325.22 |                           1086.75 |          238.47 |
|   2023 | Allier               |                 1479.59 |                           1244.75 |          234.84 |
|   2023 | Creuse               |                 1136.25 |                            902.58 |          233.67 |
|   2023 | Cher                 |                 1509.17 |                           1280.58 |          228.59 |

Root Mean Squared Error (RMSE) on test set: 288.5808959841751
Coefficient of determination (R^2) on test set: 0.9698584025766345

### Combining All Previous Code to Find the Optimal Departments to Invest in
Finally, all previous analyses were combined to identify the optimal departments for investment based on specific criteria. The process involved filtering departments based on their price compared to the national average, consistent price increases, and predicted price increases for 2023.

```python
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

# From these departments, figure out which ones have seen a constant increase (from 2020 to 2021 and 2021 to 2022)

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
    print("Top 10 departments with the biggest increase in average price for 2023 among departments below the national average and with constant increase:")
    print(departments_2023_sorted[departments_2023_sorted['department'].isin(departments_of_interest)].head(10))
else:
    print("No departments meet all the criteria.")
```
Based on this, two departments meet all our criteria and are the most optimal departments to invest in: 

|   year | department      | predicted_average_price | average_price_per_year_department | price_increase |
|-------:|:----------------|------------------------:|----------------------------------:|----------------:|
|   2023 | Hautes-Pyrenees |                 2002.86 |                           1721.25 |          281.61 |
|   2023 | Allier          |                 1479.59 |                           1244.75 |          234.84 |
