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

# Finding if all departments followed the national trend and had their price increase for both 2020 to 2021 and 2021 to 2022
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