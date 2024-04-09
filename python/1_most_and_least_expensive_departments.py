
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
