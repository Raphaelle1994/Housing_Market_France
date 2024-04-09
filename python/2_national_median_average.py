# Find the national median and average 

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

