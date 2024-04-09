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

### Exploring the Data and Creating Visualizations using Python
🔗[Link to Python](https://github.com/Raphaelle1994/Housing_Market_France/tree/main/python)

Exploratory Data Analysis (EDA) was performed using Python, focusing on identifying the most and least expensive departments regardless of the year and calculating national median and average prices. Visualizations such as bar plots were created to present the findings effectively. 

### Finding If the Housing Price Increased in France from 2018 to 2022
Statistical analysis was conducted to determine if housing prices increased in all French departments from 2018 to 2022. The analysis included identifying departments with consistent price increases and visualizing the percentage change in housing prices for selected departments.

### Using Machine Learning to Predict Expected Increases in Price
A Linear Regression model was implemented to predict housing prices for 2023 based on historical data. The model's predictions were evaluated using metrics such as Root Mean Squared Error (RMSE) and the coefficient of determination (R^2).

### Combining All Previous Code to Find the Optimal Departments to Invest in
Finally, all previous analyses were combined to identify the optimal departments for investment based on specific criteria. The process involved filtering departments based on their price compared to the national average, consistent price increases, and predicted price increases for 2023.
