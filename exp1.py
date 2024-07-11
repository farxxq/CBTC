import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = r'c:/Users/Farooq/OneDrive/Desktop/Internship/CipherByte DataScience/Unemployment in India.xlsx'
unemployment_data = pd.read_excel(file_path, sheet_name='Unemployment in India')

# Data Preprocessing
# Convert Date column to datetime format
unemployment_data['Date'] = pd.to_datetime(unemployment_data['Date'], dayfirst=True)

# Handling missing values (if any)
unemployment_data = unemployment_data.dropna()

# Trend Analysis
# Plotting the overall trend of unemployment rates over time
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=unemployment_data, errorbar=None)
plt.title('Overall Trend of Unemployment Rate in India')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.show()

# Regional Analysis
# Plotting unemployment rates for different regions
plt.figure(figsize=(14, 8))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', hue='Region', data=unemployment_data, errorbar=None)
plt.title('Unemployment Rate by Region in India')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
plt.show()

# Impact of Covid-19
# Defining the date when Covid-19 started impacting the data (March 2020)
covid_start_date = pd.to_datetime('2020-03-01')

# Plotting unemployment rates before and after Covid-19
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=unemployment_data[unemployment_data['Date'] < covid_start_date], label='Before Covid-19', errorbar=None)
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=unemployment_data[unemployment_data['Date'] >= covid_start_date], label='During Covid-19', errorbar=None)
plt.title('Unemployment Rate Before and During Covid-19')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.legend()
plt.show()

# Save the processed data (optional)
unemployment_data.to_csv('Processed_Unemployment_Data.csv', index=False)
