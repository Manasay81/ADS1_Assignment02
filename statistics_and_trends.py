# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset 
file_path = 'heart_failure_clinical_records_dataset.csv' 
heart_failure_df = pd.read_csv(file_path)

# Displaying the first few rows of the dataset
print(heart_failure_df.head())

# Histogram for Age Distribution
plt.figure(figsize=(8, 6))
plt.hist(heart_failure_df['age'], bins=10, color='blue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Scatter Plot for Age vs Serum Creatinine
plt.figure(figsize=(8, 6))
plt.scatter(heart_failure_df['age'], heart_failure_df['serum_creatinine'], color='red')
plt.title('Age vs Serum Creatinine')
plt.xlabel('Age')
plt.ylabel('Serum Creatinine')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))

# Categorizing 'Age' into groups
heart_failure_df['Age Group'] = pd.cut(heart_failure_df['age'], bins=[40, 50, 60, 70, 80, 90, 100], 
                                       labels=['40s', '50s', '60s', '70s', '80s', '90s'])

# Creating a violin plot for Platelets by Age Group
plt.figure(figsize=(10, 6))
sns.violinplot(x='Age Group', y='platelets', data=heart_failure_df)
plt.title('Platelets by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Platelets (kiloplatelets/mL)')
plt.show()

# Heatmap for Correlation Matrix
correlation_matrix = heart_failure_df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis')
plt.title('Correlation Matrix')
plt.show()

# Displaying basic statistics
describe_stats = heart_failure_df.describe()
print(describe_stats)
numeric_cols = heart_failure_df.select_dtypes(include=[np.number]).columns
skewness = heart_failure_df[numeric_cols].skew().to_string()
print("Skewness:\n", skewness)
kurtosis = heart_failure_df[numeric_cols].kurtosis().to_string()
print("\nKurtosis:\n", kurtosis)
