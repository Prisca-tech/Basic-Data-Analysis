# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load iris dataset from sklearn and convert to DataFrame
    iris_raw = load_iris()
    iris = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)
    iris['species'] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)
except Exception as e:
    print("Error loading dataset:", e)
    exit()

# Display first few rows
print("First 5 rows:")
print(iris.head())

# Check data types and missing values
print("\nData types:")
print(iris.dtypes)
print("\nMissing values per column:")
print(iris.isnull().sum())

# Task 2: Basic Data Analysis

# Basic stats of numerical columns
print("\nDescriptive statistics:")
print(iris.describe())

# Group by species and compute mean sepal length
species_group = iris.groupby('species')['sepal length (cm)'].mean()
print("\nAverage Sepal Length by Species:")
print(species_group)

# Observation:
# - Setosa has the smallest average sepal length
# - Versicolor and Virginica have progressively larger average sepal lengths

# Task 3: Data Visualization

# 1. Line chart: Sepal length trend by sample index per species 
for species in iris['species'].unique():
    subset = iris[iris['species'] == species]
    plt.plot(subset.index, subset['sepal length (cm)'], marker='o', label=species)
plt.title('Sepal Length Trend by Sample Index per Species')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar chart: Average petal length per species
plt.figure(figsize=(7, 5))
sns.barplot(x='species', y='petal length (cm)', data=iris)
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()

# 3. Histogram: Distribution of sepal width
plt.figure(figsize=(7, 5))
sns.histplot(iris['sepal width (cm)'], bins=15, kde=True, color='skyblue')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Scatter plot: Sepal length vs Petal length colored by species
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=iris, s=70)
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.show()
