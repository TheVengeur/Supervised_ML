import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(url, names=column_names)

# Display the first few rows of the dataset
print(iris_df.head())

# Summary statistics
print(iris_df.describe())

# Check for missing values
print(iris_df.isnull().sum())

# Visualize the distribution of classes
sns.countplot(x='species', data=iris_df)
plt.title('Distribution of Iris Species')
plt.show()

# Visualize the distribution of numerical features
sns.pairplot(iris_df, hue='species')
plt.show()
