import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Load dataset
df = pd.read_csv('exercice2/dataset.csv')

# Define numerical and categorical features
numerical_features = ['age']
categorical_features = ['favorite music style']

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
encoded_features_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Concatenate numerical and encoded categorical features
df_encoded = pd.concat([df[numerical_features], encoded_features_df], axis=1)

# Define weights for numerical features
numerical_weights = {'age': 0.5}

# Define weights for categorical features and categories
categorical_weights = {
    'favorite music style': {'trap': 0.3, 'hiphop': 0.3, 'metal': 0.2, 'rock': 0.2, 'classical': 0.2, 'rap': 0.2, 'jazz': 0.2, 'other': 0.2, 'technical death metal': 0.2}
}

# Define custom dissimilarity metric
def custom_dissimilarity(sample1, sample2, numerical_weights, categorical_weights):
    # Euclidean distance for numerical features
    numerical_distance = np.sqrt(np.sum((sample1['age'] - sample2['age']) ** 2)) * numerical_weights['age']
    
    # Hamming distance for categorical features
    categorical_distance = 0
    for feature in categorical_weights:
        if feature in sample1 and feature in sample2:  # Check if feature exists in both samples
            if sample1[feature] != sample2[feature]:
                categorical_distance += categorical_weights[feature]
    
    # Combine distances
    dissimilarity = numerical_weights['age'] * numerical_distance + categorical_distance
    return dissimilarity

# Calculate dissimilarity between each pair of samples
dissimilarities = np.zeros((len(df_encoded), len(df_encoded)))
for i, sample1 in df_encoded.iterrows():
    for j, sample2 in df_encoded.iterrows():
        dissimilarities[i, j] = custom_dissimilarity(sample1, sample2, numerical_weights, categorical_weights)
np.save('exercice2/dissimilarity_matrix.npy', dissimilarities)

# Compute mean and standard deviation of the dissimilarity distribution
mean_dissimilarity = np.mean(dissimilarities)
std_dissimilarity = np.std(dissimilarities)

print("Mean Dissimilarity:", mean_dissimilarity)
print("Standard Deviation of Dissimilarity:", std_dissimilarity)

plt.figure(figsize=(10, 8))
sns.heatmap(dissimilarities, cmap="YlGnBu", square=True)
plt.title("Dissimilarity Heatmap (Age vs. Favorite Music Style)")
plt.xlabel("Sample Index")
plt.ylabel("Sample Index")
plt.show()

# Create scatter plot of dissimilarities vs. age
plt.figure(figsize=(10, 6))
plt.scatter(df_encoded['age'], dissimilarities.mean(axis=1), c='b', alpha=0.5)
plt.title('Dissimilarities vs. Age')
plt.xlabel('Age')
plt.ylabel('Mean Dissimilarity')
plt.grid(True)
plt.show()