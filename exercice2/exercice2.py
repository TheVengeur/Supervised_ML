import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('exercice2/dataset.csv')

# Define numerical and categorical features
numerical_features = ['age', 'height']
categorical_features = ['job', 'city', 'favorite music style']

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
encoded_features_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Concatenate numerical and encoded categorical features
df_encoded = pd.concat([df[numerical_features], encoded_features_df], axis=1)

# Define weights for numerical features
numerical_weights = {'age': 0.5, 'height': 0.5}

# Define weights for categorical features and categories
categorical_weights = {
    'job': {'designer': 0.2, 'fireman': 0.3, 'teacher': 0.3, 'artist': 0.2, 'doctor': 0.2, 'painter': 0.2, 'developper': 0.2, 'engineer': 0.2},
    'city': {'paris': 0.25, 'marseille': 0.25, 'madrid': 0.25, 'lille': 0.25},
    'favorite music style': {'trap': 0.3, 'hiphop': 0.3, 'metal': 0.2, 'rock': 0.2, 'classical': 0.2, 'rap': 0.2, 'jazz': 0.2, 'other': 0.2, 'technical death metal': 0.2}
}

# Define custom dissimilarity metric
def custom_dissimilarity(sample1, sample2, numerical_weights, categorical_weights):
    # Euclidean distance for numerical features
    numerical_distance = np.sqrt(np.sum((sample1['age'] - sample2['age']) ** 2)) * numerical_weights['age'] + \
                          np.sqrt(np.sum((sample1['height'] - sample2['height']) ** 2)) * numerical_weights['height']
    
    # Hamming distance for categorical features
    categorical_distance = 0
    for feature in categorical_weights:
        if feature in sample1 and feature in sample2:  # Check if feature exists in both samples
            if sample1[feature] != sample2[feature]:
                categorical_distance += categorical_weights[feature]
    
    # Combine distances
    dissimilarity = numerical_weights['age'] * numerical_distance + \
                    numerical_weights['height'] * numerical_distance + \
                    categorical_distance
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
plt.title("Dissimilarity Heatmap")
plt.xlabel("Sample Index")
plt.ylabel("Sample Index")
plt.show()