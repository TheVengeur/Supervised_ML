import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Sample dataset (replace this with your actual dataset)
data = {
    'age': [25, 30, 35, 40],
    'height': [160, 170, 175, 180],
    'job': ['Engineer', 'Doctor', 'Teacher', 'Artist'],
    'city': ['New York', 'Los Angeles', 'Chicago', 'San Francisco'],
    'favorite_music_style': ['Pop', 'Rock', 'Jazz', 'Classical']
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Define numerical and categorical features
numerical_features = ['age', 'height']
categorical_features = ['job', 'city', 'favorite_music_style']

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
encoded_features_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Concatenate numerical and encoded categorical features
df_encoded = pd.concat([df[numerical_features], encoded_features_df], axis=1)

# Define dissimilarity metric
def custom_dissimilarity(sample1, sample2, weights=None):
    # Euclidean distance for numerical features
    numerical_distance = np.sqrt(np.sum((sample1[numerical_features] - sample2[numerical_features]) ** 2))
    
    # Hamming distance for categorical features
    categorical_distance = np.sum(sample1[encoded_feature_names] != sample2[encoded_feature_names])
    
    # Combine distances
    if weights is None:
        weights = {'numerical': 1, 'categorical': 1}
    dissimilarity = weights['numerical'] * numerical_distance + weights['categorical'] * categorical_distance
    return dissimilarity

# Calculate dissimilarity between each pair of samples
dissimilarities = np.zeros((len(df), len(df)))
for i, sample1 in df_encoded.iterrows():
    for j, sample2 in df_encoded.iterrows():
        dissimilarities[i, j] = custom_dissimilarity(sample1, sample2)

# Print dissimilarity matrix
print("Dissimilarity Matrix:")
print(dissimilarities)