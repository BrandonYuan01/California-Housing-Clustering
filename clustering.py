from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer

from CustomKmeans import CustomKMeans

#  loading and basic info on data
path = "housing.csv"
housing = pd.read_csv(path)

data_size = os.path.getsize(path) / 1024 / 1024
num_entries = housing.shape[0]
num_features = housing.shape[1]
categorical_data = housing.select_dtypes(include=['object']).columns.tolist()
missing_data = housing.isnull().sum()
stats = housing.describe()

print(f"{data_size} mb")
print(f"number of entries: {num_entries}")
print(f"number of features: {num_features}")
print("The data has categorical data:")
print(categorical_data)
print("The data has missing data: \n")
print(missing_data)
print(stats)

#  visualizing and plotting correlations
correlation = housing.corr(numeric_only = True)
plt.figure(figsize=(10, 10))
sb.heatmap(correlation, annot=True, fmt=".2f")
plt.title("Feature Correlations")
plt.show()

scatter_matrix(housing, figsize=(10, 8))

#  data preproccessing and cleaning
X_train = housing

categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(include=['number']).columns

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('standard_scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot_encoder', OneHotEncoder())
])

pipeline = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

X_train = pipeline.fit_transform(X_train)

k_values = range(2, 11)
sse_values = []
for k in k_values:
    kmeans = CustomKMeans(k, distance_metric="sup")  # Use the new class
    centroids, clusters, sse = kmeans.fit(X_train)
    sse_values.append(sse)

    print(f'For k={k}: SSE={sse}')

    cluster_points = X_train[clusters == k - 2]
    cluster_mean = np.mean(cluster_points, axis=0)
    cluster_std = np.std(cluster_points, axis=0)
    print(f'For k={k}: Mean={cluster_mean}, Std Dev={cluster_std}')

    # Visualizing clustering results
    plt.scatter(X_train[:, 0], X_train[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title(f'Clustering of California Housing (k={k})')
    plt.legend()
    plt.show()

plt.plot(k_values, sse_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Plot for Optimal k')
plt.show()
