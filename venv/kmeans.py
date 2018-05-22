import sklearn
import numpy
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

print("____________________Assignment 2__________________________")
# Read dataset
data = pd.read_csv('seeds_dataset.txt', sep="\t", error_bad_lines=False)

# Make blobs
data, y = make_blobs(n_samples=180,
                     n_features=7,
                     random_state=42)
# Scatter data
plt.scatter(data[:, 0], data[:, 1])

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=0, max_iter=300).fit(data)

# Labels
labels = kmeans.labels_
# Centroids, scatter and color yellow
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=50, c='yellow')

# Show visualization
plt.title("KMeans Clusters")
plt.show()
