from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_circles
from sklearn.datasets.samples_generator import make_moons
from sklearn.datasets.samples_generator import make_s_curve
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


n=1500
X, y = make_blobs(n_samples = n,
                  n_features = 3,
                  random_state = 42)
plt.scatter(X[:,0], X[:,1])

kmeans = KMeans(n_clusters = 3, random_state = 0)
kmeans.fit(X)
plt.scatter(X[:, 0], X[:, 1],c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'yellow')
plt.show()