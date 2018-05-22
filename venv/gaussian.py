import sklearn
import numpy as np

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture

print("____________________Assignment 2__________________________")
# Read dataset
data = pd.read_csv('seeds_dataset.txt', sep="\t", error_bad_lines=False)

gmm = GaussianMixture(n_components=3, covariance_type='full', tol=0.001)
gmm.fit(data)

labels = gmm.predict(data)

plt.scatter(data.values[:, 0], data.values[:, 1], c=labels, s=40, cmap='viridis')
plt.show()


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


plot_gmm(gmm, data.values)

######################################################
