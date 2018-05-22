import sklearn
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn import datasets
from sklearn import svm
from sklearn.mixture import GMM
from sklearn import preprocessing
from sklearn.cluster import KMeans




print("____________________Assignment 2__________________________")

data = pd.read_csv('seeds_dataset.txt', sep="\t", error_bad_lines=False)

kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

data['clusters'] = labels

colors = ["g.", "r.", "b."]



#print(data.groupby(['clusters']).mean())
#print(centroids)



