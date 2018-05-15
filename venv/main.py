import sklearn
import numpy
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import k_means
#digits = datasets.load_digits()

#clf = svm.SVC(gamma=0.001, C=100)
#x,y = digits.data[:-10],digits.target[:-10]

#clf.fit(x,y)

#print("Prediction of last:",clf.predict(digits.data[[-2]]))
#plt.imshow(digits.images[-2], cmap = plt.cm.gray_r, interpolation="nearest")
#plt.show()


print("____________________Assignment 2_________________________")

with open("seeds_dataset.txt","r") as myFile:
    data=myFile.read()
