import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster import hierarchy

# Importing the dataset
dataset = pd.read_csv(r"E:\Download\Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

import scipy.cluster.hierarchy as sch

dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('euclidean distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,linkage='ward')
y_hc=hc.fit_predict(X)

# visulization the clusters

plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='cluster 1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='cluster 2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='cluster 3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='cluster 4')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='magenta',label='cluster 5')
plt.title('clusters of customers')
plt.xlabel('annual income(r$)')
plt.ylabel('spending score(1-100)')
plt.legend()
plt.show()

dataset['cluster']=y_hc