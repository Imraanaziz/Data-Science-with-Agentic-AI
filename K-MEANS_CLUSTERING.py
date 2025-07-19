import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 


dataset = pd.read_csv(r"E:\Download\Mall_Customers.csv")
x = dataset.iloc[:, [3, 4]].values 

from sklearn.cluster import KMeans 

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of cluster')
plt.ylabel('Wcss')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,linkage='ward')
y_hc=hc.fit_predict(x)

# visulization the clusters

plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c='red',label='cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c='blue',label='cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c='green',label='cluster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c='cyan',label='cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c='magenta',label='cluster 5')
plt.title('clusters of customers')
plt.xlabel('annual income(r$)')
plt.ylabel('spending score(1-100)')
plt.legend()
plt.show()

dataset['cluster']=y_hc 


