import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
import matplotlib.pyplot as plt
import seaborn as sns
from Data import rfm_scaled, rfm

kmeans = KMeans(n_clusters=4, max_iter=50, random_state=42)
kmeans.fit(rfm_scaled)

kmeans.labels_


ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_scaled)
    
    ssd.append(kmeans.inertia_)
    
# plt.plot(ssd)
# plt.savefig('ssd.png')


range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_scaled)
    
    clusters = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(rfm_scaled, clusters)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))



kmeans = KMeans(n_clusters=3, max_iter=50, random_state= 42)
kmeans.fit(rfm_scaled)

kmeans.labels_


rfm['Cluster_Id'] = kmeans.labels_
print(rfm.head)

# boxplpot = sns.boxplot(x='Cluster_Id', y='Monetary', data=rfm)
# fig = boxplpot.get_figure()
# fig.savefig('boxplot_1.png')  

# boxplot = sns.boxplot(x='Cluster_Id', y='Frequency', data=rfm)
# fig = boxplot.get_figure()
# fig.savefig('boxplot_2.png')


# boxplot = sns.boxplot(x='Cluster_Id', y='Recency', data=rfm)
# fig = boxplot.get_figure()
# fig.savefig('boxplot_3.png')




