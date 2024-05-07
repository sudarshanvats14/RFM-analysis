import pandas as pd
import sklearn
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
import matplotlib.pyplot as plt
import seaborn as sns
from clustering import rfm, rfm_scaled



mergings = linkage(rfm_scaled, method="single", metric='euclidean')
#dendrogram(mergings)
# plt.show()
# plt.savefig('single_linkage.png')


mergings = linkage(rfm_scaled, method="complete", metric='euclidean')
# dendrogram(mergings)
# plt.show()
# plt.savefig('complete_linkage.png')


mergings = linkage(rfm_scaled, method="average", metric='euclidean')
# dendrogram(mergings)
# plt.show()
# plt.savefig('average_linkage.png')


cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
cluster_labels


rfm['Cluster_Labels'] = cluster_labels
print(rfm.head)

boxplpot = sns.boxplot(x='Cluster_Labels', y='Recency', data=rfm)
fig = boxplpot.get_figure()
fig.savefig('boxplot_6.png')  
