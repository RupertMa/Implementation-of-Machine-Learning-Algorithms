'''@author: Yibo Ma'''

import numpy as np


def loadDataset():
	with open('clusters.txt') as f:
		data=[]
		for entry in f.readlines():
			data.append(map(float,entry.strip().split(',')))
        data=np.array(data)
        return data

def distEclu(vA,vB):
	return np.sqrt(np.sum((vA-vB)**2))

def randCent(data,k):
	n,d=data.shape
	centroids=np.zeros((k,d))
	centroids=data[np.random.randint(0,n,size=k)]
	return centroids


def kMeans(data,k=3):
	clusters=np.zeros((data.shape[0],2))
	centroids=randCent(data,k)
	clusterChanged=True
	while clusterChanged:
		clusterChanged=False
		for entryIndex in range(len(data)):
			minDis=np.inf
			for centIndex in range(k):
				dist=distEclu(data[entryIndex],centroids[centIndex])
				if dist<minDis:
					minDis=dist
					minIndex=centIndex
			if clusters[entryIndex][0]!=minIndex: clusterChanged=True
			clusters[entryIndex]=(minIndex,minDis**2)
		for centIndex in range(k):
			centroids[centIndex,:]=np.average(data[np.nonzero(clusters[:,0]==centIndex)[0]],axis=0)
	return centroids,clusters


def plotKmeans(centroids,clusters,data):
	import matplotlib.pyplot as plt
	color=['aqua','coral','seagreen']
	k=len(centroids)
	for centIndex in range(k):
		x=data[np.nonzero(clusters[:,0]==centIndex)[0]][:,0]
		y=data[np.nonzero(clusters[:,0]==centIndex)[0]][:,1]
		plt.scatter(x,y,c=color[centIndex])	
	x=[centroid[0] for centroid in centroids] 
	y=[centroid[1] for centroid in centroids]
	plt.scatter(x,y,c='navy',s=50)
	plt.title('K-means Visualization')
	plt.show()


def divisiveClustering(data,k):
	n,d=data.shape
	clusters=np.zeros((n,2))
	centroid=np.average(data,axis=0)
	for j in range(n):
		clusters[j,1]=distEclu(data[j,:],centroid)**2
	centroids=[centroid]
	while len(centroids)<k:
		sse=np.inf
		index=None
		for i in range(len(centroids)):
			subdata=data[np.nonzero(clusters[:,0]==i)[0],:]
			unselectedclst=clusters[np.nonzero(clusters[:,0]!=i)[0],:]
			subcentroids,subclusters=kMeans(subdata,2)
			newsse=np.sum(subclusters[:,1])+np.sum(unselectedclst[:,1])
			if newsse<sse:
				sse=newsse
				newcentroids=subcentroids
				newclusters=subclusters
				index=i
		newclusters[np.nonzero(newclusters[:,0]==1)[0],0]=len(centroids)
		newclusters[np.nonzero(newclusters[:,0]==0)[0],0]=index
		clusters[np.nonzero(clusters[:,0]==index)[0],:]=newclusters
		centroids[index]=newcentroids[0]
		centroids.append(newcentroids[1])
	return centroids,clusters
		

#np.random.seed(3)
data=loadDataset()

centroids,clusters=divisiveClustering(data,3)
print "======Centriods======="
for centroid in centroids:
	print centroid
plotKmeans(centroids,clusters,data)







