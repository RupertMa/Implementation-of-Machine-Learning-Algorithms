import numpy as np

def loadDataSet():
	data=np.loadtxt("classification.txt",delimiter=",")
	X=np.ones((data.shape[0],4))
	X[:,1:]=data[:,0:3] #input values Xj for each example j
	D=data[:,4] #desired output Dj
	return X,D

def pocket(X,D,alpha=0.01,iteration=7000):
	W=np.random.random((4,)) #weights of each variable
	NumIter_MissPoints=[]
	for index in range(iteration):
		Y=np.sum(W*X,axis=1)
		Y[np.nonzero(Y>=0)]=1.
		Y[np.nonzero(Y<0)]=-1.
		ViolatedCons=X[np.nonzero(Y!=D)]
		ViolatedD=D[np.nonzero(Y!=D)]
		NumIter_MissPoints.append([index,len(ViolatedCons),W])
		if len(ViolatedCons)==0:
			break
		i=np.random.randint(len(ViolatedCons))
		W=W+alpha*ViolatedD[i]*ViolatedCons[i]		
	return NumIter_MissPoints

def plot(NumIter,MissPoints):
	import matplotlib.pyplot as plt
	plt.plot(NumIter,MissPoints,linewidth=0.1)
	plt.xlabel("The number of iteration")
	plt.ylabel("The number of misclassified points")
	plt.show()


def Accuracy(W,X,D):
	Y=np.sum(W*X,axis=1)
	Y[np.nonzero(Y>=0)]=1.
	Y[np.nonzero(Y<0)]=-1.
	accuracy=sum(Y==D)/float(len(D))
	return accuracy

def main():
	X,D=loadDataSet()
	NumIter_MissPoints=pocket(X,D)
	BestW=NumIter_MissPoints[np.argsort([i[1] for i in NumIter_MissPoints])[0]]
	print "In the %d iteraion, we got the least number of misclassified points %d " %(BestW[0],BestW[1])
	print "And the weights are:",BestW[2]
	print "Accuracy: ",Accuracy(BestW[2],X,D)
	NumIter=[i[0] for i in NumIter_MissPoints]
	MissPoints=[i[1] for i in NumIter_MissPoints]
	plot(NumIter,MissPoints)

if __name__ == "__main__":
    main()